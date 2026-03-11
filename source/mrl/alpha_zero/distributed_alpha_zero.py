from typing import Callable
import os
import sys
import asyncio
from asyncio.subprocess import Process
from dataclasses import dataclass
import torch
from mrl.alpha_zero.model_trainer import ModelTrainer
from mrl.alpha_zero.model_evaluation  import EvaluationLoop
from mrl.alpha_zero.model_updater import ModelUpdater
from mrl.configuration.alpha_zero_configuration import HDF5AlphaZeroConfiguration


@dataclass
class CollectorProcess:
    process: "asyncio.subprocess.Process"
    stdout_task: asyncio.Task
    stdin_task: asyncio.Task


class FileCollection:

    def __init__(self, queue):
        self.queue = queue

    def __aiter__(self):
        return self

    async def __anext__(self):
        file = await self.queue.get()
        if file is None:
            raise StopAsyncIteration()
        return file


class ExperienceCollector:

    def __init__(self, config: HDF5AlphaZeroConfiguration):
        self.config = config
        self.file_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.processes: list[Process] = []
        self.wait_completion_task: asyncio.Task | None = None
        self.writers: list[asyncio.StreamWriter] = []
        self.writer_lock = asyncio.Lock()

    async def __aenter__(self):
        self.writers = []
        self.file_queue = asyncio.Queue()
        server = await asyncio.start_server(
            self._handle_client,
            self.config.server_hostname,
            self.config.server_port
        )
        asyncio.create_task(self._serve(server))
        self.processes = await asyncio.gather(*(
            self._spawn_hd5f_process(i)
            for i in range(self.config.collector.number_of_processes)
        ))
        self.wait_completion_task = asyncio.create_task(
            self._wait_completion(server)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        assert self.wait_completion_task is not None
        await self.wait_completion_task

    def get_files(self):
        return FileCollection(self.file_queue)

    async def _serve(self, server):
        async with server:
            await server.serve_forever()

    async def _handle_client(self, reader, writer):
        async with self.writer_lock:
            self.writers.append(writer)
        await self._receive(reader)
        writer.close()
        await writer.wait_closed()
        async with self.writer_lock:
            self.writers.remove(writer)

    async def _spawn_hd5f_process(self, index):
        cmd = [
            sys.executable, '-m', 'mrl.alpha_zero.self_play_process',
            '--config-file', self.config.config_file_path,
            '--process-index', str(index)
        ]
        return await asyncio.create_subprocess_exec(*cmd)

    async def _receive(self, reader):
        while True:
            data = await reader.readline()
            if not data:
                break
            message = data.decode().strip()
            if message == "Done":
                break
            if message.endswith(".h5"):
                await self.file_queue.put(message)

    async def _wait_completion(self, server):
        await asyncio.gather(*(
            process.wait() for process in self.processes
        ))
        await self.file_queue.put(None)
        server.close()
        await server.wait_closed()

    async def _broadcast(self, message):
        for writer in self.writers:
            try:
                writer.write((message + "\n").encode())
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass

        for writer in self.writers:
            try:
                await writer.drain()
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass

    async def notify_better_model(self):
        await self._broadcast("A better model is available")


class DistributedAlphaZero:

    def __init__(self, config: HDF5AlphaZeroConfiguration):
        self.config = config
        self.model = config.oracle
        if not isinstance(self.model, torch.nn.Module):
            raise TypeError(
                "Oracle class {type(self.model)} is not a torch Module. "
            )
        self.model_trainer = ModelTrainer(self.model, config.trainer)

        self.model_updater = ModelUpdater(config.game, config)

        self.report_generator: Callable[[], None]
        if config.report_generator is None:
            self.report_generator = lambda: None
        else:
            self.report_generator = EvaluationLoop(
                config.game,
                self.model,
                config.report_generator
            )

    def train(self):
        asyncio.run(self._train())

    async def _train(self):
        self.model.save(self.config.oracle_file_path)
        async with ExperienceCollector(self.config) as experience_collector:
            async for file_path in experience_collector.get_files():
                better_model = self._process_once(file_path)
                if better_model:
                    await experience_collector.notify_better_model()

    def _process_once(self, file_path):
        self.model_trainer.train_from_hdf5(file_path)
        os.remove(file_path)
        better_model = self.model_updater.save_if_better(self.model)
        if better_model:
            self.report_generator()
        return better_model
