import argparse
import os
import time
import asyncio
import yaml
from mrl.alpha_zero.context import HDF5AlphaZeroContext, UnionAlphaZeroContext
from mrl.alpha_zero.experience_collector import SingleHDF5Collector
from mrl.configuration.alpha_zero_configuration import AlphaZeroConfiguration
from mrl.configuration.alpha_zero_runner_factory import AlphaZeroRunnerFactory


async def main(args = None):
    args = parse_arguments(args)
    with open(args.config_file, "r", encoding = 'UTF-8') as yaml_file:
        data = yaml.safe_load(yaml_file)
    configuration = AlphaZeroConfiguration.model_validate(
        data | {'config_file_path': args.config_file}
    )
    play_process = PlayProcess(AlphaZeroRunnerFactory().make_context(configuration))
    await play_process.run(args.process_index, args.verbose)


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type = str, required = True)
    parser.add_argument('--process-index', type = int, required = True)
    parser.add_argument('--verbose', action = "store_true")
    return parser.parse_args(args)


class PlayProcess:

    def __init__(self, context: UnionAlphaZeroContext):
        if not isinstance(context, HDF5AlphaZeroContext):
            raise TypeError(
                "Self-play processes require an HDF5 alpha zero configuration."
            )
        self.context = context
        self.model = context.oracle
        self.model_time = None
        self.collector = SingleHDF5Collector(context.game, self.model, context.collector)

    async def run(self, process_index, verbose):
        if not (self.context.workspace_path.exists() and self.context.workspace_path.is_dir()):
            raise RuntimeError(
                f"Workspace {self.context.workspace_path} does not exist or "
                "is not a directory. Please ensure to create it prior to "
                "creating a self play process."
            )
        epochs_per_process = (
            self.context.number_of_epochs // self.context.collector.number_of_processes
        )
        if (self.context.number_of_epochs % self.context.collector.number_of_processes) > 0:
            epochs_per_process += 1
        async with ModelLoader(self.model, self.context) as loader:
            for epoch in range(epochs_per_process):
                await asyncio.sleep(0)
                if verbose:
                    await self._notify_epoch(epoch, loader)
                file_name = f"{self.context.hdf5_path_prefix}_{process_index}_{epoch}.h5"
                file_path = f"{self.context.workspace_path}/{file_name}"
                self.collector.collect(file_path)
                await loader.notify(file_path)
            await loader.notify("Done")

    async def _notify_epoch(self, epoch, loader):
        await loader.notify(
            f"Running epoch {epoch} with model {self.context.oracle_file_path} with "
            f"timestamp {loader.get_active_model_timestamp()}"
        )


class ModelLoader:

    def __init__(self, model, context: HDF5AlphaZeroContext):
        self.context = context
        self.model = model
        self.stdin = None
        self.task: asyncio.Task | None = None
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self.model_time = self._get_model_file_timestamp()

    async def __aenter__(self):
        self.reader, self.writer = await asyncio.open_connection(
            self.context.server_hostname,
            self.context.server_port
        )
        self.task = asyncio.create_task(self._listen())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        assert self.writer is not None
        assert self.task is not None
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass
        self.writer.close()
        await self.writer.wait_closed()

    async def notify(self, message):
        assert self.writer is not None
        self.writer.write((message + "\n").encode())
        await self.writer.drain()

    async def _listen(self):
        assert self.reader is not None
        while True:
            line = await self.reader.readline()
            if not line:
                break
            text = line.decode().rstrip()
            if text == "A better model is available":
                self._robust_load()

    def _robust_load(self):
        """Reading might happen concurrently with writing"""
        number_of_attempts = 10
        wait_time_per_attempt = 1
        for attempt in range(number_of_attempts):
            try:
                self.model.load(self.context.oracle_file_path)
                self.model_time = self._get_model_file_timestamp()
            except Exception as error:  # pylint: disable=broad-exception-caught
                if attempt == (number_of_attempts - 1):
                    raise error
                time.sleep(wait_time_per_attempt)
            else:
                break

    def get_active_model_timestamp(self):
        return self.model_time

    def _get_model_file_timestamp(self):
        if os.path.exists(self.context.oracle_file_path):
            return os.path.getmtime(self.context.oracle_file_path)
        return None


if __name__ == "__main__":
    asyncio.run(main())
