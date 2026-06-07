from unittest.mock import patch
from collections.abc import Generator
from typing import Any, cast
import os
import sys
import shutil
import asyncio
import random
import pytest
import yaml
import h5py
from mrl.alpha_zero.context import HDF5AlphaZeroContext
from mrl.alpha_zero.self_play_process import main
from mrl.configuration.alpha_zero_configuration import AlphaZeroConfiguration
from mrl.configuration.alpha_zero_runner_factory import AlphaZeroRunnerFactory


@pytest.fixture
def workspace() -> Generator[str, None, None]:
    workspace_path = 'test_workspace'
    if os.path.exists(workspace_path):
        shutil.rmtree(workspace_path)
    os.mkdir(workspace_path)
    yield workspace_path
    if os.path.exists(workspace_path):
        shutil.rmtree(workspace_path)


@pytest.fixture
def server_port() -> int:
    return random.randint(20000, 50000)


@pytest.fixture
def context(
    number_of_epochs: int,
    number_of_simulations: int,
    workspace: str,
    server_port: int,
) -> Generator[HDF5AlphaZeroContext, None, None]:
    data = yaml.safe_load(f"""
        type: HDF5
        game:
            name: MCTSTicTacToe
            first_player: X
        oracle:
            name: OpenSpielMLP
            capacity:
                input_size: 18
                output_size: 9
                nn_depth: 3
                nn_width: 3
            file_path: self_play_test
        evaluation:
            episodes: 10
            max_models: 2
            policy:
                name: DeterministicOraclePolicy
        collector:
            number_of_processes: 1
            mcts:
                number_of_simulations: {number_of_simulations}
                pucb_constant: 1.0
                pucb_increase: 0.0
                discount_factor: 1.0
            max_buffer_length: 1000
            number_of_episodes: 2
            temperature_schedule:
                - [0, 1.0] 
        trainer:
            batch_size: 32
            max_training_epochs: 1
            early_stop_loss: 1e-3
            learning_rate: 1e-3
            loading_workers: 1
        report_generator:
            number_of_tests: 10
            buckets:
                - [-1.0, 0.25]
                - [0.25, 0.75]
                - [0.75, 2.00]
            policies:
                X:
                    name: DeterministicOraclePolicy
                    oracle: TrainedOracle
                O:
                    name: RandomPolicy
        number_of_epochs: {number_of_epochs}
        config_file_path: test_config.json
        hdf5_path_prefix: test_data
        server_hostname: 127.0.0.1
        server_port: {server_port}
        workspace_path: {workspace}
        """
    )
    declarative_config = AlphaZeroConfiguration.model_validate(data)
    context = AlphaZeroRunnerFactory().make_context(declarative_config)
    assert isinstance(context, HDF5AlphaZeroContext)

    yield context
    assert context.config_file_path is not None
    if os.path.exists(context.config_file_path):
        os.remove(context.config_file_path)


@pytest.fixture
def oracle_file(context: HDF5AlphaZeroContext) -> Generator[str, None, None]:
    model = cast(Any, context.oracle)
    model.save(context.oracle_file_path)
    assert context.oracle_file_path is not None

    yield str(context.oracle_file_path)
    if os.path.exists(context.oracle_file_path):
        os.remove(context.oracle_file_path)


@pytest.fixture
def expected_hdf5_files(
    context: HDF5AlphaZeroContext
) -> Generator[tuple[str, ...], None, None]:
    expected_files = tuple(
        f"{context.workspace_path}/{context.hdf5_path_prefix}_1_{i}.h5"
        for i in range(context.number_of_epochs)
    )

    yield expected_files
    for file_path in expected_files:
        if os.path.exists(file_path):
            os.remove(file_path)


@pytest.mark.parametrize('number_of_epochs, number_of_simulations', [(2, 1)])
@pytest.mark.asyncio
@pytest.mark.quick
async def test_play_process(
    context: HDF5AlphaZeroContext,
    expected_hdf5_files: tuple[str, ...]
):
    with patch("mrl.alpha_zero.self_play_process.ModelLoader"):
        await main([
            '--config-file', str(context.config_file_path),
            '--process-index', '1',
        ])
    _assert_expected_files_are_not_empty(expected_hdf5_files)


def _assert_expected_files_are_not_empty(expected_files: tuple[str, ...]):
    for data_file in expected_files:
        assert os.path.exists(data_file)
        with h5py.File(data_file, 'r') as hdf5_file:
            observations = hdf5_file['observations']
            assert isinstance(observations, h5py.Dataset)
            length = len(observations)
        assert length > 0


@pytest.mark.parametrize('number_of_epochs, number_of_simulations', [(5, 25)])
@pytest.mark.asyncio
@pytest.mark.slow
async def test_replace_model(
    context: HDF5AlphaZeroContext,
    oracle_file: str,
    expected_hdf5_files: tuple[str, ...]
):
    assert oracle_file == str(context.oracle_file_path)
    test = TestReplaceModel(context)
    await test.run()
    _assert_expected_files_are_not_empty(expected_hdf5_files)


class TestReplaceModel:

    def __init__(self, context: HDF5AlphaZeroContext):
        self.messages: list[str] = []
        self.config_file_path = context.config_file_path
        self.oracle_file_path = context.oracle_file_path
        self.oracle = context.oracle
        self.game = context.game
        self.server_port = context.server_port

    async def run(self):
        early_timestamp = os.path.getmtime(self.oracle_file_path)
        cmd = [
            sys.executable, "-m", "mrl.alpha_zero.self_play_process",
            '--config-file', str(self.config_file_path),
            '--process-index', '1',
            '--verbose'
        ]
        server = await asyncio.start_server(
            self._handle_client,
            "127.0.0.1",
            self.server_port,
        )
        asyncio.create_task(self._serve(server))
        process = await asyncio.create_subprocess_exec(*cmd)
        await process.wait()
        server.close()
        await server.wait_closed()

        late_timestamp = os.path.getmtime(self.oracle_file_path)
        epoch_messages = tuple(m for (i, m) in enumerate(self.messages) if i % 2 == 0)
        file_messages = tuple(m for (i, m) in enumerate(self.messages) if i % 2 == 1)
        assert \
            epoch_messages[0], \
            f"Running epoch 0 with model self_play_test with timestamp {early_timestamp}"
        assert \
            epoch_messages[4], \
            f"Running epoch 4 with model self_play_test with timestamp {late_timestamp}"
        for (i, message) in enumerate(file_messages):
            assert message == f"test_workspace/test_data_1_{i}.h5"

    async def _serve(self, server):
        async with server:
            await server.serve_forever()

    async def _handle_client(self, reader, writer):
        await self._receive(reader, writer)
        writer.close()
        await writer.wait_closed()

    async def _receive(self, reader, writer):
        while True:
            data = await reader.readline()
            if not data:
                break
            message = data.decode().strip()
            if message == "Done":
                break
            self.messages.append(message)
            if message == "test_data_1_0.h5":
                self.oracle.save(self.oracle_file_path)
                writer.write("A better model is available\n".encode())
                await writer.drain()
