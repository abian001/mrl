from unittest.mock import patch
from collections.abc import Generator
import os
import sys
import shutil
import asyncio
import pytest
import yaml
import h5py
from mrl.alpha_zero.self_play_process import main
from mrl.configuration.alpha_zero_configuration import (
    AlphaZeroConfiguration,
    HDF5AlphaZeroConfiguration
)


@pytest.fixture
def workspace() -> Generator[str, None, None]:
    workspace_path = 'test_workspace'
    if os.path.exists(workspace_path):
        raise RuntimeError(
            f"Workspace {workspace_path} should not exists before running the tests"
        )
    os.mkdir(workspace_path)
    yield workspace_path
    shutil.rmtree(workspace_path)


@pytest.fixture
def configuration(
    number_of_epochs: int,
    number_of_simulations: int,
    workspace: str
) -> Generator[HDF5AlphaZeroConfiguration, None, None]:
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
        collector:
            number_of_processes: 1
            mcts:
                number_of_simulations: {number_of_simulations}
                pucb_constant: 1.0
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
            oracle_led_players: ['X']
            number_of_tests: 10
            buckets:
                - [-1.0, 0.25]
                - [0.25, 0.75]
                - [0.75, 2.00]
        number_of_epochs: {number_of_epochs}
        evaluation_episodes: 10
        max_old_models: 2
        config_file_path: test_config.json
        hdf5_path_prefix: test_data
        server_hostname: 127.0.0.1
        server_port: 8888
        workspace_path: {workspace}
        """
    )
    config = AlphaZeroConfiguration(alpha_zero = data).alpha_zero
    config.save()
    assert isinstance(config, HDF5AlphaZeroConfiguration)

    yield config
    assert config.config_file_path is not None
    if os.path.exists(config.config_file_path):
        os.remove(config.config_file_path)


@pytest.fixture
def oracle_file(configuration: HDF5AlphaZeroConfiguration) -> Generator[str, None, None]:
    model = configuration.oracle
    model.save(configuration.oracle_file_path)
    assert configuration.oracle_file_path is not None

    yield str(configuration.oracle_file_path)
    if os.path.exists(configuration.oracle_file_path):
        os.remove(configuration.oracle_file_path)


@pytest.fixture
def expected_hdf5_files(
    configuration: HDF5AlphaZeroConfiguration
) -> Generator[tuple[str, ...], None, None]:
    expected_files = tuple(
        f"{configuration.workspace_path}/{configuration.hdf5_path_prefix}_1_{i}.h5"
        for i in range(configuration.number_of_epochs)
    )

    yield expected_files
    for file_path in expected_files:
        if os.path.exists(file_path):
            os.remove(file_path)


@pytest.mark.parametrize('number_of_epochs, number_of_simulations', [(2, 1)])
@pytest.mark.asyncio
@pytest.mark.quick
async def test_play_process(
    configuration: HDF5AlphaZeroConfiguration,
    expected_hdf5_files: tuple[str, ...]
):
    with patch("mrl.alpha_zero.self_play_process.ModelLoader"):
        await main([
            '--config-file', str(configuration.config_file_path),
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
    configuration: HDF5AlphaZeroConfiguration,
    oracle_file: str,
    expected_hdf5_files: tuple[str, ...]
):
    assert oracle_file == str(configuration.oracle_file_path)
    test = TestReplaceModel(configuration)
    await test.run()
    _assert_expected_files_are_not_empty(expected_hdf5_files)


class TestReplaceModel:

    def __init__(self, configuration: HDF5AlphaZeroConfiguration):
        self.messages: list[str] = []
        self.config_file_path = configuration.config_file_path
        self.oracle_file_path = configuration.oracle_file_path
        self.oracle = configuration.oracle
        self.game = configuration.game

    async def run(self):
        early_timestamp = os.path.getmtime(self.oracle_file_path)
        cmd = [
            sys.executable, "-m", "mrl.alpha_zero.self_play_process",
            '--config-file', str(self.config_file_path),
            '--process-index', '1',
            '--verbose'
        ]
        server = await asyncio.start_server(self._handle_client, "127.0.0.1", 8888)
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
