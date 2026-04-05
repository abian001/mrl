from pathlib import Path
from dataclasses import dataclass
from collections.abc import Generator
import shutil
import pytest
import yaml
from mrl.configuration.alpha_zero_configuration import AlphaZeroConfiguration
from mrl.configuration.alpha_zero_runner_factory import AlphaZeroRunnerFactory


# Shallow tests for alpha zero implementation
# tests with memory_type InMemory run alpha_zero.py
# tests with memory_type HDF5 run distributed_alpha_zero.py
@pytest.mark.parametrize('number_of_processes', [1, 2])
@pytest.mark.parametrize('memory_type', ['InMemory', 'HDF5'])
@pytest.mark.slow
def test_alpha_zero(
    specification: "Specification",
    context,
    _clear_workspace: None
):
    alpha_zero = AlphaZeroRunnerFactory().make_alpha_zero(context)
    alpha_zero.train()

    for (prefix, expected) in specification.expected_file_prefixes.items():
        path = Path(prefix)
        file_exists = any(x.name.startswith(path.name) for x in path.parent.iterdir())
        assert file_exists is expected, f"For prefix {prefix}, {file_exists} != {expected}"


@dataclass
class Specification:
    number_of_processes: int
    memory_type: str
    model_file: str = 'alpha_zero_test_model'
    config_file: str = 'alpha_zero_test_config_file.yaml'
    data_file_prefix: str = 'alpha_zero_test_data_file'
    workspace: str = 'test_workspace'

    @property
    def expected_file_prefixes(self) -> dict[str, bool]:
        return {
            self.config_file: self.memory_type == 'HDF5',
            f'{self.workspace}/{self.model_file}': True,
            f'{self.workspace}/{self.model_file}_scores.yaml': True,
            f'{self.workspace}/{self.model_file}_': True,
            f'{self.workspace}/{self.data_file_prefix}': False
        }


@pytest.fixture
def specification(number_of_processes: int, memory_type: str) -> Specification:
    return Specification(number_of_processes, memory_type)


@pytest.fixture
def context(specification: Specification):
    config_dict = yaml.safe_load(f"""
        type: {specification.memory_type}
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
            file_path: {specification.model_file}
        evaluation:
            episodes: 10
            max_old_models: 10
            policy:
                name: DeterministicOraclePolicy
        collector:
            number_of_processes: {specification.number_of_processes}
            mcts:
                number_of_simulations: 1
                pucb_constant: 1.0
                discount_factor: 1.0
            max_buffer_length: 100
            number_of_episodes: 30
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
        number_of_epochs: 2
        config_file_path: {specification.config_file}
        hdf5_path_prefix: {specification.data_file_prefix}
        server_hostname: 127.0.0.1
        server_port: 8888
        cofig_file_path: test_path
        workspace_path: {specification.workspace}
    """)
    return AlphaZeroRunnerFactory().make_context(
        AlphaZeroConfiguration.model_validate(config_dict)
    )


@pytest.fixture
def _clear_workspace(specification: Specification) -> Generator[None, None, None]:
    yield
    prefixes = [specification.model_file, specification.config_file, specification.data_file_prefix]
    for path in Path.cwd().iterdir():
        if path.is_file() and any(path.name.startswith(x) for x in prefixes):
            path.unlink()

    workspace = Path(specification.workspace)
    if workspace.exists():
        shutil.rmtree(workspace)
