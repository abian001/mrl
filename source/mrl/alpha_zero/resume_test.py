from dataclasses import dataclass
from pathlib import Path
from collections.abc import Generator
from typing import Any, cast
import os
import yaml
import torch
import pytest
from mrl.alpha_zero.alpha_zero import AlphaZero
from mrl.alpha_zero.distributed_alpha_zero import DistributedAlphaZero
from mrl.alpha_zero.model_updater import ModelUpdater, AverageScore
from mrl.alpha_zero.experience_collector import Buffer
from mrl.configuration.alpha_zero_configuration import (
    AlphaZeroConfiguration,
    InMemoryAlphaZeroConfiguration,
    HDF5AlphaZeroConfiguration,
)


@dataclass
class Specification:
    memory_type: str
    config_file_path: str
    workspace_path: str
    model_value: float = 0.5


@pytest.fixture
def in_memory_specification(tmp_path: Path) -> Specification:
    return Specification(
        memory_type = "InMemory",
        config_file_path = str(tmp_path / "in_memory_config.yaml"),
        workspace_path = str(tmp_path),
    )


@pytest.fixture
def hdf5_specification(tmp_path: Path) -> Specification:
    return Specification(
        memory_type = "HDF5",
        config_file_path = str(tmp_path / "hdf5_config.yaml"),
        workspace_path = str(tmp_path),
    )


def _make_configuration(specification: Specification):
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
                nn_depth: 1
                nn_width: 2
            file_path: resume_model
        evaluation:
            episodes: 2
            max_old_models: 2
            policy:
                name: DeterministicOraclePolicy
        collector:
            number_of_processes: 1
            mcts:
                number_of_simulations: 1
                pucb_constant: 1.0
                discount_factor: 1.0
            max_buffer_length: 10
            number_of_episodes: 1
            temperature_schedule:
                - [0, 1.0]
        trainer:
            batch_size: 1
            max_training_epochs: 1
            early_stop_loss: 0.001
            learning_rate: 0.001
            loading_workers: 1
        report_generator:
            oracle_led_players: ['X']
            number_of_tests: 1
            buckets:
                - [-1.0, 2.0]
        number_of_epochs: 1
        config_file_path: {specification.config_file_path}
        workspace_path: {specification.workspace_path}
        hdf5_path_prefix: resume_data
        server_hostname: 127.0.0.1
        server_port: 8888
    """)
    return AlphaZeroConfiguration(alpha_zero = config_dict).alpha_zero


@pytest.fixture
def in_memory_configuration(
    in_memory_specification: Specification
) -> InMemoryAlphaZeroConfiguration:
    config = _make_configuration(in_memory_specification)
    assert isinstance(config, InMemoryAlphaZeroConfiguration)
    return config


@pytest.fixture
def hdf5_configuration(
    hdf5_specification: Specification
) -> HDF5AlphaZeroConfiguration:
    config = _make_configuration(hdf5_specification)
    assert isinstance(config, HDF5AlphaZeroConfiguration)
    return config


@pytest.fixture
def saved_in_memory_model(
    in_memory_specification: Specification
) -> Generator[float, None, None]:
    config = _make_configuration(in_memory_specification)
    assert isinstance(config, InMemoryAlphaZeroConfiguration)
    _fill_model(config.oracle, in_memory_specification.model_value)
    config.oracle.save(config.oracle_file_path)
    yield in_memory_specification.model_value
    if os.path.exists(config.oracle_file_path):
        os.remove(config.oracle_file_path)


@pytest.fixture
def saved_hdf5_model(
    hdf5_specification: Specification
) -> Generator[float, None, None]:
    config = _make_configuration(hdf5_specification)
    assert isinstance(config, HDF5AlphaZeroConfiguration)
    _fill_model(config.oracle, hdf5_specification.model_value)
    config.oracle.save(config.oracle_file_path)
    yield hdf5_specification.model_value
    if os.path.exists(config.oracle_file_path):
        os.remove(config.oracle_file_path)


@pytest.fixture
def old_in_memory_model(
    in_memory_configuration: InMemoryAlphaZeroConfiguration
) -> Generator[str, None, None]:
    old_model_path = f"{in_memory_configuration.oracle_file_path}_old_0"
    in_memory_configuration.oracle.save(in_memory_configuration.oracle_file_path)
    in_memory_configuration.oracle.save(old_model_path)
    yield old_model_path
    if os.path.exists(old_model_path):
        os.remove(old_model_path)
    if os.path.exists(in_memory_configuration.oracle_file_path):
        os.remove(in_memory_configuration.oracle_file_path)


@pytest.mark.quick
def test_model_updater_restores_scores_from_yaml(
    in_memory_configuration: InMemoryAlphaZeroConfiguration,
    old_in_memory_model: str
):
    updater = ModelUpdater(in_memory_configuration.game, in_memory_configuration)
    tracker = AverageScore()
    tracker.append(0.25)
    tracker.append(0.75)
    updater.scores[old_in_memory_model] = tracker
    in_memory_configuration.oracle.save(in_memory_configuration.oracle_file_path)
    updater.scores.save(updater.old_models)

    restored = ModelUpdater(in_memory_configuration.game, in_memory_configuration)
    assert old_in_memory_model in restored.scores
    assert restored.scores[old_in_memory_model].count == 2
    assert restored.scores[old_in_memory_model].average_score == 0.5


@pytest.mark.quick
def test_model_updater_resume_without_saved_scores_does_not_crash(
    in_memory_configuration: InMemoryAlphaZeroConfiguration,
    old_in_memory_model: str,
    monkeypatch: pytest.MonkeyPatch
):
    _ = old_in_memory_model  # Needed to create the model file.
    updater = ModelUpdater(in_memory_configuration.game, in_memory_configuration)
    monkeypatch.setattr(updater, "_evaluate_once", lambda lead, opponent: 0.0)

    updater.save_if_better(in_memory_configuration.oracle)
    assert os.path.exists(updater.scores_path)


@pytest.mark.quick
def test_alpha_zero_train_resume_loads_saved_model(
    in_memory_configuration: InMemoryAlphaZeroConfiguration,
    saved_in_memory_model: float,
    monkeypatch: pytest.MonkeyPatch
):
    alpha_zero = AlphaZero(in_memory_configuration)
    before = _first_parameter(alpha_zero.model)
    empty_buffer = cast(Buffer[Any], [])

    def collect() -> tuple[Buffer[Any], bool]:
        return empty_buffer, False

    monkeypatch.setattr(alpha_zero.collector, "collect", collect)
    alpha_zero.train(resume = True)
    after = _first_parameter(alpha_zero.model)

    assert not torch.allclose(before, after)
    assert torch.allclose(after, torch.full_like(after, saved_in_memory_model))


@pytest.mark.quick
def test_distributed_alpha_zero_train_resume_loads_saved_model(
    hdf5_configuration: HDF5AlphaZeroConfiguration,
    saved_hdf5_model: float,
    monkeypatch: pytest.MonkeyPatch
):
    class EmptyFiles:

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration()

    class StubExperienceCollector:

        def __init__(self, _config):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        def get_files(self):
            return EmptyFiles()

        async def notify_better_model(self):
            return None

    monkeypatch.setattr(
        "mrl.alpha_zero.distributed_alpha_zero.ExperienceCollector",
        StubExperienceCollector
    )

    alpha_zero = DistributedAlphaZero(hdf5_configuration)
    before = _first_parameter(alpha_zero.model)
    alpha_zero.train(resume = True)
    after = _first_parameter(alpha_zero.model)

    assert not torch.allclose(before, after)
    assert torch.allclose(after, torch.full_like(after, saved_hdf5_model))


def _fill_model(model: object, value: float):
    module = cast(torch.nn.Module, model)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(value)


def _first_parameter(model: object) -> torch.Tensor:
    module = cast(torch.nn.Module, model)
    return next(module.parameters()).detach().clone()
