from dataclasses import dataclass
from pathlib import Path
from collections.abc import Generator
from typing import Any, Protocol, cast
import os
import yaml
import torch
import pytest
from mrl.alpha_zero.context import HDF5AlphaZeroContext, InMemoryAlphaZeroContext
from mrl.alpha_zero.alpha_zero import AlphaZero
from mrl.alpha_zero.distributed_alpha_zero import DistributedAlphaZero
from mrl.alpha_zero.experience_collector import Buffer
from mrl.configuration.alpha_zero_configuration import AlphaZeroConfiguration
from mrl.configuration.alpha_zero_runner_factory import AlphaZeroRunnerFactory


class Savable(Protocol):

    def save(self, file_path) -> None:
        """Save model state to disk."""


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
            max_models: 2
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
            number_of_tests: 1
            buckets:
                - [-1.0, 2.0]
            policies:
                X:
                    name: DeterministicOraclePolicy
                    oracle: TrainedOracle
                O:
                    name: RandomPolicy
        number_of_epochs: 1
        config_file_path: {specification.config_file_path}
        workspace_path: {specification.workspace_path}
        hdf5_path_prefix: resume_data
        server_hostname: 127.0.0.1
        server_port: 8888
    """)
    return AlphaZeroRunnerFactory().make_context(
        AlphaZeroConfiguration.model_validate(config_dict)
    )


@pytest.fixture
def in_memory_context(
    in_memory_specification: Specification
) -> InMemoryAlphaZeroContext:
    context = _make_configuration(in_memory_specification)
    assert isinstance(context, InMemoryAlphaZeroContext)
    return context


@pytest.fixture
def hdf5_context(
    hdf5_specification: Specification
) -> HDF5AlphaZeroContext:
    context = _make_configuration(hdf5_specification)
    assert isinstance(context, HDF5AlphaZeroContext)
    return context


@pytest.fixture
def saved_in_memory_model(
    in_memory_specification: Specification
) -> Generator[float, None, None]:
    context = _make_configuration(in_memory_specification)
    assert isinstance(context, InMemoryAlphaZeroContext)
    _fill_model(context.oracle, in_memory_specification.model_value)
    oracle: Savable = cast(Savable, context.oracle)
    oracle.save(context.oracle_file_path)  # pylint: disable=no-member
    yield in_memory_specification.model_value
    if os.path.exists(context.oracle_file_path):
        os.remove(context.oracle_file_path)


@pytest.fixture
def saved_hdf5_model(
    hdf5_specification: Specification
) -> Generator[float, None, None]:
    context = _make_configuration(hdf5_specification)
    assert isinstance(context, HDF5AlphaZeroContext)
    _fill_model(context.oracle, hdf5_specification.model_value)
    oracle: Savable = cast(Savable, context.oracle)
    oracle.save(context.oracle_file_path)  # pylint: disable=no-member
    yield hdf5_specification.model_value
    if os.path.exists(context.oracle_file_path):
        os.remove(context.oracle_file_path)


@pytest.mark.quick
def test_alpha_zero_train_resume_loads_saved_model(
    in_memory_context: InMemoryAlphaZeroContext,
    saved_in_memory_model: float,
    monkeypatch: pytest.MonkeyPatch
):
    alpha_zero = AlphaZero(in_memory_context)
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
    hdf5_context: HDF5AlphaZeroContext,
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

    alpha_zero = DistributedAlphaZero(hdf5_context)
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
