from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mrl.alpha_zero.experience_collector import CollectorConfiguration
from mrl.alpha_zero.mcts import MCTSGame
from mrl.alpha_zero.model_trainer import ModelTrainerConfiguration
from mrl.alpha_zero.oracle import TrainableOracle
from mrl.configuration.alpha_zero_configuration import (
    ModelTestConfiguration,
    OracleConfiguration,
)
from mrl.game.game import Policy


@dataclass(frozen = True)
class EvaluationPolicies:
    lead: Policy
    opponent: Policy


@dataclass(frozen = True)
class EvaluationContext:
    episodes: int
    max_old_models: int
    oracle: TrainableOracle
    policies: EvaluationPolicies


@dataclass
class _BaseAlphaZeroContext:  # pylint: disable=too-many-instance-attributes
    game: MCTSGame
    oracle_configuration: OracleConfiguration
    oracle: TrainableOracle
    trainer: ModelTrainerConfiguration
    collector: CollectorConfiguration
    number_of_epochs: int
    report_generator: ModelTestConfiguration
    config_file_path: Path
    workspace_path: Path
    evaluation: EvaluationContext
    oracle_file_path: Path


@dataclass
class InMemoryAlphaZeroContext(_BaseAlphaZeroContext):
    type: Literal["InMemory"] = "InMemory"


@dataclass
class HDF5AlphaZeroContext(_BaseAlphaZeroContext):
    hdf5_path_prefix: Path | None = None
    server_hostname: str | None = None
    server_port: int | None = None
    type: Literal["HDF5"] = "HDF5"


UnionAlphaZeroContext = InMemoryAlphaZeroContext | HDF5AlphaZeroContext
AlphaZeroContext = UnionAlphaZeroContext
