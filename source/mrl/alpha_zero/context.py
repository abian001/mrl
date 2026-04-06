from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from trueskill import TrueSkill  # type: ignore[import-untyped]

from mrl.alpha_zero.experience_collector import CollectorConfiguration
from mrl.alpha_zero.mcts import MCTSGame
from mrl.alpha_zero.model_trainer import ModelTrainerConfiguration
from mrl.alpha_zero.oracle import TrainableOracle
from mrl.configuration.alpha_zero_configuration import (
    OracleConfiguration,
)
from mrl.game.game import Policy


@dataclass(frozen = True)
class EvaluationPolicies:
    lead: Policy
    opponents: list[Policy]


@dataclass(frozen = True)
class EvaluationContext:
    episodes: int
    max_models: int
    oracles: list[TrainableOracle]
    policies: EvaluationPolicies
    true_skill: TrueSkill = field(default_factory = TrueSkill)
    uncertainty_penalty_coefficient: float = 3.0
    discount_factor: float = 1.0


@dataclass(frozen = True)
class ReportGeneratorContext:
    policies: dict[Any, Policy]
    observed_players: tuple[Any, ...]
    number_of_tests: int
    buckets: tuple[tuple[float, float], ...] | None


@dataclass
class _BaseAlphaZeroContext:  # pylint: disable=too-many-instance-attributes
    game: MCTSGame
    oracle_configuration: OracleConfiguration
    oracle: TrainableOracle
    trainer: ModelTrainerConfiguration
    collector: CollectorConfiguration
    number_of_epochs: int
    report_generator: ReportGeneratorContext | None
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
