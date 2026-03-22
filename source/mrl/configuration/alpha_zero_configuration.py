from pathlib import Path
from typing import Any, Literal, Union, Annotated
import yaml
from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_serializer,
    PrivateAttr
)
from mrl.configuration.gui_factory import make_gui
from mrl.configuration.player_utils import validate_player
from mrl.configuration.mcts_factories import make_oracle, make_mcts_game
from mrl.tkinter_gui.gui import Gui
from mrl.alpha_zero.experience_collector import CollectorConfiguration
from mrl.alpha_zero.oracle import Oracle
from mrl.alpha_zero.mcts import MCTSGame
from mrl.alpha_zero.model_trainer import ModelTrainerConfiguration


Player = Any  # Will become the specific game's player at run time

class ModelTestConfiguration(BaseModel):
    oracle_led_players: tuple[Player]
    number_of_tests: int
    buckets: tuple[tuple[float, float], ...] | None

    @field_serializer("oracle_led_players")
    def serialize_players(self, players: tuple[Player], _info: Any):
        return [x.value for x in players]

    @field_serializer("buckets")
    def buckets_to_list(self, buckets: tuple[tuple[float, float]], _info: Any):
        return [[x[0], x[1]] for x in buckets]


class EvaluationConfiguration(BaseModel):
    episodes: int
    max_old_models: int
    policy_data: dict = Field(alias = 'policy')


class OutputSize(BaseModel):
    output_size: int | None = None


class OracleConfiguration(BaseModel):
    file_path: Path
    capacity: OutputSize | None = None
    base_output_size: int | None = Field(alias = 'output_size', default = None)

    @field_serializer("file_path")
    def file_path_to_string(self, file_path: Path, _info: Any):
        return str(file_path)

    @property
    def output_size(self) -> int | None:
        if self.capacity is None:
            return self.base_output_size
        return self.capacity.output_size or self.base_output_size


class _BaseAlphaZeroConfiguration(BaseModel):
    game_data: dict = Field(alias = 'game')
    oracle_data: dict = Field(alias = 'oracle')
    gui_data: dict = Field(alias = 'gui', default_factory = dict)
    trainer: ModelTrainerConfiguration
    collector: CollectorConfiguration
    number_of_epochs: int
    report_generator: ModelTestConfiguration
    config_file_path: Path
    workspace_path: Path = Field(default = Path("workspace"))
    evaluation: EvaluationConfiguration
    _game: MCTSGame | None = PrivateAttr(default = None)
    _oracle_config: OracleConfiguration | None = PrivateAttr(default = None)
    _oracle: Oracle | None = PrivateAttr(default = None)
    _gui: Gui | None = PrivateAttr(default = None)

    @property
    def game(self):
        if self._game is None:
            self._game = make_mcts_game(self.game_data)
            if not isinstance(self._game, MCTSGame):
                raise TypeError(
                    f'Game class {type(self._game)} is not an mcts game. '
                    'Only MCTS Games are supported for alpha zero.'
                )
        return self._game

    @property
    def oracle(self):
        if self._oracle is None:
            self._oracle = make_oracle(self.oracle_data | {'game': self.game})
            if not isinstance(self._oracle, Oracle):
                raise TypeError(
                    f'Oracle class {type(self._oracle)} is not a valid oracle '
                    'class. Only classes derived form Oracle are supported.'
                )
        return self._oracle

    @property
    def oracle_config(self) -> OracleConfiguration:
        if self._oracle_config is None:
            self._oracle_config = OracleConfiguration.model_validate(self.oracle_data)
        return self._oracle_config

    @property
    def oracle_file_path(self) -> Path:
        if self.oracle_config.file_path.is_absolute():
            return self.oracle_config.file_path
        return self.workspace_path / self.oracle_config.file_path

    @property
    def gui(self) -> Gui:
        if self._gui is None:
            self._gui = make_gui(self.game_data, self.gui_data)
        return self._gui

    def save(self, file_path: str | None = None):
        save_path = file_path or self.config_file_path
        if save_path is None:
            raise ValueError(
                "No output file defined when saving alpha zero configuration. "
                "Either pass a path to save() or include config_file_path in the configuration."
            )
        with open(save_path, "w", encoding = 'UTF-8') as yaml_file:
            yaml.dump(self.model_dump(by_alias = True), yaml_file)

    @field_serializer("config_file_path")
    def config_file_path_to_string(self, config_file_path: Path, _info: Any):
        return str(config_file_path)

    @field_serializer("workspace_path")
    def workspace_path_to_string(self, workspace_path: Path, _info: Any):
        return str(workspace_path)

    @model_validator(mode = "after")
    def validate_report_generator_oracle_led_players(self):
        self.report_generator.oracle_led_players = tuple(
            validate_player(self.game, player)
            for player in self.report_generator.oracle_led_players
        )
        return self

    @model_validator(mode = "after")
    def validate_output_size(self):
        any_perspective = next(iter(self.game.get_perspectives().values()))
        if not all(
            any_perspective.action_space_dimension == p.action_space_dimension
            for p in self.game.get_perspectives().values()
        ):
            raise TypeError(
                f'Perspectives of game {type(self.game)} return different action '
                'space sizes. This implementation of alpha zero only '
                'supports games whose perspectives have the same action space.'
            )
        output_size = self.oracle_config.output_size
        if output_size is not None and any_perspective.action_space_dimension != output_size:
            raise TypeError(
                'Mismatched output sizes between the game and the model. The game '
                f'output space size is {any_perspective.action_space_dimension}. '
                f'The model output size is {self.oracle_config.output_size}.'
            )
        return self


def make_new_oracle_clone(config: _BaseAlphaZeroConfiguration) -> Oracle:
    return make_oracle(config.oracle_data | {'game': config.game})


class HDF5AlphaZeroConfiguration(_BaseAlphaZeroConfiguration):
    type: Literal["HDF5"]
    hdf5_path_prefix: Path
    server_hostname: str
    server_port: int

    @field_serializer("hdf5_path_prefix")
    def hdf5_path_prefix_to_string(self, hdf5_path_prefix: Path, _info: Any):
        return str(hdf5_path_prefix)

    @model_validator(mode = "after")
    def validate_config_file_path(self):
        # configuration file must exist because HDF5 relies on sharing the
        # file across processes.
        if not self.config_file_path.exists():
            self.save()
        return self


class InMemoryAlphaZeroConfiguration(_BaseAlphaZeroConfiguration):
    type: Literal["InMemory"]


UnionAlphaZeroConfiguration = Annotated[
    Union[InMemoryAlphaZeroConfiguration, HDF5AlphaZeroConfiguration],
    Field(discriminator = 'type')
]


class AlphaZeroConfiguration(BaseModel):
    alpha_zero: UnionAlphaZeroConfiguration
