from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_serializer, model_validator

from mrl.alpha_zero.experience_collector import CollectorConfiguration
from mrl.alpha_zero.model_trainer import ModelTrainerConfiguration
from mrl.configuration.factory import ObjectConfiguration
Player = Any


class ModelTestConfiguration(BaseModel):
    oracle_led_players: tuple[Player]
    number_of_tests: int
    buckets: tuple[tuple[float, float], ...] | None

    @field_serializer("oracle_led_players")
    def serialize_players(self, players: tuple[Player], _info: Any):
        return [getattr(player, 'value', player) for player in players]

    @field_serializer("buckets")
    def buckets_to_list(self, buckets: tuple[tuple[float, float], ...] | None, _info: Any):
        if buckets is None:
            return None
        return [[x[0], x[1]] for x in buckets]


class EvaluationConfiguration(BaseModel):
    episodes: int
    max_old_models: int
    policy_configuration: ObjectConfiguration = Field(alias = 'policy')


class CapacityConfiguration(BaseModel, extra = 'allow'):
    output_size: int | None = None


class OracleConfiguration(ObjectConfiguration):
    file_path: Path
    capacity: CapacityConfiguration | None = None
    base_output_size: int | None = Field(alias = 'output_size', default = None)

    @field_serializer("file_path")
    def file_path_to_string(self, file_path: Path, _info: Any):
        return str(file_path)

    @property
    def output_size(self) -> int | None:
        if self.capacity is None:
            return self.base_output_size
        return self.capacity.output_size or self.base_output_size


class ManualPlayConfiguration(BaseModel):
    manual_player: str | None = None
    policy_configuration: ObjectConfiguration = Field(
        alias = 'autonomous_policy',
        default_factory = lambda: ObjectConfiguration(name = 'DeterministicOraclePolicy')
    )


class _BaseAlphaZeroConfiguration(BaseModel):
    game_configuration: ObjectConfiguration = Field(alias = 'game')
    oracle_configuration: OracleConfiguration = Field(alias = 'oracle')
    trainer: ModelTrainerConfiguration
    collector: CollectorConfiguration
    number_of_epochs: int
    report_generator: ModelTestConfiguration
    config_file_path: Path
    workspace_path: Path = Field(default = Path("workspace"))
    evaluation: EvaluationConfiguration

    @property
    def oracle_file_path(self) -> Path:
        oracle_file_path = self.oracle_configuration.file_path  # pylint: disable=no-member
        if oracle_file_path.is_absolute():
            return oracle_file_path
        return self.workspace_path / oracle_file_path

    @field_serializer("config_file_path")
    def config_file_path_to_string(self, config_file_path: Path, _info: Any):
        return str(config_file_path)

    @field_serializer("workspace_path")
    def workspace_path_to_string(self, workspace_path: Path, _info: Any):
        return str(workspace_path)


class HDF5AlphaZeroConfiguration(_BaseAlphaZeroConfiguration):
    type: Literal["HDF5"]
    hdf5_path_prefix: Path
    server_hostname: str
    server_port: int

    @field_serializer("hdf5_path_prefix")
    def hdf5_path_prefix_to_string(self, hdf5_path_prefix: Path, _info: Any):
        return str(hdf5_path_prefix)


class InMemoryAlphaZeroConfiguration(_BaseAlphaZeroConfiguration):
    type: Literal["InMemory"]


_UnionAlphaZeroConfiguration = Annotated[
    Union[InMemoryAlphaZeroConfiguration, HDF5AlphaZeroConfiguration],
    Field(discriminator = 'type')
]


class AlphaZeroConfiguration(BaseModel):
    data: _UnionAlphaZeroConfiguration

    @model_validator(mode = 'before')
    @classmethod
    def _wrap_data(cls, data):
        if isinstance(data, dict) and 'data' not in data:
            return {'data': data}
        return data

    def __getattr__(self, name: str):
        return getattr(self.data, name)

    def model_dump(self, *args, **kwargs):
        return self.data.model_dump(*args, **kwargs)
