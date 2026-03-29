from pathlib import Path

from pydantic import BaseModel, Field, field_serializer

from mrl.configuration.factory import ObjectConfiguration


class EvaluationConfiguration(BaseModel):
    observed_players: tuple[str, ...] | None = None
    number_of_tests: int = 1
    buckets: tuple[tuple[float, float], ...] = Field(
        default_factory = lambda: ((float("-inf"), float("+inf")),)
    )


class OracleConfiguration(ObjectConfiguration):
    file_path: Path | None = None

    @field_serializer("file_path")
    def file_path_to_string(self, file_path: Path | None, _info):
        return None if file_path is None else str(file_path)


class PolicyConfiguration(ObjectConfiguration):
    oracle: str | OracleConfiguration | None = None


class GameRunnerConfiguration(BaseModel):
    game_configuration: ObjectConfiguration = Field(alias = 'game')
    oracle_configurations: dict[str, OracleConfiguration] = Field(
        alias = 'oracles',
        default_factory = dict
    )
    shared_policy_configurations: dict[str, PolicyConfiguration] = Field(
        alias = 'shared_policies',
        default_factory = dict
    )
    policy_configurations: dict[str, PolicyConfiguration | str] = Field(
        alias = 'policies',
        default_factory = dict
    )
    gui_configuration: ObjectConfiguration | None = Field(alias = 'gui', default = None)
    stdin_configuration: ObjectConfiguration | None = Field(alias = 'stdin_policy', default = None)
    evaluation_config: EvaluationConfiguration = Field(
        alias = 'evaluation',
        default_factory = EvaluationConfiguration
    )
