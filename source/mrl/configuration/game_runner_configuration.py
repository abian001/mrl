import os
from pathlib import Path
from dataclasses import dataclass
from typing import Generic, Protocol, Any, runtime_checkable
from pydantic import BaseModel, Field, field_serializer
from mrl.game.game import Policy, Game, Player
from mrl.configuration.factory import ObjectConfiguration
from mrl.configuration.game_factories import make_game, make_policy, make_stdin_policy
from mrl.configuration.gui_factory import make_gui
from mrl.configuration.mcts_factories import make_oracle
from mrl.configuration.player_utils import validate_player
from mrl.tkinter_gui.gui import Gui
from mrl.test_utils.policies import ManualPolicy
from mrl.alpha_zero.oracle import Oracle


@dataclass
class EvaluationSpecification(Generic[Player]):
    observed_players: tuple[Player, ...]
    number_of_tests: int
    buckets: tuple[tuple[float, float], ...]


@dataclass
class GameRunnerSpecification(Generic[Player]):
    game: Game
    policies: dict[Player, Policy]
    evaluation: EvaluationSpecification
    _game_configuration: ObjectConfiguration
    _gui_configuration: ObjectConfiguration | None
    _stdin_configuration: ObjectConfiguration | None
    _gui: Gui | None = None

    @property
    def gui(self) -> Gui:
        if self._gui is None:
            self._gui = make_gui(self._game_configuration, self._gui_configuration)
        return self._gui

    @property
    def policies_with_manual_replaced(self) -> dict[Player, Policy]:
        return self.policies | {
            player: make_stdin_policy(
                self._game_configuration,
                self._stdin_configuration,
                player
            )
            for (player, policy) in self.policies.items()
            if isinstance(policy, ManualPolicy)
        }


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


def make_game_runner_specification(data: dict) -> GameRunnerSpecification:
    configuration = GameRunnerConfiguration.model_validate(data)
    game = make_game(configuration.game_configuration)

    oracles = {
        name: _make_oracle(oracle_config, game)
        for (name, oracle_config) in configuration.oracle_configurations.items()
    }
    shared_policies = {
        name: _make_policy(policy_config, game, oracles)
        for (name, policy_config) in configuration.shared_policy_configurations.items()
    }
    policies = {
        player: _make_player_policy(policy_config, game, oracles, player, shared_policies)
        for (player, policy_config) in (
            (validate_player(game, player), policy_config)
            for (player, policy_config) in configuration.policy_configurations.items()
        )
    }

    missing_player = next((
        player for player in game.get_players()
        if player not in policies.keys()
    ), None)
    if missing_player is not None:
        raise TypeError(
            f"Missing policy specification for player {missing_player}."
        )

    observed_players = (
        game.get_players()
        if configuration.evaluation_config.observed_players is None
        else tuple(
            (validate_player(game, player)
            for player in configuration.evaluation_config.observed_players)
        )
    )
    return GameRunnerSpecification(
        game,
        policies,
        EvaluationSpecification(
            observed_players,
            configuration.evaluation_config.number_of_tests,
            configuration.evaluation_config.buckets
        ),
        configuration.game_configuration,
        configuration.gui_configuration,
        configuration.stdin_configuration
    )


@runtime_checkable
class Loadable(Protocol):

    def load(self, file_path: str) -> None:
        """Load the oracle parameters from file"""


def _make_oracle(oracle_config: OracleConfiguration, game: Game) -> Oracle:
    oracle = make_oracle(oracle_config, extra_arguments = {'game': game})
    _maybe_load_oracle(oracle_config, oracle)
    return oracle


def _maybe_load_oracle(oracle_config: OracleConfiguration, oracle: Oracle) -> None:
    file_path = oracle_config.file_path
    if file_path is None:
        return
    if not isinstance(oracle, Loadable):
        raise TypeError(
            f"When creating oracle {oracle_config.name}, there was an attempt "
            "to load parameters, but the oracle does not implement "
            "the method load(self, file_path: str) -> None."
        )
    if os.path.exists(file_path):
        oracle.load(str(file_path))
        return
    raise FileNotFoundError(
        f"When creating oracle {oracle_config.name}, there was an attempt "
        f"to load parameters from {file_path}, but the file does not exist."
    )


def _make_policy(
    policy_config: PolicyConfiguration,
    game: Game,
    oracles: dict[str, Oracle],
    player: Any = None
) -> Policy:
    extra_arguments: dict[str, object] = {'game': game}
    if player is not None:
        extra_arguments['player'] = player
    oracle = _resolve_policy_oracle(policy_config.oracle, game, oracles)
    if oracle is not None:
        extra_arguments['oracle'] = oracle
    return make_policy(policy_config, extra_arguments = extra_arguments)


def _make_player_policy(
    policy_config: PolicyConfiguration | str,
    game: Game,
    oracles: dict[str, Oracle],
    player: Any,
    shared_policies: dict[str, Policy]
) -> Policy:
    if isinstance(policy_config, str):
        policy = shared_policies.get(policy_config)
        if policy is None:
            raise TypeError(
                f"Missing specification for policy {policy_config}."
            )
        return policy

    return _make_policy(policy_config, game, oracles, player)


def _resolve_policy_oracle(
    oracle_config: str | OracleConfiguration | None,
    game: Game,
    oracles: dict[str, Oracle]
) -> Oracle | None:
    if oracle_config is None:
        return None
    if isinstance(oracle_config, str):
        oracle = oracles.get(oracle_config)
        if oracle is None:
            raise TypeError(
                f"Missing specification for oracle {oracle_config}."
            )
        return oracle
    return _make_oracle(oracle_config, game)
