import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable, Generic, Any
from pydantic import BaseModel, Field
from mrl.game.game import Policy, Game, Player
from mrl.alpha_zero.oracle import Oracle
from mrl.configuration.game_factories import make_game, make_policy, make_stdin_policy
from mrl.configuration.player_utils import validate_player
from mrl.configuration.mcts_factories import make_oracle
from mrl.configuration.gui_factory import make_gui
from mrl.tkinter_gui.gui import Gui
from mrl.test_utils.policies import ManualPolicy


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
    _game_data: dict
    _gui_data: dict | None
    _stdin_data: dict | None
    _gui: Gui | None = None

    @property
    def gui(self) -> Gui:
        if self._gui is None:
            self._gui = make_gui(self._game_data, self._gui_data)
        return self._gui

    @property
    def policies_with_manual_replaced(self) -> dict[Player, Policy]:
        return self.policies | {
            player: make_stdin_policy(
                self._game_data,
                self._stdin_data,
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


class GameRunnerConfiguration(BaseModel):
    game_data: dict = Field(alias = 'game')
    oracle_configs: dict[str, dict] = Field(alias = 'oracles', default_factory = dict)
    shared_policy_configs: dict[str, dict] = Field(
        alias = 'shared_policies',
        default_factory = dict
    )
    policy_configs: dict[str, dict | str] = Field(alias = 'policies', default_factory = dict)
    gui_data: dict | None = Field(alias = 'gui', default = None)
    stdin_data: dict | None = Field(alias = 'stdin_policy', default = None)
    evaluation_config: EvaluationConfiguration = Field(
        alias = 'evaluation',
        default_factory = EvaluationConfiguration
    )


def make_game_runner_specification(data: dict) -> GameRunnerSpecification:
    configuration = GameRunnerConfiguration.model_validate(data)
    game = make_game(configuration.game_data)

    oracles = {
        name: _make_oracle(oracle_config, game)
        for (name, oracle_config) in configuration.oracle_configs.items()
    }
    shared_policies = {
        name: _make_shared_policy(policy_config, game, oracles)
        for (name, policy_config) in configuration.shared_policy_configs.items()
    }
    policies = {
        player: _make_policy(policy_config, player, game, oracles, shared_policies)
        for (player, policy_config) in (
            (validate_player(game, player), policy_config)
            for (player, policy_config) in configuration.policy_configs.items()
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
        configuration.game_data,
        configuration.gui_data,
        configuration.stdin_data
    )


@runtime_checkable
class Loadable(Protocol):
    """A protocol for an oracle"""

    def load(self, file_path: str) -> None:
        """Load the oracle parameters from file"""


def _make_oracle(
    oracle_config: dict,
    game: Game
) -> Oracle:
    oracle = make_oracle(oracle_config | {'game': game})
    if not isinstance(oracle, Oracle):
        raise TypeError(
            f"Invalid oracle class {type(oracle)}. "
            "Oracle classes should derive from Oracle."
        )

    file_path = oracle_config.get('file_path')
    if file_path is not None and oracle_config.get('load', True):
        if not isinstance(oracle, Loadable):
            raise TypeError (
                f"When creating oracle {oracle_config['name']}, there was an attempt "
                "to load parameters, but the oracle does not implement "
                "the method load(self, file_path: str) -> None."
            )
        if os.path.exists(file_path):
            oracle.load(file_path)
        else:
            raise FileNotFoundError(
                f"When creating oracle {oracle_config['name']}, there was an attempt "
                f"to load parameters from {file_path}, but the file does not exist."
            )

    if file_path is None and oracle_config.get('load', False):
        raise ValueError(
            f"When creating oracle {oracle_config['name']}, there is an explicit request "
            "to load parameters, but no file has been defined"
        )
    return oracle


def _make_shared_policy(
    policy_config: dict,
    game: Game,
    oracles: dict[str, Oracle]
) -> Policy:
    oracle_config = policy_config.get('oracle')
    if oracle_config is None:
        return make_policy(policy_config | {'game': game})

    if isinstance(oracle_config, str):
        oracle = oracles.get(oracle_config)
        if oracle is None:
            raise TypeError(
                f"Missing specification for oracle {oracle_config}."
            )
    else:
        oracle = make_oracle(oracle_config | {'game': game})

    policy = make_policy(policy_config | {'game': game, 'oracle': oracle})
    if not isinstance(policy, Policy):
        raise TypeError(
            f"Invalid policy class {type(policy)}. "
            "Policy classes should derive from Policy."
        )
    return policy


def _make_policy(
    policy_config: dict | str,
    player: Any,
    game: Game,
    oracles: dict[str, Oracle],
    shared_policies: dict[str, Policy]
) -> Policy:
    if isinstance(policy_config, str):
        policy = shared_policies.get(policy_config)
        if policy is None:
            raise TypeError(
                f"Missing specification for policy {policy_config}."
            )
    else:
        policy = _make_shared_policy(policy_config | {'player': player}, game, oracles)
    return policy
