import os
from typing import Any, Protocol, runtime_checkable

from mrl.alpha_zero.oracle import Oracle
from mrl.configuration.game_runner_configuration import (
    GameRunnerConfiguration,
    OracleConfiguration,
    PolicyConfiguration,
)
from mrl.configuration.player_utils import validate_player
from mrl.game.game import Game, Policy
from mrl.test_utils.policies import ManualPolicy
from mrl.configuration.runner_factories import (
    make_game,
    make_gui,
    make_oracle,
    make_policy,
    make_stdin_policy,
)


class PoliciesConfigurationProtocol(Protocol):
    oracle_configurations: dict[str, OracleConfiguration]
    shared_policy_configurations: dict[str, PolicyConfiguration]
    policy_configurations: dict[str, PolicyConfiguration | str]


class GameRunnerFactory:

    def make_runner_game(self, configuration: GameRunnerConfiguration) -> Game:
        return make_game(configuration.game_configuration)

    def make_policies(
        self,
        configuration: PoliciesConfigurationProtocol,
        game: Game,
        default_oracles: dict[str, Oracle] | None = None,
    ) -> dict[Any, Policy]:
        oracles = {
            name: self._make_oracle_configuration(oracle_configuration, game)
            for (name, oracle_configuration) in configuration.oracle_configurations.items()
        }
        shared_policies = {
            name: self._make_policy_configuration(
                policy_configuration,
                game,
                oracles,
                default_oracles = default_oracles,
            )
            for (name, policy_configuration) in configuration.shared_policy_configurations.items()
        }
        policies = {
            player: self._make_player_policy(
                policy_configuration,
                game,
                oracles,
                player,
                shared_policies,
                default_oracles = default_oracles,
            )
            for (player, policy_configuration) in (
                (validate_player(game, player), policy_configuration)
                for (player, policy_configuration) in configuration.policy_configurations.items()
            )
        }
        self._validate_all_players_have_policies(game, policies)
        return policies

    def make_terminal_policies(
        self,
        configuration: GameRunnerConfiguration,
        policies: dict[Any, Policy],
    ) -> dict[Any, Policy]:
        return policies | {
            player: make_stdin_policy(
                configuration.game_configuration,
                configuration.stdin_configuration,
                player = player,
            )
            for (player, policy) in policies.items()
            if _is_manual_policy(policy)
        }

    def make_runner_gui(self, configuration: GameRunnerConfiguration):
        return make_gui(
            configuration.game_configuration,
            configuration.gui_configuration,
        )

    def make_observed_players(
        self,
        configuration: GameRunnerConfiguration,
        game: Game,
    ):
        if configuration.evaluation_config.observed_players is None:
            return game.get_players()
        return tuple(
            validate_player(game, player)
            for player in configuration.evaluation_config.observed_players
        )

    def _make_oracle_configuration(
        self,
        oracle_configuration: OracleConfiguration,
        game: Game,
    ) -> Oracle:
        oracle = make_oracle(oracle_configuration, game)
        _maybe_load_oracle(oracle_configuration, oracle)
        return oracle

    def _make_policy_configuration(  # pylint: disable=too-many-positional-arguments
        self,
        policy_configuration: PolicyConfiguration,
        game: Game,
        oracles: dict[str, Oracle],
        player: Any = None,
        default_oracles: dict[str, Oracle] | None = None,
    ) -> Policy:
        oracle = self._resolve_policy_oracle(
            policy_configuration.oracle,
            game,
            oracles,
            default_oracles = default_oracles,
        )
        return make_policy(
            policy_configuration,
            game = game,
            oracle = oracle,
            player = player,
        )

    def _make_player_policy(  # pylint: disable=too-many-positional-arguments
        self,
        policy_configuration: PolicyConfiguration | str,
        game: Game,
        oracles: dict[str, Oracle],
        player: Any,
        shared_policies: dict[str, Policy],
        default_oracles: dict[str, Oracle] | None = None,
    ) -> Policy:
        if isinstance(policy_configuration, str):
            policy = shared_policies.get(policy_configuration)
            if policy is None:
                raise TypeError(
                    f"Missing specification for policy {policy_configuration}."
                )
            return policy
        return self._make_policy_configuration(
            policy_configuration = policy_configuration,
            game = game,
            oracles = oracles,
            player = player,
            default_oracles = default_oracles,
        )

    def _resolve_policy_oracle(
        self,
        oracle_configuration: str | OracleConfiguration | None,
        game: Game,
        oracles: dict[str, Oracle],
        default_oracles: dict[str, Oracle] | None = None,
    ) -> Oracle | None:
        if oracle_configuration is None:
            return None
        if isinstance(oracle_configuration, str):
            if default_oracles is not None and oracle_configuration in default_oracles:
                return default_oracles[oracle_configuration]
            oracle = oracles.get(oracle_configuration)
            if oracle is None:
                raise TypeError(
                    f"Missing specification for oracle {oracle_configuration}."
                )
            return oracle
        return self._make_oracle_configuration(oracle_configuration, game)

    def _validate_all_players_have_policies(
        self,
        game: Game,
        policies: dict[Any, Policy],
    ) -> None:
        missing_player = next((
            player for player in game.get_players()
            if player not in policies.keys()
        ), None)
        if missing_player is not None:
            raise TypeError(
                f"Missing policy specification for player {missing_player}."
            )


@runtime_checkable
class Loadable(Protocol):

    def load(self, file_path: str) -> None:
        """Load the oracle parameters from file."""


def _maybe_load_oracle(oracle_configuration: OracleConfiguration, oracle: Oracle) -> None:
    file_path = oracle_configuration.file_path
    if file_path is None:
        return
    if not isinstance(oracle, Loadable):
        raise TypeError(
            f"When creating oracle {oracle_configuration.name}, there was an attempt "
            "to load parameters, but the oracle does not implement "
            "the method load(self, file_path: str) -> None."
        )
    if os.path.exists(file_path):
        oracle.load(str(file_path))
        return
    raise FileNotFoundError(
        f"When creating oracle {oracle_configuration.name}, there was an attempt "
        f"to load parameters from {file_path}, but the file does not exist."
    )


def _is_manual_policy(policy: Policy) -> bool:
    return isinstance(policy, ManualPolicy)
