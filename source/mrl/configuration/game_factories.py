from typing import Any
from mrl.configuration.factory import ObjectConfiguration, make_object
from mrl.configuration.predefined import predefined_stdin_policies
from mrl.game.game import Game, Policy


def make_game(game_config: ObjectConfiguration) -> Game:
    game = make_object(game_config)
    if not isinstance(game, Game):
        raise TypeError(
            f"Invalid game class {type(game)}. "
            "Game classes should derive from class Game."
    )
    return game


def make_policy(
    policy_config: ObjectConfiguration,
    extra_arguments: dict[str, Any] | None = None
) -> Policy:
    policy = make_object(policy_config, extra_arguments = extra_arguments)
    if not isinstance(policy, Policy):
        raise TypeError(
            f"Invalid policy class {type(policy)}. "
            "Policy classes should derive from class Policy."
        )
    return policy


def make_stdin_policy(
    game_configuration: ObjectConfiguration,
    stdin_policy_configuration: ObjectConfiguration | None = None,
    player: Any = None
) -> Policy:
    if stdin_policy_configuration is None:
        stdin_policy_configuration = _get_predefined_stdin_policy_configuration(game_configuration)

    extra_arguments = {} if player is None else {'player': player}
    return make_policy(stdin_policy_configuration, extra_arguments = extra_arguments)


def _get_predefined_stdin_policy_configuration(
    game_configuration: ObjectConfiguration
) -> ObjectConfiguration:
    predefined_policy = predefined_stdin_policies.get(game_configuration.name)
    if predefined_policy is None:
        raise TypeError(
            f"No predefined stdin policy for game {game_configuration.name}."
        )
    return ObjectConfiguration(name = predefined_policy[0], module = predefined_policy[1])
