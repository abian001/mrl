from typing import Any
from mrl.configuration.factory import Configuration, make_object
from mrl.configuration.predefined import predefined_stdin_policies
from mrl.game.game import Game, Policy


def make_game(game_config: dict) -> Game:
    game = make_object(game_config)
    if not isinstance(game, Game):
        raise TypeError(
            f"Invalid game class {type(game)}. "
            "Game classes should derive from class Game."
        )
    return game


def make_policy(policy_config: dict) -> Policy:
    policy = make_object(policy_config)
    if not isinstance(policy, Policy):
        raise TypeError(
            f"Invalid policy class {type(policy)}. "
            "Policy classes should derive from class Policy."
        )
    return policy


def make_stdin_policy(
    game_data: dict,
    stdin_policy_data: dict | None = None,
    player: Any = None
) -> Policy:
    if stdin_policy_data is None:
        stdin_policy_data = _get_predefined_stdin_policy_data(game_data)

    return make_policy(
        stdin_policy_data |
        {'game_data': game_data} |
        ({} if player is None else {'player': player})
    )


def _get_predefined_stdin_policy_data(game_data: dict) -> dict:
    base_game_config = Configuration.model_validate(game_data)
    predefined_policy = predefined_stdin_policies.get(base_game_config.name)
    if predefined_policy is None:
        raise TypeError(
            f"No predefined stdin policy for game {base_game_config.name}."
        )
    return {'name': predefined_policy[0], 'module': predefined_policy[1]}
