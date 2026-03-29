from pathlib import Path
from typing import Any

from mrl.alpha_zero.mcts import MCTSGame
from mrl.alpha_zero.oracle import Oracle, TrainableOracle
from mrl.configuration import factory as _factory
from mrl.configuration.factory import ObjectConfiguration, make_object
from mrl.configuration.predefined import predefined_guis, predefined_stdin_policies
from mrl.game.game import Policy
from mrl.tkinter_gui.gui import Gui

make_game = _factory.make_game


def make_mcts_game(game_configuration: ObjectConfiguration) -> MCTSGame:
    game = make_object(game_configuration)
    if not isinstance(game, MCTSGame):
        raise TypeError(
            f"Invalid game class {type(game)}. "
            "MCTS game classes should derive from class MCTSGame."
        )
    return game


def make_oracle(
    oracle_configuration: ObjectConfiguration,
    game: Any,
) -> Oracle:
    oracle = make_object(
        oracle_configuration,
        extra_arguments = {'game': game},
    )
    if not isinstance(oracle, Oracle):
        raise TypeError(
            f"Invalid oracle class {type(oracle)}. "
            "Oracle classes should derive from class Oracle."
        )
    return oracle


def make_trainable_oracle(
    oracle_configuration: ObjectConfiguration,
    game: Any,
) -> TrainableOracle:
    oracle = make_oracle(oracle_configuration, game)
    if not isinstance(oracle, TrainableOracle):
        raise TypeError(
            f"Invalid oracle class {type(oracle)}. "
            "Oracle classes should derive from class TrainableOracle."
        )
    return oracle


def make_policy(
    policy_configuration: ObjectConfiguration,
    *,
    game: Any,
    oracle: Oracle | None = None,
    player: Any = None,
) -> Policy:
    extra_arguments: dict[str, Any] = {'game': game}
    if oracle is not None:
        extra_arguments['oracle'] = oracle
    if player is not None:
        extra_arguments['player'] = player
    policy = make_object(policy_configuration, extra_arguments = extra_arguments)
    if not isinstance(policy, Policy):
        raise TypeError(
            f"Invalid policy class {type(policy)}. "
            "Policy classes should derive from class Policy."
        )
    return policy


def make_stdin_policy(
    game_configuration: ObjectConfiguration,
    stdin_configuration: ObjectConfiguration | None = None,
    *,
    player: Any = None,
) -> Policy:
    if stdin_configuration is None:
        stdin_configuration = _get_predefined_stdin_policy_configuration(game_configuration)
    extra_arguments = {} if player is None else {'player': player}
    policy = make_object(stdin_configuration, extra_arguments = extra_arguments)
    if not isinstance(policy, Policy):
        raise TypeError(
            f"Invalid policy class {type(policy)}. "
            "Policy classes should derive from class Policy."
        )
    return policy


def make_gui(
    game_configuration: ObjectConfiguration,
    gui_configuration: ObjectConfiguration | None = None,
):
    if gui_configuration is None:
        gui_configuration = _get_predefined_gui_configuration(game_configuration)

    gui = make_object(
        gui_configuration,
        extra_arguments = {'game_configuration': game_configuration}
    )
    if not isinstance(gui, Gui):
        raise TypeError(
            f"Invalid gui class {type(gui)}. "
            "Gui classes should derive from tkinter_gui.gui.Gui."
        )
    return gui


def _get_oracle_file_path(
    workspace_path: Path,
    oracle_file_path: Path,
) -> Path:
    if oracle_file_path.is_absolute():
        return oracle_file_path
    return workspace_path / oracle_file_path


def _get_predefined_stdin_policy_configuration(
    game_configuration: ObjectConfiguration,
) -> ObjectConfiguration:
    predefined_policy = predefined_stdin_policies.get(game_configuration.name)
    if predefined_policy is None:
        raise TypeError(
            f"No predefined stdin policy for game {game_configuration.name}."
        )
    return ObjectConfiguration(name = predefined_policy[0], module = predefined_policy[1])


def _get_predefined_gui_configuration(
    game_configuration: ObjectConfiguration,
) -> ObjectConfiguration:
    predefined_gui = predefined_guis.get(game_configuration.name)
    if predefined_gui is None:
        raise TypeError(
            f"No predefined gui for game {game_configuration.name}."
        )
    return ObjectConfiguration(name = predefined_gui[0], module = predefined_gui[1])
