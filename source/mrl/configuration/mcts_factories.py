from typing import Any
from mrl.configuration.factory import ObjectConfiguration, make_object
from mrl.alpha_zero.oracle import Oracle
from mrl.alpha_zero.mcts import MCTSGame

def make_mcts_game(game_config: ObjectConfiguration) -> MCTSGame:
    game = make_object(game_config)
    if not isinstance(game, MCTSGame):
        raise TypeError(
            f"Invalid game class {type(game)}. "
            "MCTS game classes should derive from class MCTSGame."
        )
    return game


def make_oracle(
    oracle_config: ObjectConfiguration,
    extra_arguments: dict[str, Any] | None = None
) -> Oracle:
    oracle = make_object(oracle_config, extra_arguments = extra_arguments)
    if not isinstance(oracle, Oracle):
        raise TypeError(
            f"Invalid oracle class {type(oracle)}. "
            "Oracle classes should derive from class Oracle."
        )
    return oracle
