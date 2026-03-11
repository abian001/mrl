import pytest
from mrl.game.game_loop import make_game_loop
from mrl.alpha_zero.random_rollout import RandomRollout
from mrl.alpha_zero.mcts import MCTSGame, MCTSPolicy, MemoryfullMCTSPolicy, MCTSConfiguration
from mrl.alpha_zero.mcts_observation import MCTSObservation
from mrl.tic_tac_toe.game import Player
from mrl.tic_tac_toe.mcts_game import MCTSTicTacToe
from mrl.test_utils.policies import RandomPolicy


@pytest.mark.quick
def test_random_rollout(game: MCTSGame):
    oracle: RandomRollout = RandomRollout(number_of_rollouts = 2, game = game)
    observation: MCTSObservation = MCTSObservation(
        game.make_initial_state(),
        game.get_perspectives()[Player.O]
    )
    value = oracle.get_value(observation)
    probabilities = oracle.get_probabilities(observation, (True,) * 9)
    assert 0.0 <= value <= 1.0
    assert tuple(probabilities) == (1.0,) * 9


@pytest.fixture
def game() -> MCTSGame:
    return MCTSTicTacToe()


@pytest.fixture
def policy(game: MCTSGame, memoryfull_policy: bool) -> MCTSPolicy:
    PolicyClass: type
    if memoryfull_policy:
        PolicyClass = MemoryfullMCTSPolicy
    else:
        PolicyClass = MCTSPolicy
    return PolicyClass(
        game = game,
        oracle = RandomRollout(
            game = game,
            number_of_rollouts = 10
        ),
        mcts = MCTSConfiguration(number_of_simulations = 125)
    )

@pytest.mark.parametrize('memoryfull_policy', [True, False])
@pytest.mark.slow
def test_manual_play(game: MCTSGame, policy: MCTSPolicy):
    game_loop = make_game_loop(
        game = game,
        policies = {
            Player.O: RandomPolicy(),
            Player.X: policy
        }
    )
    game_loop.run()
