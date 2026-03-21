from dataclasses import dataclass
import pytest
from mrl.tic_tac_toe.game import Player
from mrl.tic_tac_toe.mcts_game import MCTSTicTacToe, Perspective
from mrl.alpha_zero.mcts import (
    MCTSGame,
    MCTSPolicy,
    NonDeterministicMCTSPolicy,
    MCTSConfiguration
)
from mrl.alpha_zero.mcts_observation import MCTSObservation
from mrl.alpha_zero.oracle import Oracle, LegalMask, Probabilities
from mrl.game.game_loop import make_game_loop
from mrl.test_utils.policies import RandomPolicy


class TestOracle(Oracle):

    def get_probabilities(
        self,
        observation: MCTSObservation,
        legal_mask: LegalMask
    ) -> Probabilities:
        return tuple(1.0 for i in range(9))

    def get_value(self, observation: MCTSObservation) -> float:
        return 0.5


@pytest.fixture
def tic_tac_toe() -> tuple[MCTSGame, Oracle]:
    return MCTSTicTacToe(), TestOracle()


@dataclass
class PolicyData:
    deterministic: bool
    number_of_simulations: int


@pytest.fixture
def policy(tic_tac_toe: tuple[MCTSGame, Oracle], policy_data: PolicyData) -> MCTSPolicy:
    if policy_data.deterministic:
        PolicyClass = MCTSPolicy
    else:
        PolicyClass = NonDeterministicMCTSPolicy

    game, oracle = tic_tac_toe
    return PolicyClass(
        game = game,
        oracle = oracle,
        mcts = MCTSConfiguration(
            number_of_simulations = policy_data.number_of_simulations,
            pucb_constant = 1.0
        )
    )


@pytest.mark.parametrize('policy_data', [
    PolicyData(deterministic = True, number_of_simulations = 10),
    PolicyData(deterministic = False, number_of_simulations = 10)
])
@pytest.mark.quick
def test_mcts_call(tic_tac_toe: tuple[MCTSGame, Oracle], policy: MCTSPolicy):
    game, _ = tic_tac_toe
    state = game.make_initial_state()
    perspective = Perspective(Player.O)
    policy(MCTSObservation(state, perspective), perspective.get_action_space(state))


@pytest.mark.parametrize('policy_data', [
    PolicyData(deterministic = True, number_of_simulations = 125)
])
@pytest.mark.slow
def test_mcts_run(tic_tac_toe: tuple[MCTSGame, Oracle], policy: MCTSPolicy):
    game, _ = tic_tac_toe
    game_loop = make_game_loop(
        game = game,
        policies = {
            Player.O: RandomPolicy(),
            Player.X: policy
        }
    )
    game_loop.run()


class FirstMoveOracle(Oracle):

    def get_probabilities(
        self,
        observation: MCTSObservation,
        legal_mask: LegalMask
    ) -> Probabilities:
        return (1.0,) + (0.0,) * 8  # always choose the first move

    def get_value(self, observation: MCTSObservation) -> float:
        if len(observation.action_space) == 9:  # initial state
            return -1.0
        return 1.0  # later states


@pytest.mark.quick
def test_mcts_prefers_first_move_after_two_simulations(tic_tac_toe: tuple[MCTSGame, Oracle]):
    game, _ = tic_tac_toe
    policy = NonDeterministicMCTSPolicy(
        game = game,
        oracle = FirstMoveOracle(),
        mcts = MCTSConfiguration(
            number_of_simulations = 2,
            pucb_constant = 1.0,
            temperature = 1.0
        )
    )
    state = game.make_initial_state()
    perspective = Perspective(Player.O)

    _, probabilities = policy.get_action_and_probabilities(
        MCTSObservation(state, perspective),
        temperature = 1.0
    )

    assert probabilities[0] == pytest.approx(1.0)
    assert probabilities[1:].sum() == pytest.approx(0.0)
