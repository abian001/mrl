import yaml
import pytest
from mrl.configuration.game_runner_configuration import make_game_runner_specification
from mrl.tic_tac_toe.game import Player
from mrl.tic_tac_toe.mcts_game import MCTSTicTacToe
from mrl.rock_paper_scissors.game import RockPaperScissorsStdin, Player as RPSPlayer
from mrl.alpha_zero.mcts import MCTSPolicy, NonDeterministicMCTSPolicy
from mrl.test_utils.policies import ManualPolicy


@pytest.fixture
def configuration() -> dict:
    return yaml.safe_load("""
            game:
                name: MCTSTicTacToe
                first_player: X
            policies:
                X:
                    name: NonDeterministicMCTSPolicy
                    mcts:
                        number_of_simulations: 1
                        pucb_constant: 1.0
                        discount_factor: 1.0
                        temperature: 1.0
                    oracle:
                        name: OpenSpielMLP
                        capacity:
                            input_size: 1
                            output_size: 9
                            nn_depth: 1
                            nn_width: 1
                O: det_mcts
            shared_policies:
                det_mcts:
                    name: MCTSPolicy
                    mcts:
                        number_of_simulations: 1
                        pucb_constant: 1.0
                        discount_factor: 1.0
                    oracle: conv_9
            oracles:
                conv_9:
                    name: OpenSpielConv
                    capacity:
                        input_shape: [1, 1, 1]
                        output_size: 9
                        nn_depth: 1
                        nn_width: 1
                    file_path: tic_tac_toe_model
                    load: false
        """)


@pytest.mark.quick
def test_load_configuration(configuration: dict):
    config = make_game_runner_specification(configuration)
    assert isinstance(config.game, MCTSTicTacToe)
    assert Player.X in config.policies
    assert Player.O in config.policies
    assert isinstance(config.policies[Player.X], NonDeterministicMCTSPolicy)
    assert isinstance(config.policies[Player.O], MCTSPolicy)
    assert config.evaluation.observed_players == (Player.O, Player.X)
    assert config.evaluation.number_of_tests == 1
    assert config.evaluation.buckets == ((float("-inf"), float("+inf")),)


@pytest.mark.quick
def test_load_policy_with_player():
    config = make_game_runner_specification(
        data = yaml.safe_load("""
            game:
                name: RockPaperScissors
            policies:
                O:
                    name: ManualPolicy
                X:
                    name: RockPaperScissorsStdin
            evaluation:
                observed_players: ['X']
                number_of_tests: 2
                buckets:
                    - [-inf, -0.0]
                    - [0.0, +inf]
        """)
    )
    replaced_policies = config.policies_with_manual_replaced
    assert isinstance(config.policies[RPSPlayer.O], ManualPolicy)
    assert isinstance(config.policies[RPSPlayer.X], RockPaperScissorsStdin)
    assert isinstance(replaced_policies[RPSPlayer.O], RockPaperScissorsStdin)
    assert isinstance(replaced_policies[RPSPlayer.X], RockPaperScissorsStdin)
    assert config.evaluation.observed_players == (RPSPlayer.X,)
    assert config.evaluation.number_of_tests == 2
    assert config.evaluation.buckets == ((float("-inf"), 0.0), (0.0, float("+inf")))
