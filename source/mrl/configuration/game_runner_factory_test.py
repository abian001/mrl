import yaml
import pytest
from mrl.configuration.game_runner_configuration import GameRunnerConfiguration
from mrl.configuration.game_runner_factory import GameRunnerFactory
from mrl.game.game import Policy
from mrl.tic_tac_toe.game import Player
from mrl.tic_tac_toe.mcts_game import MCTSTicTacToe
from mrl.rock_paper_scissors.game import RockPaperScissorsStdin, Player as RPSPlayer
from mrl.alpha_zero.mcts import MCTSPolicy, NonDeterministicMCTSPolicy
from mrl.test_utils.policies import ManualPolicy


@pytest.fixture
def factory() -> GameRunnerFactory:
    return GameRunnerFactory()


@pytest.fixture
def configuration() -> GameRunnerConfiguration:
    return GameRunnerConfiguration.model_validate(yaml.safe_load("""
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
        """))


@pytest.mark.quick
def test_make_policies(
    factory: GameRunnerFactory,
    configuration: GameRunnerConfiguration,
) -> None:
    game = factory.make_runner_game(configuration)
    policies = factory.make_policies(configuration, game)
    observed_players = factory.make_observed_players(configuration, game)

    assert isinstance(game, MCTSTicTacToe)
    assert Player.X in policies
    assert Player.O in policies
    assert isinstance(policies[Player.X], NonDeterministicMCTSPolicy)
    assert isinstance(policies[Player.O], MCTSPolicy)
    assert observed_players == (Player.O, Player.X)
    assert configuration.evaluation_config.number_of_tests == 1
    assert configuration.evaluation_config.buckets == ((float("-inf"), float("+inf")),)


@pytest.fixture
def terminal_configuration() -> GameRunnerConfiguration:
    return GameRunnerConfiguration.model_validate(yaml.safe_load("""
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
    """))


@pytest.mark.quick
def test_make_terminal_policies(
    factory: GameRunnerFactory,
    terminal_configuration: GameRunnerConfiguration,
) -> None:
    game = factory.make_runner_game(terminal_configuration)
    policies = factory.make_policies(terminal_configuration, game)
    replaced_policies: dict[RPSPlayer, Policy] = factory.make_terminal_policies(
        terminal_configuration,
        policies,
    )
    observed_players = factory.make_observed_players(terminal_configuration, game)

    assert isinstance(policies[RPSPlayer.O], ManualPolicy)
    assert isinstance(policies[RPSPlayer.X], RockPaperScissorsStdin)
    assert isinstance(replaced_policies[RPSPlayer.O], RockPaperScissorsStdin)
    assert isinstance(replaced_policies[RPSPlayer.X], RockPaperScissorsStdin)
    assert observed_players == (RPSPlayer.X,)
    assert terminal_configuration.evaluation_config.number_of_tests == 2
    assert terminal_configuration.evaluation_config.buckets == (
        (float("-inf"), 0.0),
        (0.0, float("+inf")),
    )
