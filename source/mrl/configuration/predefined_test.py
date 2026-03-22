import pytest
from mrl.configuration.factory import ObjectConfiguration
from mrl.configuration.gui_factory import make_gui
from mrl.configuration.game_factories import make_stdin_policy, make_game, make_policy
from mrl.configuration.game_runner_configuration import PolicyConfiguration
from mrl.configuration.mcts_factories import make_oracle
from mrl.game.game import Policy
from mrl.tic_tac_toe.game import TicTacToe, TicTacToeStdin
from mrl.tic_tac_toe.mcts_game import MCTSTicTacToe
from mrl.straight_four.game import StraightFour, StraightFourStdin
from mrl.straight_four.mcts_game import MCTSStraightFour
from mrl.xiangqi.game import Xiangqi, XiangqiStdin
from mrl.xiangqi.enumerable_game import XiangqiStdin as EnumerableXiangqiStdin
from mrl.xiangqi.mcts_game import MCTSXiangqi
from mrl.rock_paper_scissors.game import RockPaperScissors, RockPaperScissorsStdin
from mrl.test_utils.policies import RandomPolicy, FirstChoicePolicy, ManualPolicy
from mrl.alpha_zero.models import OpenSpielMLP, OpenSpielConv, OpenSpielResnet
from mrl.alpha_zero.random_rollout import RandomRollout
from mrl.alpha_zero.mcts import MCTSPolicy, NonDeterministicMCTSPolicy, MemoryfullMCTSPolicy
from mrl.alpha_zero.oracle import DeterministicOraclePolicy, StochasticOraclePolicy
from mrl.tkinter_gui.gui import Gui


@pytest.mark.parametrize('game_class', [
    TicTacToe,
    MCTSTicTacToe,
    StraightFour,
    MCTSStraightFour
])
@pytest.mark.quick
def test_game_factory(game_class: type):
    instance = make_game(
        ObjectConfiguration.model_validate({
            'name': game_class.__name__,
            'first_player': 'X'
        })
    )
    assert isinstance(instance, game_class)


@pytest.mark.parametrize('game_class', [
    Xiangqi,
    MCTSXiangqi
])
@pytest.mark.quick
def test_xiangqi_factory(game_class: type):
    instance = make_game(
        ObjectConfiguration.model_validate({
            'name': game_class.__name__,
            'first_player': 'Black'
        })
    )
    assert isinstance(instance, game_class)


@pytest.mark.quick
def test_rock_paper_scissors_factory():
    instance = make_game(
        ObjectConfiguration.model_validate({
            'name': 'RockPaperScissors',
            'step_limit': 5
        })
    )
    assert isinstance(instance, RockPaperScissors)


@pytest.mark.parametrize('policy_class', [
    RandomPolicy,
    FirstChoicePolicy,
    ManualPolicy,
    TicTacToeStdin,
    StraightFourStdin,
    XiangqiStdin,
    EnumerableXiangqiStdin
])
@pytest.mark.quick
def test_policy_factory(policy_class: type):
    instance = make_policy(ObjectConfiguration(name = policy_class.__name__))
    assert isinstance(instance, policy_class)


@pytest.mark.quick
def test_stdin_rock_paper_scissors_factory():
    instance = make_policy(
        ObjectConfiguration.model_validate({
            'name': 'RockPaperScissorsStdin',
            'player': 'O'
        })
    )
    assert isinstance(instance, RockPaperScissorsStdin)


@pytest.mark.quick
def test_open_spiel_mlp_factory():
    instance = make_oracle(
        ObjectConfiguration.model_validate({
            'name': 'OpenSpielMLP',
            'capacity': {
                'input_size': '1',
                'output_size': '1',
                'nn_width': '1',
                'nn_depth': '1'
            }
        })
    )
    assert isinstance(instance, OpenSpielMLP)


@pytest.mark.parametrize('oracle_class', [OpenSpielConv, OpenSpielResnet])
@pytest.mark.quick
def test_open_spiel_cnn_factory(oracle_class: type):
    instance = make_oracle(
        ObjectConfiguration.model_validate({
            'name': oracle_class.__name__,
            'capacity': {
                'input_shape': ['1', '1', '1'],
                'output_size': '1',
                'nn_width': '1',
                'nn_depth': '1'
            }
        })
    )
    assert isinstance(instance, oracle_class)


@pytest.mark.quick
def test_random_rollout_factory():
    instance = make_oracle(
        ObjectConfiguration.model_validate({
            'name': 'RandomRollout',
            'number_of_rollouts': '1'
        }),
        extra_arguments = {'game': MCTSTicTacToe()}
    )
    assert isinstance(instance, RandomRollout)


@pytest.mark.parametrize('policy_class', [
    MCTSPolicy,
    NonDeterministicMCTSPolicy,
    MemoryfullMCTSPolicy
])
@pytest.mark.quick
def test_oracle_policy_factory(policy_class: type):
    game = MCTSTicTacToe()
    instance = make_policy(
        PolicyConfiguration.model_validate({
            'name': policy_class.__name__,
            'mcts': {
                'number_of_simulations': '25',
                'pucb_constant': '1.0',
                'discount_factor': '1.0',
                'temperature': '1.0'
            }
        }),
        extra_arguments = {
            'game': game,
            'oracle': RandomRollout(game)
        }
    )
    assert isinstance(instance, policy_class)


@pytest.mark.parametrize('policy_class', [
    DeterministicOraclePolicy,
    StochasticOraclePolicy
])
@pytest.mark.quick
def test_simple_oracle_policy_factory(policy_class: type):
    game = MCTSTicTacToe()
    instance = make_policy(
        ObjectConfiguration(name = policy_class.__name__),
        extra_arguments = {
            'game': game,
            'oracle': RandomRollout(game)
        }
    )
    assert isinstance(instance, policy_class)


@pytest.mark.parametrize('game_class, module_name', [
    (TicTacToe, 'tic_tac_toe'),
    (MCTSTicTacToe, 'tic_tac_toe'),
    (StraightFour, 'straight_four'),
    (MCTSStraightFour, 'straight_four'),
    (Xiangqi, 'xiangqi'),
    (MCTSXiangqi, 'xiangqi')
])
@pytest.mark.quick
def test_make_gui(game_class: type, module_name: str, monkeypatch: pytest.MonkeyPatch):

    class MockGui(Gui):
        def __init__(self):  # pylint: disable=super-init-not-called
            pass

    monkeypatch.setattr(
        f'mrl.{module_name}.tkinter_gui.make_gui',
        lambda game_configuration: MockGui()
    )
    instance = make_gui(ObjectConfiguration(name = game_class.__name__))
    assert isinstance(instance, Gui)


@pytest.mark.parametrize('game_class', [
    TicTacToe,
    MCTSTicTacToe,
    StraightFour,
    MCTSStraightFour,
    Xiangqi,
    MCTSXiangqi
])
@pytest.mark.quick
def test_make_stdin(game_class: type):
    instance = make_stdin_policy(ObjectConfiguration(name = game_class.__name__))
    assert isinstance(instance, Policy)
