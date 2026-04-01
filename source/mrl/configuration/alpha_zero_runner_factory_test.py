from pathlib import Path
from typing import Generator
import os
import yaml
import pytest
from mrl.configuration.factory import ObjectConfiguration
from mrl.configuration.alpha_zero_runner_configuration import AlphaZeroRunnerConfiguration
from mrl.configuration.alpha_zero_runner_factory import AlphaZeroRunnerFactory
from mrl.tkinter_gui.gui import Gui
from mrl.tic_tac_toe.game import Player
from mrl.tic_tac_toe.mcts_game import MCTSTicTacToe
from mrl.alpha_zero.models import OpenSpielConv
from mrl.alpha_zero.mcts import MCTSPolicy
from mrl.alpha_zero.oracle import DeterministicOraclePolicy


class _PerspectiveWithDimension:

    def __init__(self, action_space_dimension: int):
        self.action_space_dimension = action_space_dimension

    def get_observation(self, _state):
        raise NotImplementedError

    def get_action_space(self, _state):
        raise NotImplementedError

    def get_reward(self, _state):
        raise NotImplementedError


class _InconsistentMCTSGame:

    def make_initial_state(self):
        raise NotImplementedError

    def get_players(self):
        return (Player.X, Player.O)

    def get_perspectives(self):
        return {
            Player.X: _PerspectiveWithDimension(9),
            Player.O: _PerspectiveWithDimension(8)
        }

    def update(self, _state, _action):
        raise NotImplementedError

    def restore(self, _observation):
        raise NotImplementedError


@pytest.fixture
def factory() -> AlphaZeroRunnerFactory:
    return AlphaZeroRunnerFactory()


@pytest.fixture
def configuration_data(
    memory_type: str,
    config_file_path: str,
) -> dict:
    return yaml.safe_load(f"""
        type: {memory_type}
        game:
            name: MCTSTicTacToe
            first_player: X
        oracle:
            name: OpenSpielConv
            capacity:
                input_shape: [1, 1, 1]
                output_size: 9
                nn_depth: 1
                nn_width: 1
            file_path: tic_tac_toe_model
        collector:
            number_of_processes: 1
            mcts:
                number_of_simulations: 25
                pucb_constant: 1.0
                discount_factor: 1.0
            max_buffer_length: 1000
            number_of_episodes: 1
            temperature_schedule:
                - [0, 1.0] 
        trainer:
            batch_size: 32
            max_training_epochs: 1
            early_stop_loss: 1e-3
            loss_observer: null
            learning_rate: 1e-3
            loading_workers: 1
        report_generator:
            oracle_led_players: ['X']
            number_of_tests: 10
            buckets:
                - [-inf, 0.25]
                - [0.25, 0.75]
                - [0.75, +inf]
        manual_play:
            manual_player: 'O'
            autonomous_policy:
                name: MCTSPolicy
                mcts:
                    number_of_simulations: 25
                    pucb_constant: 1.0
                    discount_factor: 1.0
        number_of_epochs: 10 
        evaluation:
            episodes: 10
            max_old_models: 5
            policy:
                name: DeterministicOraclePolicy
        hdf5_path_prefix: data_file
        server_hostname: 127.0.0.1
        server_port: 8888
        config_file_path: {config_file_path}
    """)


@pytest.fixture
def configuration(
    configuration_data: dict,
) -> AlphaZeroRunnerConfiguration:
    return AlphaZeroRunnerConfiguration.model_validate(configuration_data)


@pytest.fixture
def default_manual_play_configuration(
    configuration_data: dict,
) -> AlphaZeroRunnerConfiguration:
    configuration_data = configuration_data.copy()
    del configuration_data['manual_play']
    return AlphaZeroRunnerConfiguration.model_validate(configuration_data)


@pytest.fixture
def _clear_hd5f_config(config_file_path: str) -> Generator[None, None, None]:
    yield
    if os.path.exists(config_file_path):
        os.remove(config_file_path)


@pytest.mark.parametrize('memory_type', ['InMemory', 'HDF5'])
@pytest.mark.parametrize('config_file_path', ['test_config.yaml'])
@pytest.mark.quick
def test_make_context_and_manual_play(
    factory: AlphaZeroRunnerFactory,
    memory_type: str,
    configuration: AlphaZeroRunnerConfiguration,
    _clear_hd5f_config: None,
) -> None:
    context = factory.make_context(configuration.alpha_zero)
    manual_player = factory.make_manual_player(configuration, context)
    autonomous_policy = factory.make_autonomous_policy(configuration, context)

    assert context.report_generator is not None
    assert context.report_generator.buckets is not None
    assert context.type == memory_type
    assert context.collector.temperature_schedule == ((0, 1.0),)
    assert context.report_generator.oracle_led_players == (Player.X,)
    assert context.oracle_file_path == Path("workspace/tic_tac_toe_model")
    assert context.report_generator.buckets[0][0] == float("-inf")
    assert manual_player == Player.O
    assert isinstance(context.game, MCTSTicTacToe)
    assert isinstance(context.oracle, OpenSpielConv)
    assert isinstance(autonomous_policy, MCTSPolicy)


@pytest.mark.parametrize('memory_type', ['InMemory'])
@pytest.mark.parametrize('config_file_path', ['test_config.yaml'])
@pytest.mark.quick
def test_default_manual_play(
    factory: AlphaZeroRunnerFactory,
    config_file_path: str,
    default_manual_play_configuration: AlphaZeroRunnerConfiguration,
) -> None:
    context = factory.make_context(default_manual_play_configuration.alpha_zero)
    manual_player = factory.make_manual_player(default_manual_play_configuration, context)
    autonomous_policy = factory.make_autonomous_policy(
        default_manual_play_configuration,
        context,
    )
    assert isinstance(manual_player, Player)
    assert isinstance(autonomous_policy, DeterministicOraclePolicy)
    assert not os.path.exists(config_file_path)


@pytest.mark.parametrize('memory_type', ['InMemory'])
@pytest.mark.parametrize('config_file_path', ['test_config.yaml'])
@pytest.mark.quick
def test_make_manual_gui(
    factory: AlphaZeroRunnerFactory,
    configuration: AlphaZeroRunnerConfiguration,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class MockGui(Gui):
        def __init__(self):  # pylint: disable=super-init-not-called
            pass

    configuration.gui_configuration = ObjectConfiguration.model_validate({
        'name': 'make_gui',
        'module': 'mrl.tic_tac_toe.tkinter_gui',
    })
    original_make_object = __import__(
        'mrl.configuration.runner_factories',
        fromlist = ['make_object']
    ).make_object

    def make_gui_only(object_configuration, *args, **kwargs):
        if object_configuration.name == 'make_gui':
            return MockGui()
        return original_make_object(object_configuration, *args, **kwargs)

    monkeypatch.setattr(
        'mrl.configuration.runner_factories.make_object',
        make_gui_only
    )
    context = factory.make_context(configuration.alpha_zero)
    gui = factory.make_manual_gui(configuration, context)
    assert isinstance(gui, Gui)


@pytest.mark.parametrize('memory_type', ['InMemory'])
@pytest.mark.parametrize('config_file_path', ['test_config.yaml'])
@pytest.mark.quick
def test_validate_top_level_output_size_mismatch(
    factory: AlphaZeroRunnerFactory,
    configuration_data: dict,
    _clear_hd5f_config: None,
) -> None:
    del configuration_data['oracle']['capacity']['output_size']
    configuration_data['oracle']['output_size'] = 8
    configuration = AlphaZeroRunnerConfiguration.model_validate(configuration_data)

    with pytest.raises(TypeError, match = 'Mismatched output sizes between the game and the model'):
        factory.make_context(configuration.alpha_zero)


@pytest.mark.parametrize('memory_type', ['InMemory'])
@pytest.mark.parametrize('config_file_path', ['test_config.yaml'])
@pytest.mark.quick
def test_validate_capacity_output_size_mismatch(
    factory: AlphaZeroRunnerFactory,
    configuration_data: dict,
    _clear_hd5f_config: None,
) -> None:
    configuration_data['oracle']['capacity']['output_size'] = 8
    configuration = AlphaZeroRunnerConfiguration.model_validate(configuration_data)

    with pytest.raises(TypeError, match = 'Mismatched output sizes between the game and the model'):
        factory.make_context(configuration.alpha_zero)


@pytest.mark.parametrize('memory_type', ['InMemory'])
@pytest.mark.parametrize('config_file_path', ['test_config.yaml'])
@pytest.mark.quick
def test_validate_report_generator_oracle_led_players(
    factory: AlphaZeroRunnerFactory,
    configuration_data: dict,
    _clear_hd5f_config: None,
) -> None:
    configuration_data['report_generator']['oracle_led_players'] = ['missing']
    configuration = AlphaZeroRunnerConfiguration.model_validate(configuration_data)

    with pytest.raises(TypeError, match = 'Invalid player missing'):
        factory.make_context(configuration.alpha_zero)


@pytest.mark.parametrize('memory_type', ['InMemory'])
@pytest.mark.parametrize('config_file_path', ['test_config.yaml'])
@pytest.mark.quick
def test_game_requires_mcts_game(
    factory: AlphaZeroRunnerFactory,
    configuration_data: dict,
    _clear_hd5f_config: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        'mrl.configuration.runner_factories.make_object',
        lambda *_args, **_kwargs: object()
    )
    configuration = AlphaZeroRunnerConfiguration.model_validate(configuration_data)

    with pytest.raises(TypeError, match = 'Invalid game class'):
        factory.make_context(configuration.alpha_zero)


@pytest.mark.parametrize('memory_type', ['InMemory'])
@pytest.mark.parametrize('config_file_path', ['test_config.yaml'])
@pytest.mark.quick
def test_oracle_requires_oracle_type(
    factory: AlphaZeroRunnerFactory,
    configuration_data: dict,
    _clear_hd5f_config: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = AlphaZeroRunnerConfiguration.model_validate(configuration_data)
    original_make_object = __import__(
        'mrl.configuration.runner_factories',
        fromlist = ['make_object']
    ).make_object

    def make_bad_oracle(object_configuration, *args, **kwargs):
        if object_configuration.name == configuration.alpha_zero.oracle_configuration.name:
            return object()
        return original_make_object(object_configuration, *args, **kwargs)

    monkeypatch.setattr(
        'mrl.configuration.runner_factories.make_object',
        make_bad_oracle
    )

    with pytest.raises(TypeError, match = 'Invalid oracle class'):
        factory.make_context(configuration.alpha_zero)


@pytest.mark.parametrize('memory_type', ['InMemory'])
@pytest.mark.parametrize('config_file_path', ['test_config.yaml'])
@pytest.mark.quick
def test_validate_same_action_space_size_for_all_perspectives(
    factory: AlphaZeroRunnerFactory,
    configuration_data: dict,
    _clear_hd5f_config: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        'mrl.configuration.alpha_zero_runner_factory.make_mcts_game',
        lambda *_args, **_kwargs: _InconsistentMCTSGame()
    )
    configuration = AlphaZeroRunnerConfiguration.model_validate(configuration_data)

    with pytest.raises(TypeError, match = 'return different action space sizes'):
        factory.make_context(configuration.alpha_zero)
