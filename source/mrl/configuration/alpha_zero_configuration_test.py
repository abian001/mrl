from pathlib import Path
from typing import Generator
import os
import yaml
import pytest
from mrl.configuration.alpha_zero_configuration import (
    AlphaZeroConfiguration,
    InMemoryAlphaZeroConfiguration,
    OracleConfiguration
)
from mrl.configuration.factory import ObjectConfiguration
from mrl.tic_tac_toe.game import Player
from mrl.tic_tac_toe.mcts_game import MCTSTicTacToe
from mrl.alpha_zero.models import OpenSpielConv


@pytest.fixture
def configuration(memory_type: str, config_file_path: str) -> dict:
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
                player: 'O'
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
def _clear_hd5f_config(config_file_path: str) -> Generator[None, None, None]:
    yield
    if os.path.exists(config_file_path):
        os.remove(config_file_path)


@pytest.mark.parametrize('memory_type', ['InMemory', 'HDF5'])
@pytest.mark.parametrize('config_file_path', ['test_config.yaml'])
@pytest.mark.quick
def test_load_configuration(memory_type: str, configuration: dict, _clear_hd5f_config: None):
    config = AlphaZeroConfiguration.model_validate({'alpha_zero': configuration})
    assert config.alpha_zero.report_generator is not None
    assert config.alpha_zero.report_generator.buckets is not None
    config.model_dump()  # serialization succeeds
    assert config.alpha_zero.type == memory_type
    assert config.alpha_zero.collector.temperature_schedule == ((0, 1.0),)
    assert config.alpha_zero.report_generator.oracle_led_players == (Player.X,)
    assert config.alpha_zero.oracle_file_path == Path("workspace/tic_tac_toe_model")
    assert config.alpha_zero.report_generator.buckets[0][0] == float("-inf")
    assert config.alpha_zero.workspace_path == Path("workspace")
    assert isinstance(config.alpha_zero.game, MCTSTicTacToe)
    assert isinstance(config.alpha_zero.oracle, OpenSpielConv)


@pytest.mark.parametrize('memory_type', ['InMemory'])
@pytest.mark.parametrize('config_file_path', ['test_config.yaml'])
@pytest.mark.quick
def test_validate_top_level_output_size_mismatch(
    configuration: dict,
    memory_type: str,
    config_file_path: str,
    _clear_hd5f_config: None
):
    _ = memory_type, config_file_path
    del configuration['oracle']['capacity']['output_size']
    configuration['oracle']['output_size'] = 8

    with pytest.raises(TypeError, match = 'Mismatched output sizes between the game and the model'):
        AlphaZeroConfiguration.model_validate({'alpha_zero': configuration})


@pytest.mark.parametrize('memory_type', ['InMemory'])
@pytest.mark.parametrize('config_file_path', ['test_config.yaml'])
@pytest.mark.quick
def test_validate_capacity_output_size_mismatch(
    configuration: dict,
    memory_type: str,
    config_file_path: str,
    _clear_hd5f_config: None
):
    _ = memory_type, config_file_path
    configuration['oracle']['capacity']['output_size'] = 8

    with pytest.raises(TypeError, match = 'Mismatched output sizes between the game and the model'):
        AlphaZeroConfiguration.model_validate({'alpha_zero': configuration})


@pytest.mark.parametrize('memory_type', ['InMemory'])
@pytest.mark.parametrize('config_file_path', ['test_config.yaml'])
@pytest.mark.quick
def test_validate_report_generator_oracle_led_players(
    configuration: dict,
    memory_type: str,
    config_file_path: str,
    _clear_hd5f_config: None
):
    _ = memory_type, config_file_path
    configuration['report_generator']['oracle_led_players'] = ['missing']

    with pytest.raises(TypeError, match = 'Invalid player missing'):
        AlphaZeroConfiguration.model_validate({'alpha_zero': configuration})


@pytest.mark.parametrize('memory_type', ['InMemory'])
@pytest.mark.parametrize('config_file_path', ['test_config.yaml'])
@pytest.mark.quick
def test_game_requires_mcts_game(
    configuration: dict,
    memory_type: str,
    config_file_path: str,
    _clear_hd5f_config: None,
    monkeypatch: pytest.MonkeyPatch
):
    _ = memory_type, config_file_path
    monkeypatch.setattr(
        'mrl.configuration.alpha_zero_configuration.make_mcts_game',
        lambda *_args, **_kwargs: object()
    )

    with pytest.raises(TypeError, match = 'is not an mcts game'):
        _ = AlphaZeroConfiguration.model_validate({'alpha_zero': configuration})


@pytest.mark.parametrize('memory_type', ['InMemory'])
@pytest.mark.parametrize('config_file_path', ['test_config.yaml'])
@pytest.mark.quick
def test_oracle_requires_oracle_type(
    configuration: dict,
    memory_type: str,
    config_file_path: str,
    _clear_hd5f_config: None,
    monkeypatch: pytest.MonkeyPatch
):
    _ = memory_type, config_file_path
    config = AlphaZeroConfiguration.model_validate({'alpha_zero': configuration})
    monkeypatch.setattr(
        'mrl.configuration.alpha_zero_configuration.make_oracle',
        lambda *_args, **_kwargs: object()
    )

    with pytest.raises(TypeError, match = 'is not a valid oracle class'):
        _ = config.alpha_zero.oracle


class _PerspectiveWithDimension:

    def __init__(self, action_space_dimension: int):
        self.action_space_dimension = action_space_dimension

    def get_observation(self, _state):
        raise NotImplementedError

    def get_action_space(self, _state):
        raise NotImplementedError

    def get_payoff(self, _state):
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


@pytest.mark.parametrize('memory_type', ['InMemory'])
@pytest.mark.parametrize('config_file_path', ['test_config.yaml'])
@pytest.mark.quick
def test_validate_same_action_space_size_for_all_perspectives(
    configuration: dict,
    memory_type: str,
    config_file_path: str,
    _clear_hd5f_config: None,
    monkeypatch: pytest.MonkeyPatch
):
    _ = memory_type, config_file_path
    monkeypatch.setattr(
        'mrl.configuration.alpha_zero_configuration.make_mcts_game',
        lambda *_args, **_kwargs: _InconsistentMCTSGame()
    )

    with pytest.raises(TypeError, match = 'return different action space sizes'):
        AlphaZeroConfiguration.model_validate({'alpha_zero': configuration})


@pytest.mark.quick
def test_save_requires_output_file():
    configuration = InMemoryAlphaZeroConfiguration.model_construct(
        game_configuration = ObjectConfiguration(name = 'MCTSTicTacToe'),
        oracle_configuration = OracleConfiguration(
            name = 'OpenSpielConv',
            file_path = Path('tic_tac_toe_model')
        ),
        gui_configuration = None,
        trainer = None,
        collector = None,
        number_of_epochs = 1,
        report_generator = None,
        config_file_path = None,
        workspace_path = Path('workspace'),
        evaluation = None,
        type = 'InMemory'
    )

    with pytest.raises(
        ValueError,
        match = 'No output file defined when saving alpha zero configuration'
    ):
        configuration.save()
