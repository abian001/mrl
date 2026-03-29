from pathlib import Path
import yaml
import pytest
from mrl.configuration.alpha_zero_configuration import AlphaZeroConfiguration


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


@pytest.mark.parametrize('memory_type', ['InMemory', 'HDF5'])
@pytest.mark.parametrize('config_file_path', ['test_config.yaml'])
@pytest.mark.quick
def test_load_configuration(memory_type: str, configuration: dict):
    config = AlphaZeroConfiguration.model_validate(configuration)
    assert config.report_generator is not None
    assert config.report_generator.buckets is not None
    config.model_dump()  # serialization succeeds
    assert config.type == memory_type
    assert config.collector.temperature_schedule == ((0, 1.0),)
    assert config.report_generator.oracle_led_players == ('X',)
    assert config.report_generator.buckets[0][0] == float("-inf")
    assert config.workspace_path == Path("workspace")
