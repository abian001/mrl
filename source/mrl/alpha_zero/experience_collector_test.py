import os
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Generator
from typing import Literal
import time
import h5py
import pytest
from mrl.alpha_zero.oracle import Oracle, LegalMask, Probabilities
from mrl.alpha_zero.mcts import MCTSGame
from mrl.alpha_zero.mcts_observation import MCTSObservation
from mrl.alpha_zero.experience_collector import (
    CollectorConfiguration,
    make_buffer_collector,
    make_hdf5_collector,
)
from mrl.alpha_zero.mcts_test import TestOracle
from mrl.configuration.alpha_zero_configuration import MCTSConfiguration
from mrl.xiangqi.mcts_game import MCTSXiangqi
from mrl.tic_tac_toe.mcts_game import MCTSTicTacToe


@dataclass
class CorrectnessTestData:
    configuration: CollectorConfiguration
    observations: tuple[tuple[float, ...], ...]
    probabilities: tuple[tuple[float, ...], ...]
    payoffs: tuple[float, ...]


correctness_test_data: tuple[CorrectnessTestData, ...] = (
    CorrectnessTestData(
        configuration = CollectorConfiguration(
            mcts = MCTSConfiguration(
                number_of_simulations = 1,
                pucb_constant = 1.0,
                discount_factor = 1.0
            ),
            max_buffer_length = 6,
            number_of_episodes = 1,
            temperature_schedule = ((0, 1.0),),
            number_of_processes = 1
        ),
        observations = (
            (1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.),
            (1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.),
            (0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.),
            (0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0.),
            (0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.),
            (0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.)
        ),
        probabilities = (
            (0., 0., 0., 0., 1., 0., 0., 0., 0.),
            (0., 0., 1., 0., 0., 0., 0., 0., 0.),
            (1., 0., 0., 0., 0., 0., 0., 0., 0.),
            (0., 0., 0., 0., 0., 1., 0., 0., 0.),
            (0., 0., 0., 1., 0., 0., 0., 0., 0.),
            (0., 1., 0., 0., 0., 0., 0., 0., 0.)
        ),
        payoffs = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    ),
    CorrectnessTestData(
        configuration = CollectorConfiguration(
            mcts = MCTSConfiguration(
                number_of_simulations = 1,
                pucb_constant = 1.0,
                discount_factor = 1.0
            ),
            max_buffer_length = 12,
            number_of_episodes = 2,
            temperature_schedule = ((0, 1.0),),
            number_of_processes = 2
        ),
        observations = (
            (0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.),
            (1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.),
            (1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.),
            (0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.),
            (0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0.),
            (0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.)
        ) * 2,
        probabilities = (
            (0., 1., 0., 0., 0., 0., 0., 0., 0.),
            (0., 0., 0., 0., 1., 0., 0., 0., 0.),
            (0., 0., 1., 0., 0., 0., 0., 0., 0.),
            (1., 0., 0., 0., 0., 0., 0., 0., 0.),
            (0., 0., 0., 0., 0., 1., 0., 0., 0.),
            (0., 0., 0., 1., 0., 0., 0., 0., 0.)
        ) * 2,
        payoffs = (0.0, 1.0, 1.0, 1.0, 0.0, 0.0) * 2
    )
)


@pytest.mark.parametrize('test_data', correctness_test_data)
@pytest.mark.quick
def test_buffer_collector(test_data: CorrectnessTestData):
    game, oracle = tic_tac_toe()
    collector = make_buffer_collector(game, oracle, test_data.configuration)
    buffer, buffer_ready = collector.collect()
    assert buffer_ready
    records = tuple(buffer[index] for index in range(len(buffer)))
    assert tuple(tuple(record[0]) for record in records) == test_data.observations
    assert tuple(tuple(record[1]) for record in records) == test_data.probabilities
    assert tuple(record[2] for record in records) == test_data.payoffs


@dataclass
class Hdf5TestData:
    configuration: CollectorConfiguration
    file_prefix: str
    expected_records: int
    expected_observation_size: int
    expected_probability_size: int

    @property
    def collect_file(self) -> str:
        return f'{self.file_prefix}.h5'

    @property
    def data_file(self) -> str:
        if self.configuration.number_of_processes == 1:
            return self.collect_file
        return f'{self.file_prefix}_0.h5'

    @property
    def expected_shapes(self) -> tuple[tuple[int, int], ...]:
        return (
            (self.expected_records, self.expected_observation_size),
            (self.expected_records, self.expected_probability_size),
            (self.expected_records, 1)
        )

    def clear_files(self):
        for path in Path.cwd().iterdir():
            if path.is_file() and path.name.startswith(self.file_prefix):
                path.unlink()


@pytest.fixture
def hdf5_test_data(number_of_processes: int) -> Generator[Hdf5TestData, None, None]:
    data = Hdf5TestData(
        configuration = CollectorConfiguration(
            mcts = MCTSConfiguration(
                number_of_simulations = 1,
                pucb_constant = 1.0,
                discount_factor = 1.0
            ),
            max_buffer_length = 5,
            number_of_episodes = 2,
            temperature_schedule = ((0, 1.0),),
            number_of_processes = number_of_processes
        ),
        file_prefix = 'experience_collector_test_hdf5',
        expected_records = 14,
        expected_observation_size = 18,
        expected_probability_size = 9
    )
    yield data
    data.clear_files()


@pytest.mark.parametrize('number_of_processes', [1, 2])
@pytest.mark.quick
def test_hd5f_collector(hdf5_test_data: Hdf5TestData):
    game, oracle = tic_tac_toe()
    collector = make_hdf5_collector(game, oracle, hdf5_test_data.configuration)
    collector.collect(hdf5_test_data.collect_file)
    _assert_hdf5_has_shapes(hdf5_test_data.data_file, hdf5_test_data.expected_shapes)


def _assert_hdf5_has_shapes(file_path: str, expected_shapes: tuple[tuple[int, int], ...]):
    assert os.path.exists(file_path)
    with h5py.File(file_path, 'r') as hdf5_file:
        observations = hdf5_file['observations']
        probabilities = hdf5_file['probabilities']
        payoffs = hdf5_file['payoffs']
        assert isinstance(observations, h5py.Dataset)
        assert isinstance(probabilities, h5py.Dataset)
        assert isinstance(payoffs, h5py.Dataset)
        shapes = (
            observations.shape,  # pylint: disable=no-member
            probabilities.shape,  # pylint: disable=no-member
            payoffs.shape  # pylint: disable=no-member
        )
        assert shapes == expected_shapes


@dataclass
class PerformanceTestData:
    configuration: CollectorConfiguration
    game_name: Literal['TicTacToe'] | Literal['Xiangqi']
    max_duration_in_seconds: float
    _game: MCTSGame | None = None
    _oracle: Oracle | None = None

    @property
    def game(self) -> MCTSGame:
        if self._game is None:
            self._game, self._oracle = self._make_game_and_oracle()
        return self._game

    @property
    def oracle(self) -> Oracle:
        if self._oracle is None:
            self._game, self._oracle = self._make_game_and_oracle()
        return self._oracle

    def _make_game_and_oracle(self) -> tuple[MCTSGame, Oracle]:
        if self.game_name == 'TicTacToe':
            return tic_tac_toe()
        if self.game_name == 'Xiangqi':
            return xiangqi()
        assert False, 'Unexpected game name in PerformanceTestData'


performance_test_data: tuple[PerformanceTestData, ...] = (
    PerformanceTestData(
        configuration = CollectorConfiguration(
            mcts = MCTSConfiguration(
                number_of_simulations = 125,
                pucb_constant = 1.0,
                discount_factor = 1.0
            ),
            max_buffer_length = 10000,
            number_of_episodes = 180,
            temperature_schedule = ((0, 1.0),),
            number_of_processes = 1
        ),
        game_name = 'TicTacToe',
        max_duration_in_seconds = 5.30
    ),
    PerformanceTestData(
        configuration = CollectorConfiguration(
            mcts = MCTSConfiguration(
                number_of_simulations = 125,
                pucb_constant = 1.0,
                discount_factor = 1.0
            ),
            max_buffer_length = 10000,
            number_of_episodes = 180,
            temperature_schedule = ((0, 1.0),),
            number_of_processes = 4
        ),
        game_name = 'TicTacToe',
        max_duration_in_seconds = 1.73
    ),
    PerformanceTestData(
        configuration = CollectorConfiguration(
            mcts = MCTSConfiguration(
                number_of_simulations = 1,
                pucb_constant = 1.0,
                discount_factor = 1.0
            ),
            max_buffer_length = 10000,
            number_of_episodes = 36,
            temperature_schedule = ((0, 1.0),),
            number_of_processes = 1
        ),
        game_name = 'Xiangqi',
        max_duration_in_seconds = 20.00
    ),
    PerformanceTestData(
        configuration = CollectorConfiguration(
            mcts = MCTSConfiguration(
                number_of_simulations = 1,
                pucb_constant = 1.0,
                discount_factor = 1.0
            ),
            max_buffer_length = 10000,
            number_of_episodes = 36,
            temperature_schedule = ((0, 1.0),),
            number_of_processes = 4
        ),
        game_name = 'Xiangqi',
        max_duration_in_seconds = 7.80
    )
)


@pytest.mark.parametrize('test_data', performance_test_data)
@pytest.mark.performance
def test_collector_performance(test_data: PerformanceTestData):
    collector = make_buffer_collector(test_data.game, test_data.oracle, test_data.configuration)
    start_time = time.perf_counter()
    collector.collect()
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(duration)
    assert duration < test_data.max_duration_in_seconds


def tic_tac_toe() -> tuple[MCTSGame, Oracle]:
    return MCTSTicTacToe(), TestOracle()


def xiangqi() -> tuple[MCTSGame, Oracle]:
    return MCTSXiangqi(), TestXiangqiOracle()


class TestXiangqiOracle(Oracle):

    def get_probabilities(
        self,
        observation: MCTSObservation,
        legal_mask: LegalMask
    ) -> Probabilities:
        return tuple(1.0 for i in range(127))

    def get_value(self, observation: MCTSObservation) -> float:
        return 0.5
