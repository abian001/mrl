from typing import Protocol, TypeVar, Any, Generic, Iterator
from abc import abstractmethod
from collections import deque
import multiprocessing
import os
import time
import pickle
import numpy as np
import h5py
from pydantic import BaseModel, field_serializer
from mrl.game.game import Player, Reward, FinalCheckable, HasActivePlayer
from mrl.alpha_zero.oracle import Oracle, Payoff, Probabilities
from mrl.alpha_zero.mcts import NonDeterministicMCTSPolicy, MCTSGame, MCTSConfiguration


RewardExperience = tuple[np.ndarray, Probabilities, Reward]
RewardBuffers = dict[Player, list[RewardExperience]]
PayoffExperience = tuple[np.ndarray, Probabilities, Payoff]


class CollectorConfiguration(BaseModel):
    mcts: MCTSConfiguration
    max_buffer_length: int
    number_of_episodes: int
    temperature_schedule: tuple[tuple[int, float]]
    number_of_processes: int

    @field_serializer("temperature_schedule")
    def temperature_schedule_to_list(
        self,
        temperature_schedule: tuple[tuple[int, float]],
        _info: Any
    ):
        return [[x[0], x[1]] for x in temperature_schedule]


def make_buffer_collector(game: MCTSGame, model: Oracle, config: CollectorConfiguration):
    if config.number_of_processes > 1:
        return SharedProcessCollector(game, model, config)
    return SingleBufferCollector(game, model, config)


def make_hdf5_collector(game: MCTSGame, model: Oracle, config: CollectorConfiguration):
    if config.number_of_processes > 1:
        return MultiHDF5Collector(game, model, config)
    return SingleHDF5Collector(game, model, config)


def get_hdf5_dataset(
    hdf5_file: h5py.File,
    dataset_name: str,
    file_path: str | None
) -> h5py.Dataset:
    resolved_file_path = file_path or "<unknown>"
    try:
        dataset = hdf5_file[dataset_name]
    except KeyError as error:
        raise KeyError(
            f"HDF5 file {resolved_file_path} is missing dataset {dataset_name!r}."
        ) from error
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(
            f"HDF5 entry {dataset_name!r} in file {resolved_file_path} is not a dataset."
        )
    return dataset


class TemperatureSchedule:

    def __init__(self, schedule: tuple[tuple[int, float]]):
        self.schedule = schedule

    def __call__(self, step):
        return max(
            (s for s in self.schedule if s[0] <= step),
            key = lambda x: x[0]
        )[1]


class MCTSState(FinalCheckable, HasActivePlayer, Protocol):
    pass


class ExperienceCollector(Generic[Player]):

    def __init__(
        self,
        game: MCTSGame[MCTSState, Player],
        model: Oracle,
        config: CollectorConfiguration
    ) -> None:
        self.config = config
        self.game = game
        self.policy = NonDeterministicMCTSPolicy(
            game = game,
            oracle = model,
            mcts = config.mcts
        )
        self.perspectives = self.policy.perspectives
        self.discount_factor = config.mcts.discount_factor
        self.temperature_schedule = TemperatureSchedule(self.config.temperature_schedule)

    def _play_one_episode(self, buffers: RewardBuffers[Player]) -> MCTSState:
        state = self.game.make_initial_state()
        step = 0
        while not state.is_final:
            observation = self.perspectives[state.active_player].get_observation(state)
            action, probabilities = self.policy.get_action_and_probabilities(
                observation = observation,
                temperature = self.temperature_schedule(step)
            )
            buffers[state.active_player].append(
                (observation.core, probabilities, observation.reward)
            )
            state = self.game.update(state, action)
            step += 1
        return state


Item = TypeVar("Item")


class Buffer(Protocol[Item]):

    @abstractmethod
    def append(self, item: Item, /) -> None:
        """Adds an item to the buffer"""

    @abstractmethod
    def __len__(self) -> int:
        """Size of the container"""

    @abstractmethod
    def __getitem__(self, index: int, /) -> Item:
        """Returns an item"""

    @abstractmethod
    def __iter__(self) -> Iterator[Item]:
        """Iterates through the items in the buffer"""


class LocalExperienceCollector(Protocol):

    @abstractmethod
    def collect(self) -> tuple[Buffer[Any], bool]:
        """Collect the episodes from the buffer"""

    @abstractmethod
    def save_training_data(self, file_path: str) -> None:
        """Save the training data to a file"""

    @abstractmethod
    def load_training_data(self, file_path: str) -> None:
        """Load the training data from a file"""


class DistributedExperienceCollector(Protocol):

    @abstractmethod
    def collect(self, file_path: str) -> None:
        """Collect the episodes and saves them to the file"""


class SingleProcessCollector(ExperienceCollector, Generic[Player]):

    buffer: Buffer[PayoffExperience]

    def _collect_once(self) -> None:
        buffers: RewardBuffers = {p: [] for p in self.perspectives}
        last_state = self._play_one_episode(buffers)
        self._update_buffer(buffers, last_state)

    def _update_buffer(self, buffers: RewardBuffers[Player], last_state: MCTSState) -> None:
        for (p, buffer) in buffers.items():
            payoff = self.perspectives[p].get_observation(last_state).reward
            for (observation, probabilities, reward) in reversed(buffer):
                payoff = reward + self.discount_factor * payoff
                self.buffer.append((observation, probabilities, payoff))


class SingleBufferCollector(SingleProcessCollector):

    def __init__(self, game: MCTSGame, model: Oracle, config: CollectorConfiguration):
        super().__init__(game, model, config)
        self.buffer = deque(maxlen = config.max_buffer_length)

    def collect(self) -> tuple[Buffer[Any], bool]:
        for _ in range(self.config.number_of_episodes):
            self._collect_once()
        return self.buffer, True

    def save_training_data(self, file_path: str) -> None:
        with open(file_path, "wb") as save_file:
            pickle.dump(self.buffer, save_file)

    def load_training_data(self, file_path: str) -> None:
        with open(file_path, "rb") as save_file:
            self.buffer = pickle.load(save_file)


class SingleHDF5Collector(SingleProcessCollector):

    def __init__(self, game: MCTSGame, model: Oracle, config: CollectorConfiguration):
        super().__init__(game, model, config)
        self.buffer = []
        self.file_path: str | None = None
        self.episode_count = 0

    def collect(self, file_path: str) -> None:
        self.file_path = file_path
        for _ in range(self.config.number_of_episodes):
            self._collect_once()

    def _create_hdf5_file(self, observations, probabilities, payoffs):
        with h5py.File(self.file_path, 'w') as hdf5_file:
            hdf5_file.create_dataset(
                'observations',
                maxshape = (None, observations.shape[1]),
                data = observations
            )
            hdf5_file.create_dataset(
                'probabilities',
                maxshape = (None, probabilities.shape[1]),
                data = probabilities
            )
            hdf5_file.create_dataset(
                'payoffs',
                maxshape = (None, payoffs.shape[1]),
                data = payoffs
            )

    def _append_to_hdf5_file(
        self,
        new_observations,
        new_probabilities,
        new_payoffs
    ):  # pylint: disable=no-member
        with h5py.File(self.file_path, 'a') as hdf5_file:
            observations = get_hdf5_dataset(hdf5_file, 'observations', self.file_path)
            probabilities = get_hdf5_dataset(hdf5_file, 'probabilities', self.file_path)
            payoffs = get_hdf5_dataset(hdf5_file, 'payoffs', self.file_path)

            current_size = observations.shape[0]
            new_size = current_size + new_observations.shape[0]
            observations.resize((new_size, observations.shape[1]))
            probabilities.resize((new_size, probabilities.shape[1]))
            payoffs.resize((new_size, payoffs.shape[1]))
            observations[current_size:new_size, :] = new_observations
            probabilities[current_size:new_size, :] = new_probabilities
            payoffs[current_size:new_size, :] = new_payoffs

    def _save_buffer_to_hdf5(self) -> None:
        observations = np.stack(tuple(x[0] for x in self.buffer))
        probabilities = np.stack(tuple(x[1] for x in self.buffer))
        payoffs = np.stack(tuple(np.array([x[2]], dtype = float) for x in self.buffer))
        if self.file_path is not None and os.path.exists(self.file_path):
            self._append_to_hdf5_file(observations, probabilities, payoffs)
        else:
            self._create_hdf5_file(observations, probabilities, payoffs)
        self.buffer = []

    def _collect_once(self) -> None:
        super()._collect_once()
        self.episode_count += 1
        if (
            len(self.buffer) > self.config.max_buffer_length or
            self.episode_count >= self.config.number_of_episodes
        ):
            self._save_buffer_to_hdf5()


class SharedBuffer:

    def __init__(self, config: CollectorConfiguration):
        self.manager = multiprocessing.Manager()
        self.lock = multiprocessing.Lock()
        self.process_space = config.max_buffer_length // config.number_of_processes
        self.buffer: multiprocessing.managers.ListProxy[None | PayoffExperience] = \
            self.manager.list((None,) * (self.process_space * config.number_of_processes))
        self.indices = self.manager.list(tuple(
            i * self.process_space for i in range(config.number_of_processes)
        ))
        self.filled: multiprocessing.managers.ListProxy[bool] = \
            self.manager.list((False,) * config.number_of_processes)

    def is_filled(self) -> bool:
        return all(self.filled)

    def append(self, index: int, record: PayoffExperience) -> None:
        self.buffer[self.indices[index]] = record
        self.indices[index] = (
            ((self.indices[index] + 1) % self.process_space) +
            (index * self.process_space)
        )
        self.filled[index] = \
            self.filled[index] or \
            self.indices[index] == (index * self.process_space)

    def save(self, file_path: str) -> None:
        with open(file_path, "wb") as save_file:
            pickle.dump(
                (self.process_space, list(self.buffer), list(self.indices), list(self.filled)),
                save_file
            )

    def load(self, file_path: str) -> None:
        with open(file_path, "rb") as save_file:
            data = pickle.load(save_file)
            process_space, buffer, indices, filled = data[0], data[1], data[2], data[3]
        self.buffer = self.manager.list(buffer)
        self.process_space = process_space
        self.indices = self.manager.list(indices)
        self.filled = self.manager.list(filled)


class SharedProcessCollector(ExperienceCollector, Generic[Player]):

    end_of_episode = None

    def __init__(
        self,
        game: MCTSGame[MCTSState, Player],
        model: Oracle,
        config: CollectorConfiguration
    ) -> None:
        super().__init__(game, model, config)
        self.buffer = SharedBuffer(config)
        self.episodes_per_process = \
            (self.config.number_of_episodes // self.config.number_of_processes) + \
            0 if (self.config.number_of_episodes % self.config.number_of_processes) == 0 else 1

    def collect(self) -> tuple[Buffer[Any], bool]:
        processes = []
        for i in range(self.config.number_of_processes):
            process = multiprocessing.Process(
                target = self._process_play,
                args = (i,)
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        return self.buffer.buffer, self.buffer.is_filled()

    def _process_play(self, process_index: int) -> None:
        np.random.seed(int(time.time()))
        for _ in range(self.episodes_per_process):
            buffers: RewardBuffers = {p: [] for p in self.perspectives}
            last_state = self._play_one_episode(buffers)
            self._update_buffer(buffers, last_state, process_index)

    def _update_buffer(
        self,
        buffers: RewardBuffers[Player],
        last_state: MCTSState,
        process_index: int
    ) -> None:
        for (p, buffer) in buffers.items():
            payoff = self.perspectives[p].get_observation(last_state).reward
            for (observation, probabilities, reward) in reversed(buffer):
                payoff = reward + self.discount_factor * payoff
                self.buffer.append(process_index, (observation, probabilities, payoff))

    def save_training_data(self, file_path: str) -> None:
        self.buffer.save(file_path)

    def load_training_data(self, file_path: str) -> None:
        self.buffer.load(file_path)


class MultiHDF5Collector:

    def __init__(self, game: MCTSGame, model: Oracle, config: CollectorConfiguration):
        self.game = game
        self.model = model
        self.config = config

    def collect(self, file_path: str) -> None:
        file_root = '.'.join(file_path.split('.')[:-1])
        collectors = tuple(
            SingleHDF5Collector(self.game, self.model, self.config)
            for i in range(self.config.number_of_processes)
        )
        processes = []
        for i in range(self.config.number_of_processes):
            process = multiprocessing.Process(
                target = collectors[i].collect,
                args = (f"{file_root}_{i}.h5",)
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
