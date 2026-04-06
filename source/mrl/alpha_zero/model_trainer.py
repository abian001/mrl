from dataclasses import dataclass
from typing import Protocol, TypeVar
from abc import abstractmethod
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from mrl.alpha_zero.experience_collector import get_hdf5_dataset


class TrainDataset(Dataset):

    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def __len__(self):
        return len(self.replay_buffer)

    def __getitem__(self, index):
        observation, probabilities, payoff = self.replay_buffer[index]
        return \
            torch.tensor(observation, dtype = torch.float32), \
            torch.tensor(probabilities, dtype = torch.float32), \
            torch.tensor(payoff, dtype = torch.float32)


class HDF5Dataset(Dataset):

    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None

        with h5py.File(self.file_path, 'r') as hdf5_file:
            observations = get_hdf5_dataset(hdf5_file, 'observations', self.file_path)
            self.length = len(observations)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open file on first access (needed for multi-worker DataLoader)
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')

        observations = get_hdf5_dataset(self.file, 'observations', self.file_path)
        probabilities = get_hdf5_dataset(self.file, 'probabilities', self.file_path)
        payoffs = get_hdf5_dataset(self.file, 'payoffs', self.file_path)

        observation = observations[idx]
        probabilities = probabilities[idx]
        payoff = payoffs[idx][0]
        return \
            torch.tensor(observation, dtype = torch.float32), \
            torch.tensor(probabilities, dtype = torch.float32), \
            torch.tensor(payoff, dtype = torch.float32)

    def __del__(self):
        if self.file is not None:
            self.file.close()


class Loss:

    def __init__(self):
        self.value_loss_sum = 0.0
        self.policy_loss_sum = 0.0
        self.count = 0

    def add(self, value_loss, policy_loss, batch_size):
        self.value_loss_sum += value_loss * batch_size
        self.policy_loss_sum += policy_loss * batch_size
        self.count += batch_size

    @property
    def mean_value_loss(self):
        return self.value_loss_sum / self.count

    @property
    def mean_policy_loss(self):
        return self.policy_loss_sum / self.count

    @property
    def mean_loss(self):
        return self.mean_value_loss + self.mean_policy_loss


class LossObserver():

    def notify(self, loss: Loss):
        """Act on given loss info"""


@dataclass
class ModelTrainerConfiguration:
    batch_size: int = 32
    max_training_epochs: int = 10
    early_stop_loss: float = 1e-3
    learning_rate: float = 1e-3
    loading_workers: int = 0


ItemCo = TypeVar("ItemCo", covariant = True)


class Buffer(Protocol[ItemCo]):

    @abstractmethod
    def __len__(self) -> int:
        """Size of the container"""

    @abstractmethod
    def __getitem__(self, index: int, /) -> ItemCo:
        """Returns an item"""


class ModelTrainer:

    def __init__(
        self,
        model: torch.nn.Module,
        config: ModelTrainerConfiguration,
        loss_observer: LossObserver | None = None
    ):
        self.model = model
        self.config = config
        self.loss_observer = loss_observer or LossObserver()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = config.learning_rate)
        self.policy_loss = torch.nn.CrossEntropyLoss()
        self.value_loss = torch.nn.MSELoss()

    def get_model(self):
        return self.model

    def train(self, replay_buffer: Buffer):
        dataset = TrainDataset(replay_buffer)
        self._train(dataset)

    def train_from_hdf5(self, hdf5_file_path):
        dataset = HDF5Dataset(hdf5_file_path)
        self._train(dataset)

    def _train(self, dataset: Dataset):
        self.model.train()
        data_loader = DataLoader(
            dataset,
            batch_size = self.config.batch_size,
            shuffle = True,
            num_workers = self.config.loading_workers
        )
        for _ in range(self.config.max_training_epochs):
            loss = self._train_once(data_loader)
            self.loss_observer.notify(loss)
            if loss.mean_loss < self.config.early_stop_loss:
                break
        self.model.eval()

    def _train_once(self, data_loader: DataLoader):
        loss = Loss()
        for (observation, probabilities, payoff) in data_loader:
            self.optimizer.zero_grad()
            predicted_payoff, predicted_logits = self.model(observation)
            value_loss = self.value_loss(predicted_payoff, payoff)
            policy_loss = self.policy_loss(predicted_logits, probabilities)
            total_loss = value_loss + policy_loss
            total_loss.backward()
            self.optimizer.step()
            loss.add(value_loss.item(), policy_loss.item(), len(observation))
        return loss
