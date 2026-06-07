from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable
from abc import abstractmethod
from collections.abc import Generator
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from mrl.alpha_zero.experience_collector import get_hdf5_dataset
from mrl.configuration.factory import ObjectConfiguration


class TrainDataset(Dataset):

    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def __len__(self):
        return len(self.replay_buffer)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
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


@dataclass
class TrainingMetrics:
    alpha_zero_epoch_index: int
    training_epoch_index: int
    batch_index: int
    value_loss: float
    policy_loss: float
    policy_entropy: float

    @property
    def total_loss(self):
        return self.value_loss + self.policy_loss


@runtime_checkable
class MetricsCollector(Protocol):

    def collect(self, metrics: TrainingMetrics, is_last_batch_in_epoch: bool):
        """Act on given training metrics"""


class NullMetricsCollector(MetricsCollector):

    def collect(self, metrics: TrainingMetrics, is_last_batch_in_epoch: bool):
        del metrics
        del is_last_batch_in_epoch


class YamlMetricsCollector:

    def __init__(
        self,
        file_path: Path,
        workspace_path: Path | None = None,
        sample_batch_period: int | None = None
    ):
        if not file_path.is_absolute() and workspace_path is not None:
            file_path = workspace_path / file_path
        self.file_path = file_path
        self.sample_batch_period = sample_batch_period
        self.file_path.parent.mkdir(parents = True, exist_ok = True)

    def collect(self, metrics: TrainingMetrics, is_last_batch_in_epoch: bool) -> None:
        if not self._is_sample_batch(metrics.batch_index, is_last_batch_in_epoch):
            return
        with open(self.file_path, "a", encoding = "utf-8") as metrics_file:
            metrics_file.write(yaml.dump([{
                'alpha_zero_epoch': metrics.alpha_zero_epoch_index,
                'training_epoch': metrics.training_epoch_index,
                'batch': metrics.batch_index,
                'policy_loss': metrics.policy_loss,
                'value_loss': metrics.value_loss,
                'total_loss': metrics.total_loss,
                'policy_entropy': metrics.policy_entropy,
            }]) + "\n")

    def _is_sample_batch(self, batch_count: int, is_last_batch: bool):
        return is_last_batch or (
            self.sample_batch_period is not None and
            batch_count % self.sample_batch_period == 0
        )


class ModelTrainerConfiguration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed = True)
    batch_size: int = 32
    max_training_epochs: int = 10
    early_stop_loss: float = 1e-3
    learning_rate: float = 1e-3
    loading_workers: int = 0
    metrics_collector_configuration: ObjectConfiguration | None = Field(
        alias = 'metrics_collector',
        default = None,
    )
    _metrics_collector: MetricsCollector = PrivateAttr(default_factory = NullMetricsCollector)

    @property
    def metrics_collector(self) -> MetricsCollector:
        return self._metrics_collector

    def set_metrics_collector(self, metrics_collector: MetricsCollector) -> None:
        self._metrics_collector = metrics_collector


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
    ):
        self.model = model
        self.config = config
        self.metrics_collector = config.metrics_collector
        self.alpha_zero_epoch_index = 0
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
        alpha_zero_epoch_index = self.alpha_zero_epoch_index
        data_loader = DataLoader(
            dataset,
            batch_size = self.config.batch_size,
            shuffle = True,
            num_workers = self.config.loading_workers
        )
        for training_epoch_index in range(self.config.max_training_epochs):
            mean_loss = self._train_once(
                data_loader,
                alpha_zero_epoch_index,
                training_epoch_index,
            )
            if mean_loss < self.config.early_stop_loss:
                break
        self.alpha_zero_epoch_index += 1
        self.model.eval()

    def _train_once(  # pylint: disable=too-many-locals
        self,
        data_loader: DataLoader,
        alpha_zero_epoch_index: int,
        training_epoch_index: int,
    ):
        loss_sum = 0.0
        sample_count = 0
        for (
            batch_index, (observation, probabilities, payoff), is_last
        ) in self._enumerate_with_last(data_loader):
            self.optimizer.zero_grad()
            predicted_payoff, predicted_logits = self.model(observation)
            value_loss = self.value_loss(predicted_payoff, payoff)
            policy_loss = self.policy_loss(predicted_logits, probabilities)
            total_loss = value_loss + policy_loss
            total_loss.backward()
            self.optimizer.step()
            self._collect_metrics(
                alpha_zero_epoch_index,
                training_epoch_index,
                batch_index,
                predicted_logits,
                value_loss,
                policy_loss,
                is_last,
            )
            loss_sum += total_loss.item() * len(observation)
            sample_count += len(observation)
        return loss_sum / sample_count

    def _enumerate_with_last(
        self,
        data_loader: DataLoader
    ) -> Generator[tuple[int, tuple[torch.Tensor, ...], bool], None, None]:
        iterable_data_loder = iter(data_loader)
        try:
            previous_data = next(iterable_data_loder)
            previous_index = 0
        except StopIteration:
            return  # Empty Iterator

        for (index, data) in enumerate(iterable_data_loder):
            yield (previous_index, previous_data, False)
            previous_data = data
            previous_index = index + 1

        yield (previous_index, previous_data, True)

    def _collect_metrics(  # pylint: disable=too-many-positional-arguments
        self,
        alpha_zero_epoch_index: int,
        training_epoch_index: int,
        batch_index: int,
        predicted_logits: torch.Tensor,
        value_loss: torch.Tensor,
        policy_loss: torch.Tensor,
        is_last_batch_in_epoch: bool,
    ) -> None:
        log_probabilities = torch.log_softmax(predicted_logits, dim = -1)
        normalized_probabilities = torch.exp(log_probabilities)
        policy_entropy = -(
            normalized_probabilities * log_probabilities
        ).sum(dim = -1).mean()
        metrics = TrainingMetrics(
            alpha_zero_epoch_index = alpha_zero_epoch_index,
            training_epoch_index = training_epoch_index,
            batch_index = batch_index,
            value_loss = value_loss.item(),
            policy_loss = policy_loss.item(),
            policy_entropy = policy_entropy.item(),
        )
        self.metrics_collector.collect(metrics, is_last_batch_in_epoch)
