import os
from collections.abc import Generator
import pytest
import numpy as np
import torch
import h5py
from mrl.alpha_zero.model_trainer import (
    ModelTrainer,
    ModelTrainerConfiguration,
    LossObserver,
    Loss
)


class TestModel(torch.nn.Module):

    def __init__(self, input_shape: tuple[int], policy_size: int):
        super().__init__()
        self.policy_head = torch.nn.Linear(input_shape[0], policy_size)
        self.value_head = torch.nn.Linear(input_shape[0], 1)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return value.squeeze(-1), policy_logits


class TestObserver(LossObserver):

    def __init__(self) -> None:
        self.observed_loss: Loss | None = None

    def notify(self, loss: Loss):
        self.observed_loss = loss


@pytest.fixture
def test_observer() -> TestObserver:
    return TestObserver()


@pytest.fixture
def model_trainer(test_observer: TestObserver) -> ModelTrainer:
    model = TestModel(input_shape = (1,), policy_size = 1)
    configuration = ModelTrainerConfiguration(max_training_epochs = 1)
    return ModelTrainer(model, configuration, test_observer)


@pytest.mark.quick
def test_buffer_trainer(test_observer: TestObserver, model_trainer: ModelTrainer):
    buffer = (
        ((0.0,), (0.0,), 0.0),
        ((0.1,), (0.1,), 0.1)
    )
    model_trainer.train(buffer)
    assert test_observer.observed_loss is not None


@pytest.fixture
def hdf5_file() -> Generator[str, None, None]:
    hdf5_file_path = "model_trainer_test_hdf5.h5"
    observations = np.random.rand(32, 1).astype(np.int8)
    probabilities = np.random.rand(32, 1).astype(np.float32)
    payoffs = np.random.rand(32, 1).astype(np.float32)

    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        hdf5_file.create_dataset('observations', data = observations)
        hdf5_file.create_dataset('probabilities', data = probabilities)
        hdf5_file.create_dataset('payoffs', data = payoffs)

    yield hdf5_file_path
    os.remove(hdf5_file_path)


@pytest.mark.quick
def test_hdf5_trainer(test_observer: TestObserver, model_trainer: ModelTrainer, hdf5_file: str):
    model_trainer.train_from_hdf5(hdf5_file)
    assert test_observer.observed_loss is not None
