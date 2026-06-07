import os
from pathlib import Path
from collections.abc import Generator
import yaml
import pytest
import numpy as np
import torch
import h5py
from mrl.alpha_zero.model_trainer import (
    YamlMetricsCollector,
    MetricsCollector,
    ModelTrainer,
    ModelTrainerConfiguration,
    TrainingMetrics,
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


class TestCollector(MetricsCollector):

    def __init__(self) -> None:
        self.observed_metrics: TrainingMetrics | None = None
        self.observed_last_batch: bool | None = None

    def collect(self, metrics: TrainingMetrics, is_last_batch_in_epoch: bool):
        self.observed_metrics = metrics
        self.observed_last_batch = is_last_batch_in_epoch


@pytest.fixture
def test_collector() -> TestCollector:
    return TestCollector()


@pytest.fixture
def model_trainer(test_collector: TestCollector) -> ModelTrainer:
    model = TestModel(input_shape = (1,), policy_size = 1)
    configuration = ModelTrainerConfiguration(max_training_epochs = 1)
    configuration.set_metrics_collector(test_collector)
    return ModelTrainer(model, configuration)


@pytest.mark.quick
def test_buffer_trainer(test_collector: TestCollector, model_trainer: ModelTrainer):
    buffer = (
        ((0.0,), (0.0,), 0.0),
        ((0.1,), (0.1,), 0.1)
    )
    model_trainer.train(buffer)
    assert test_collector.observed_metrics is not None
    assert test_collector.observed_last_batch is True


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
def test_hdf5_trainer(test_collector: TestCollector, model_trainer: ModelTrainer, hdf5_file: str):
    model_trainer.train_from_hdf5(hdf5_file)
    assert test_collector.observed_metrics is not None
    assert test_collector.observed_last_batch is True


@pytest.mark.quick
def test_loss_tracks_weighted_mean_over_multiple_batches():
    loss = TrainingMetrics(
        alpha_zero_epoch_index = 2,
        training_epoch_index = 0,
        batch_index = 3,
        value_loss = 8.0,
        policy_loss = 11.0,
        policy_entropy = 4.0,
    )

    assert loss.total_loss == pytest.approx(19.0)


@pytest.mark.quick
def test_yaml_metrics_collector_only_samples_last_batch(tmp_path: Path):
    collector = YamlMetricsCollector(
        file_path = Path("training_metrics.yaml"),
        workspace_path = tmp_path,
        sample_batch_period = None
    )

    number_of_records = 4
    for batch_index in range(number_of_records):
        metrics = TrainingMetrics(
            alpha_zero_epoch_index = 7,
            training_epoch_index = 3,
            batch_index = batch_index,
            value_loss = 8.0,
            policy_loss = 12.0,
            policy_entropy = 4.0,
        )
        is_last_batch = batch_index == (number_of_records - 1)
        collector.collect(metrics, is_last_batch)

    metrics_file = tmp_path / "training_metrics.yaml"
    with open(metrics_file, "r", encoding = "utf-8") as yaml_file:
        content = yaml.safe_load(yaml_file)

    assert len(content) == 1
    assert content[0]["alpha_zero_epoch"] == 7
    assert content[0]["training_epoch"] == 3
    assert content[0]["batch"] == 3
    assert content[0]["policy_loss"] == pytest.approx(12.0)
    assert content[0]["value_loss"] == pytest.approx(8.0)
    assert content[0]["total_loss"] == pytest.approx(20.0)
    assert content[0]["policy_entropy"] == pytest.approx(4.0)


@pytest.mark.quick
def test_yaml_metrics_collector_only_samples_according_to_period(tmp_path: Path):
    collector = YamlMetricsCollector(
        file_path = Path("training_metrics.yaml"),
        workspace_path = tmp_path,
        sample_batch_period = 2
    )

    number_of_records = 4
    for batch_index in range(number_of_records):
        metrics = TrainingMetrics(
            alpha_zero_epoch_index = 7,
            training_epoch_index = 3,
            batch_index = batch_index,
            value_loss = 8.0,
            policy_loss = 12.0,
            policy_entropy = 4.0,
        )
        is_last_batch = batch_index == (number_of_records - 1)
        collector.collect(metrics, is_last_batch)

    metrics_file = tmp_path / "training_metrics.yaml"
    with open(metrics_file, "r", encoding = "utf-8") as yaml_file:
        content = yaml.safe_load(yaml_file)

    assert len(content) == 3
    assert content[0]["alpha_zero_epoch"] == 7
    assert content[0]["training_epoch"] == 3
    assert content[0]["batch"] == 0
    assert content[0]["policy_loss"] == pytest.approx(12.0)
    assert content[0]["value_loss"] == pytest.approx(8.0)
    assert content[0]["total_loss"] == pytest.approx(20.0)
    assert content[0]["policy_entropy"] == pytest.approx(4.0)


@pytest.mark.quick
def test_train_increments_alpha_zero_epoch_on_each_training_run(
    test_collector: TestCollector,
    model_trainer: ModelTrainer,
):
    buffer = (
        ((0.0,), (0.0,), 0.0),
        ((0.1,), (0.1,), 0.1)
    )

    model_trainer.train(buffer)

    assert test_collector.observed_metrics is not None
    assert test_collector.observed_metrics.alpha_zero_epoch_index == 0
    assert test_collector.observed_metrics.training_epoch_index == 0
    assert test_collector.observed_metrics.batch_index == 0
    assert test_collector.observed_last_batch is True

    model_trainer.train(buffer)

    assert test_collector.observed_metrics is not None
    assert test_collector.observed_metrics.alpha_zero_epoch_index == 1
    assert test_collector.observed_metrics.training_epoch_index == 0
    assert test_collector.observed_metrics.batch_index == 0
    assert test_collector.observed_last_batch is True
