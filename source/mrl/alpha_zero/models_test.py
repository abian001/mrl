from dataclasses import dataclass
import pytest
import numpy as np
from mrl.alpha_zero.models import (
    OpenSpielMLP,
    OpenSpielConv,
    OpenSpielResnet,
    MLPCapacity,
    CNNCapacity
)
from mrl.alpha_zero.model_trainer import ModelTrainer, ModelTrainerConfiguration


@dataclass
class TestData:
    # pylint: disable=too-many-instance-attributes
    model_class: type
    model_size: MLPCapacity | CNNCapacity
    max_training_epochs: int
    train_data: tuple[tuple[
        tuple[float, ...] | np.ndarray,
        tuple[float, ...] | np.ndarray,
        float
    ], ...]
    example_observation: np.ndarray
    legal_mask: tuple[bool, ...]
    expected_full_probabilities: tuple[float, ...] | None
    expected_legal_probabilities: tuple[float, ...]


_example_state_1 = np.array((0.5, 0.1, 0.2, 0.3)).reshape((2, 1, 2))
_example_state_2 = np.array((0.1, 0.5, 0.6, 0.7)).reshape((2, 1, 2))
quick_test_data = (
    TestData(
        model_class = OpenSpielMLP,
        model_size = MLPCapacity(
            input_size = 1,
            output_size = 2,
            nn_width = 1,
            nn_depth = 1
        ),
        max_training_epochs = 1,
        train_data = (
            ((0.5,), (0.50, 0.50), 0.5),
            ((0.1,), (0.20, 0.80), 0.4)
        ),
        example_observation = np.array((0.5,)),
        legal_mask = (True, False),
        expected_full_probabilities = None,
        expected_legal_probabilities = (1.0, 0.0)
    ),
    TestData(
        model_class = OpenSpielMLP,
        model_size = MLPCapacity(
            input_size = 1,
            output_size = 2,
            nn_width = 1,
            nn_depth = 1
        ),
        max_training_epochs = 1,
        train_data = (
            (np.array((0.5,)), np.array((0.50, 0.50)), 0.5),
            (np.array((0.1,)), np.array((0.20, 0.80)), 0.4)
        ),
        example_observation = np.array((0.5,)),
        legal_mask = (True, False),
        expected_full_probabilities = None,
        expected_legal_probabilities = (1.0, 0.0)
    ),
    TestData(
        model_class = OpenSpielConv,
        model_size = CNNCapacity(
            input_shape = (2, 1, 2),
            output_size = 2,
            nn_width = 1,
            nn_depth = 1
        ),
        max_training_epochs = 1,
        train_data = (
            (_example_state_1, np.array((0.50, 0.50)), 0.5),
            (_example_state_2, np.array((0.20, 0.80)), 0.4)
        ),
        example_observation = _example_state_1,
        legal_mask = (True, False),
        expected_full_probabilities = None,
        expected_legal_probabilities = (1.0, 0.0)
    ),
    TestData(
        model_class = OpenSpielResnet,
        model_size = CNNCapacity(
            input_shape = (2, 1, 2),
            output_size = 2,
            nn_width = 1,
            nn_depth = 1
        ),
        max_training_epochs = 1,
        train_data = (
            (_example_state_1, np.array((0.50, 0.50)), 0.5),
            (_example_state_2, np.array((0.20, 0.80)), 0.4)
        ),
        example_observation = _example_state_1,
        legal_mask = (True, False),
        expected_full_probabilities = None,
        expected_legal_probabilities = (1.0, 0.0)
    )
)

@pytest.mark.parametrize('test_data', quick_test_data)
@pytest.mark.quick
def test_quick_training(test_data: TestData):
    _run_test(test_data)


_example_state_1 = np.array((0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5))
slow_test_data = (
    TestData(
        model_class = OpenSpielMLP,
        model_size = MLPCapacity(
            input_size = 9,
            output_size = 9,
            nn_width = 9,
            nn_depth = 3
        ),
        max_training_epochs = 5000,
        train_data = (
            (
                _example_state_1,
                np.array((0.25, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.25)),
                0.5
            ),
            (
                np.array((0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0)),
                np.array((0.25, 0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0)),
                0.5
            )
        ),
        example_observation = _example_state_1,
        legal_mask = (True, True, False, True, False, True, True, True, True),
        expected_full_probabilities = (0.25, 0.0, 0.25, 0.0, 0.0, 0.0, 0.25, 0.0, 0.25),
        expected_legal_probabilities = (0.3333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3333, 0.0, 0.3333)
    ),
)

@pytest.mark.parametrize('test_data', slow_test_data)
@pytest.mark.slow
def test_slow_training(test_data: TestData):
    _run_test(test_data)


def _run_test(test_data: TestData):
    model = test_data.model_class(test_data.model_size)
    configuration = ModelTrainerConfiguration(

        max_training_epochs = test_data.max_training_epochs
    )
    model_trainer = ModelTrainer(model, configuration)
    model_trainer.train(test_data.train_data)
    observation = ObservationMock(test_data.example_observation)
    value = model.get_value(observation)
    full_probabilities = model.get_probabilities(observation)
    legal_probabilities = model.get_probabilities(observation, test_data.legal_mask)
    assert isinstance(value, float)
    if test_data.expected_full_probabilities is None:
        assert sum(full_probabilities) == pytest.approx(1.0, abs = 0.01)
    else:
        expected = pytest.approx(test_data.expected_full_probabilities, abs = 0.01)
        assert full_probabilities == expected
    assert legal_probabilities == pytest.approx(test_data.expected_legal_probabilities, abs = 0.01)


class ObservationMock:

    def __init__(self, core: np.ndarray):
        self._core = core

    @property
    def core(self) -> np.ndarray:
        return self._core
