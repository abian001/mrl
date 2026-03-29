from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Protocol, Sequence
import numpy as np
from mrl.game.game import Observation


Probabilities = tuple[float, ...] | np.ndarray
LegalMask = tuple[bool, ...] | np.ndarray
Payoff = float
Action = int
DiscreteActionSpace = Sequence[Action]


class Oracle(ABC, Generic[Observation]):
    """An oracle is able to tell how good a given observation is and what is the optimal
       action strategy to take, expressed as probability distribution over a finite 
       discrete space of ordered actions.
    """

    @abstractmethod
    def get_value(self, observation: Observation) -> Payoff:
        """Return the expected payoff in the given observation"""

    @abstractmethod
    def get_probabilities(self, observation: Observation, legal_mask: LegalMask) -> Probabilities:
        """Returns the probabilities for each action.
           The legal mask defines which actions are allowed.
        """


class TrainableOracle(Oracle[Observation], ABC):

    @abstractmethod
    def save(self, file_path: str | Path) -> None:
        """Save model state to disk."""

    @abstractmethod
    def load(self, file_path: str | Path) -> None:
        """Load model state from disk."""


class HasLegalMask(Protocol):
    """Protocol for an observation, used to filter out disabled actions.
       The actions belong to a finite action space of ordered actions.
    """

    @property
    @abstractmethod
    def legal_mask(self) -> LegalMask:
        """Tells which actions are available at a given observed state"""


class OraclePolicy(ABC):
    """A policy that uses an oracle evaluation of the observed state to choose an action"""

    def __init__(self, oracle: Oracle):
        self.oracle = oracle

    def __call__(self, observation: HasLegalMask, action_space: DiscreteActionSpace) -> Action:
        probabilities = self.oracle.get_probabilities(observation, observation.legal_mask)
        return self._make_choice(probabilities)

    @abstractmethod
    def _make_choice(self, probabilities: Probabilities) -> Action:
        """Make a choice of actions"""


class DeterministicOraclePolicy(OraclePolicy):
    """A policy that always chooses the most likely next action as assessed by the oracle"""

    def _make_choice(self, probabilities: Probabilities) -> Action:
        return max(
            ((i, p) for (i, p) in enumerate(probabilities)),
            key = lambda x: x[1]
        )[0]


class StochasticOraclePolicy(OraclePolicy):
    """A policy that chooses an action according to the strategy suggested by the oracle"""

    def _make_choice(self, probabilities: Probabilities) -> Action:
        return np.random.choice(range(len(probabilities)), p = probabilities)
