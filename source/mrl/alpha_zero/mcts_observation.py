from typing import Any, Generic, TypeVar, Protocol, runtime_checkable
from abc import abstractmethod
import numpy as np
from mrl.game.game import (
    Action,
    FinalCheckable,
    StateContra,
    ActionCo,
    PayoffPerspective,
    HasActionSpaceDimension
)


@runtime_checkable
class MCTSPerspective(
    PayoffPerspective,
    HasActionSpaceDimension,
    Protocol[StateContra, ActionCo]
):  # pylint: disable=duplicate-bases

    @abstractmethod
    def get_core(self, state: StateContra) -> np.ndarray:
        """Return the essential part of the observation. This is what the oracle
           will use to make a prediction.
        """

    @abstractmethod
    def get_action_space(self, state: StateContra) -> tuple[ActionCo, ...]:
        """Returns the observed action space"""


FinalCheckableState = TypeVar('FinalCheckableState', bound = FinalCheckable)


class MCTSObservation(Generic[FinalCheckableState, Action]):

    def __init__(
        self,
        state: FinalCheckableState,
        perspective: MCTSPerspective[FinalCheckableState, Action]
    ):
        self.state = state
        self.perspective = perspective
        self._payoff: float | None = None
        self._core: np.ndarray | None = None
        self._legal_mask: np.ndarray | None = None
        self._action_space: tuple[Action, ...] | None = None

    @property
    def payoff(self) -> float:
        if self._payoff is None:
            self._payoff = self.perspective.get_payoff(self.state)
        return self._payoff

    @property
    def core(self) -> np.ndarray:
        if self._core is None:
            self._core = self.perspective.get_core(self.state)
        return self._core

    @property
    def legal_mask(self) -> np.ndarray:
        try:
            action_space_dimension = self.perspective.action_space_dimension
        except AttributeError as original_error:
            # Without this override, the __getattr__ handler catches the
            # AttributeError exception and incorrectly replaces it with
            # AttrubuteError: MCTSObservation has no attribute legal_mask
            raise ValueError(
                f"Class {type(self.perspective)} has no attribute action_space_dimension."
            ) from original_error

        if self._legal_mask is None:
            self._legal_mask = np.fromiter((
                i in self.action_space
                for i in range(action_space_dimension)
            ), dtype = bool)
        return self._legal_mask

    @property
    def action_space(self) -> tuple[Action, ...]:
        if self._action_space is None:
            self._action_space = self.perspective.get_action_space(self.state)
        return self._action_space

    def __eq__(self, other) -> bool:
        return self.state == other.state

    @property
    def is_final(self) -> bool:
        return self.state.is_final

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.state, name):
            value = getattr(self.state, name)
            if not callable(value):
                return value
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'.")
