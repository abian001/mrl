from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Generic, Protocol, TypeVar, cast
import copy
import math
import random as python_random

from mrl.game.game import (
    Action,
    ActionContra,
    ActionSpace,
    FinalCheckable,
    HasActivePlayer,
    Observation,
    Player,
    PlayerCo,
    Restorable,
    RewardObservable,
    TurnBased,
    DiscreteActionSpace
)


class RandomPolicy(Generic[Observation, Action, DiscreteActionSpace]):

    def __call__(
        self,
        observation: Observation,
        action_space: DiscreteActionSpace,
    ) -> Action:
        del observation
        return python_random.choice(action_space)


class FirstChoicePolicy(Generic[Observation, Action, DiscreteActionSpace]):

    def __call__(
        self,
        observation: Observation,
        action_space: DiscreteActionSpace,
    ) -> Action:
        del observation
        return action_space[0]


class ManualPolicy(Generic[Observation, Action, ActionSpace]):
    """Marker class that tells the runner to use manual inputs as a policy"""

    def __call__(self, observation: Observation, action_space: ActionSpace) -> Action:
        del observation, action_space
        raise TypeError("An attempt to call the marker manual policy was made.")


class SearchState(FinalCheckable, HasActivePlayer[PlayerCo], Protocol[PlayerCo]):
    """State constraints needed by AlphaBetaPolicy."""


State = TypeVar("State", bound = SearchState[Any])


class SearchGame(
    Restorable[State, Observation],
    TurnBased[State, ActionContra],
    RewardObservable[State, Player, Observation],
    Protocol[State, Observation, ActionContra, Player],
):
    """Game constraints needed by AlphaBetaPolicy."""


@dataclass(frozen = True)
class SearchWindow:
    alpha: float = -math.inf
    beta: float = math.inf


class AlphaBetaPolicy(Generic[State, Observation, Action, Player, DiscreteActionSpace]):
    """Alpha-beta search for deterministic two-player games.

    The policy evaluates the game tree from one player's perspective and assumes
    every other turn minimizes that player's payoff. It still runs on multiplayer
    games, but that approximation does not guarantee optimal play there.
    """

    def __init__(
        self,
        game: SearchGame[State, Observation, Action, Player],
        max_depth: int = 10,
        cache_size: int = 0,
    ) -> None:
        self.game = game
        self.max_depth = max_depth
        self._get_best_action_with_cache = lru_cache(maxsize = cache_size)(
            self._get_best_action
        )

    def __call__(
        self,
        observation: Observation,
        action_space: DiscreteActionSpace,
    ) -> Action:
        del action_space
        state = self.game.restore(observation)
        if self._get_best_action_with_cache.cache_info().maxsize:
            if _is_hashable(state):
                return self._get_best_action_with_cache(cast(Any, state))
            return self._get_best_action(state)
        return self._get_best_action(state)

    def _get_best_action(
        self,
        state: State,
    ) -> Action:
        active_perspective = self.game.get_perspectives()[state.active_player]
        action_space = active_perspective.get_action_space(state)
        return max(
            action_space,
            key = lambda action: self._get_payoff_value(
                state,
                action,
                state.active_player,
                depth = 1,
            )
        )

    def _get_payoff_value(  # pylint: disable=too-many-positional-arguments
        self,
        state: State,
        action: Action,
        player: Player,
        window: SearchWindow = SearchWindow(),
        depth: int = 0,
    ) -> float:
        state_copy = copy.deepcopy(state)
        state_copy = self.game.update(state_copy, action)

        if state_copy.is_final or depth >= self.max_depth:
            return self.game.get_perspectives()[player].get_reward(state_copy)

        if player == state_copy.active_player:
            return self._get_max_payoff(state_copy, player, window, depth)
        return self._get_min_payoff(state_copy, player, window, depth)

    def _get_max_payoff(
        self,
        state: State,
        player: Player,
        window: SearchWindow,
        depth: int,
    ) -> float:
        best_payoff = -math.inf
        active_perspective = self.game.get_perspectives()[state.active_player]
        action_space = active_perspective.get_action_space(state)
        for action in action_space:
            payoff = self._get_payoff_value(state, action, player, window, depth + 1)
            best_payoff = max(best_payoff, payoff)
            window = SearchWindow(
                alpha = max(window.alpha, best_payoff),
                beta = window.beta,
            )
            if window.alpha >= window.beta:
                break
        return best_payoff

    def _get_min_payoff(
        self,
        state: State,
        player: Player,
        window: SearchWindow,
        depth: int,
    ) -> float:
        worse_payoff = math.inf
        active_perspective = self.game.get_perspectives()[state.active_player]
        action_space = active_perspective.get_action_space(state)
        for action in action_space:
            payoff = self._get_payoff_value(state, action, player, window, depth + 1)
            worse_payoff = min(worse_payoff, payoff)
            window = SearchWindow(
                alpha = window.alpha,
                beta = min(window.beta, worse_payoff),
            )
            if window.alpha >= window.beta:
                break
        return worse_payoff


def _is_hashable(*values: Any) -> bool:
    try:
        hash(values)
        return True
    except TypeError:
        return False
