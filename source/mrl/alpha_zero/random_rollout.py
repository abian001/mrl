from typing import Protocol
import numpy as np
from mrl.game.game import (
    State,
    Player,
    TurnBased,
    Restorable,
    ActionContra,
    PayoffObservable
)
from mrl.alpha_zero.oracle import Oracle, LegalMask, Probabilities
from mrl.alpha_zero.mcts_observation import MCTSObservation


class Rolloutable(
    TurnBased[State, ActionContra],
    Restorable[State, MCTSObservation],
    PayoffObservable[State, Player, MCTSObservation],
    Protocol[State, Player, ActionContra]
):
    pass


class RandomRollout(Oracle):

    def __init__(self, game: Rolloutable, number_of_rollouts: int = 1):
        self.game = game
        self.number_of_rollouts = number_of_rollouts
        self.perspectives = self.game.get_perspectives()

    def get_value(self, observation: MCTSObservation) -> float:
        return sum(
            self._get_rollout_value(observation)
            for _ in range(self.number_of_rollouts)
        ) / self.number_of_rollouts

    def _get_rollout_value(self, observation: MCTSObservation) -> float:
        state = self.game.restore(observation)
        initial_player = state.active_player
        while not state.is_final:
            perspective = self.perspectives[state.active_player]
            action_space = perspective.get_action_space(state)
            action = np.random.choice(action_space)
            state = self.game.update(state, action)
        return self.perspectives[initial_player].get_payoff(state)

    def get_probabilities(
        self,
        observation: MCTSObservation,
        legal_mask: LegalMask
    ) -> Probabilities:
        return np.array(legal_mask, dtype = float)
