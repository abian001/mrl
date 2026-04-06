from enum import Enum, IntEnum
from dataclasses import dataclass
from typing import Mapping
import math
import copy
import numpy as np
from mrl.alpha_zero.mcts_observation import MCTSObservation


class Player(Enum):

    X = 'X'
    O = 'O'

    def __str__(self):
        return self.value


class Action(IntEnum):

    TAKE = 0
    LEAVE = 1


@dataclass
class State:
    value: int
    active_player: Player
    winner: Player | None = None
    step: int = 0
    is_final: bool = False


class Perspective:

    def __init__(self, player: Player):
        self.player = player
        self.opponent = Player.O if player == Player.X else Player.X

    def get_observation(self, state: State) -> MCTSObservation:
        return MCTSObservation(state, self)

    def get_reward(self, state: State) -> float:
        if state.winner is self.opponent:
            return 0.0
        base_reward = 1 - math.exp(- math.log(state.value))
        if state.winner is self.player:
            return base_reward
        return base_reward / 2.0

    def get_core(self, state: State) -> np.ndarray:
        return np.fromiter((self.get_reward(state),), dtype = float)

    def get_action_space(self, state: State) -> tuple[Action, ...]:
        return (Action.TAKE, Action.LEAVE)

    @property
    def action_space_dimension(self):
        return 2


class Centipede:

    def __init__(self, number_of_steps: int):
        self.number_of_steps = number_of_steps

    def make_initial_state(self) -> State:
        return State(1, Player.O)

    def get_players(self) -> tuple[Player, ...]:
        return (Player.O, Player.X)

    def update(self, state: State, action: Action) -> State:
        state.step += 1
        if action == Action.TAKE:
            state.is_final = True
            state.winner = state.active_player
        else:
            state.value = 2 * state.value
            state.active_player = Player.O if state.active_player is Player.X else Player.X
            state.is_final = state.step == self.number_of_steps    
        return state

    def restore(self, observation: MCTSObservation) -> State:
        return copy.deepcopy(observation.state)

    def get_perspectives(self) -> Mapping[Player, Perspective]:
        return {
            p: Perspective(p) for p in self.get_players()
        }
