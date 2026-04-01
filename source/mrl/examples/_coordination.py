from enum import Enum
from typing import Mapping
from dataclasses import dataclass
from mrl.game.game import InteractivePolicy, StopEvent


class Player(Enum):
    O = 'O'
    X = 'X'

    def __str__(self):
        return str(self.value)


class Action(Enum):
    A = 'a'
    B = 'b'

    def __str__(self):
        return str(self.value)


History = list[tuple[Action, ...]]


@dataclass
class State:
    history: History
    is_final: bool


@dataclass
class Observation:
    state: State


class Perspective:

    def get_observation(self, state: State) -> Observation:
        return Observation(state)

    def get_reward(self, state: State) -> float:
        if len(state.history) == 0:
            return 0.0
        return len([x for x in state.history if x[0] == x[1]]) / len(state.history)

    def get_action_space(self, state: State) -> tuple[Action, ...]:
        return (Action.A, Action.B)


class Coordination:

    def __init__(self, number_of_tests: int):
        self.number_of_tests = number_of_tests

    def make_initial_state(self) -> State:
        return State([], self.number_of_tests <= 0)

    def get_players(self) -> tuple[Player, ...]:
        return (Player.O, Player.X)

    def get_perspectives(self) -> Mapping[Player, Perspective]:
        unique_perspective = Perspective()
        return {
            player: unique_perspective
            for player in self.get_players()
        }

    def update(self, state: State, actions: tuple[Action, ...]) -> State:
        state.is_final = len(state.history) == self.number_of_tests - 1
        state.history += [actions]
        return state


class CoordinationStdin(InteractivePolicy):

    def __init__(self, player: Player):
        self.player = player

    def __call__(self, observation: Observation, action_space: tuple[Action, ...]) -> Action:
        print("Choose an action: a or b?")
        action = input()
        validated_action = next((
            a for a in action_space
            if str(a) == action.lower()
        ), None)
        if validated_action is None: 
            raise StopEvent(f"Invalid Action {action}.")
        return validated_action

    def notify_observation(self, observation: Observation) -> None:
        print(f"\nYou are player {self.player}.")
        print(f"Number of played steps: {len(observation.state.history)}")
        coordinations = len([x for x in observation.state.history if x[0] == x[1]])
        print(f"Number of successful coordinations: {coordinations}")

    def notify_actions(self, actions: tuple[Action, ...]) -> None:
        print(f"The following actions were played: O: {actions[0]}, X: {actions[1]}")

    def notify_stop_event(self, stop_event: StopEvent) -> None:
        print(f"A stop event was observed: {stop_event}")
