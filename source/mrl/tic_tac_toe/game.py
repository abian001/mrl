from enum import Enum
from abc import ABC, abstractmethod
from typing import Generic, Mapping
from dataclasses import dataclass
from mrl.configuration.player_utils import ChoiceFunction
from mrl.game.game import InteractivePolicy, StopEvent, Observation


Action = int


class Symbol(Enum):

    EMPTY = '0'
    O = 'O'
    X = 'X'

    def __str__(self):
        return self.name


Player = Symbol


@dataclass
class State:

    board: list[Symbol]
    active_player: Player
    winner: Player | None = None
    is_final: bool = False

    def __repr__(self):
        symbols = (
            *self.board,
            None,
            Symbol.EMPTY if self.is_final else self.active_player,
            Symbol.EMPTY if self.winner is None else self.winner
        )
        return "".join(_to_string(x) for x in symbols)


class BasePerspective(ABC, Generic[Observation]):

    def __init__(self, player: Player):
        self.player = player

    @abstractmethod
    def get_observation(self, state: State) -> Observation:
        """Returns the observation made by the player in the game"""

    def get_action_space(self, state: State) -> tuple[Action, ...]:
        return tuple(i for i in range(9) if state.board[i] == Symbol.EMPTY)

    def get_payoff(self, state: State) -> float:
        if state.winner == self.player:
            return 1.0
        if state.is_final and state.winner is None:
            return 0.5
        return 0.0


class Perspective(BasePerspective[State]):

    def get_observation(self, state: State) -> State:
        return state


class BaseTicTacToe(ABC, Generic[Observation]):

    winning_lines = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6]
    ]

    def __init__(
        self,
        first_player: ChoiceFunction | None = None
    ) -> None:
        self.first_player = first_player or ChoiceFunction(choices = 'O')

    def get_players(self) -> tuple[Player, ...]:
        return (Symbol.O, Symbol.X)

    @abstractmethod
    def get_perspectives(self) -> Mapping[Player, BasePerspective[Observation]]:
        """Returns the players' perspectives"""

    def make_initial_state(self) -> State:
        first_player = self.first_player(self)
        return State([Symbol.EMPTY] * 9, first_player)

    def update(self, state: State, action: Action) -> State:
        state.board[action] = state.active_player
        state.active_player = Symbol.O if state.active_player is Symbol.X else Symbol.X
        state.winner = self._compute_winner(state)
        state.is_final = (
            state.winner is not None or
            all(s != Symbol.EMPTY for s in state.board)
        )
        return state

    def _compute_winner(self, state: State) -> Player | None:
        for line in self.winning_lines:
            symbol = state.board[line[0]]
            if symbol != Symbol.EMPTY and all(state.board[i] == symbol for i in line[1:]):
                return symbol
        return None


class TicTacToe(BaseTicTacToe[State]):

    def get_perspectives(self) -> Mapping[Player, Perspective]:
        return {
            player: Perspective(player)
            for player in self.get_players()
        }


def _to_string(symbol: Symbol | None):
    if symbol == Symbol.EMPTY:
        return "."
    if symbol == Symbol.O:
        return "O"
    if symbol == Symbol.X:
        return "X"
    return " "


class TicTacToeStdin(InteractivePolicy):

    def __init__(self):
        self.last_state = None

    def __call__(self, state, action_space):
        assert state == self.last_state
        print("Choose - Choose an invalid action to stop")
        for x in action_space:
            print((x % 3, x // 3), x)

        try:
            action = int(input())
        except ValueError as error:
            raise StopEvent("Invalid Action") from error

        if action not in action_space:
            raise StopEvent("Invalid Action")
        return action

    def notify_observation(self, observation) -> None:
        self.last_state = observation
        print(f"\n--Active Player: {_to_string(observation.active_player)}--")
        print(f" {''.join(_to_string(x) for x in observation.board[0:3])}")
        print(f" {''.join(_to_string(x) for x in observation.board[3:6])}")
        print(f" {''.join(_to_string(x) for x in observation.board[6:9])}")
        print("------")
        if observation.is_final:
            print("The game is over")
            print(f"The winner is {observation.winner}")

    def notify_action(self, player: Player, action: Action) -> None:
        print(f"Player {player} used action {action}")

    def notify_stop_event(self, stop_event: StopEvent) -> None:
        print("Stop event observed")
