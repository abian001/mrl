from enum import Enum
from abc import ABC, abstractmethod
from typing import Generic, Mapping
from dataclasses import dataclass
from mrl.configuration.player_utils import ChoiceFunction
from mrl.game.game import InteractivePolicy, StopEvent, Observation


Action = int


class Token(Enum):

    NONE = '.'
    O = 'O'
    X = 'X'

    def __str__(self):
        return self.value


Player = Token


@dataclass
class State:
    board: list[Token]
    active_player: Player
    winner: Player | None = None
    is_final: bool = False

    def __repr__(self):
        symbols = (
            *self.board,
            None,
            Token.NONE if self.is_final else self.active_player,
            Token.NONE if self.winner is None else self.winner
        )
        return "".join(str(x) if x else ' ' for x in symbols)


class BasePerspective(ABC, Generic[Observation]):

    def __init__(self, player: Player):
        self.player = player

    @abstractmethod
    def get_observation(self, state: State) -> Observation:
        """Returns the observation made by the player in the game"""

    def get_action_space(self, state: State) -> tuple[Action, ...]:
        return tuple(i for i in range(7) if state.board[i] == Token.NONE)

    def get_reward(self, state: State) -> float:
        if state.winner == self.player:
            return 1.0
        if state.is_final and state.winner is None:
            return 0.5
        return 0.0


class Perspective(BasePerspective[State]):

    def get_observation(self, state: State) -> State:
        return state


class BaseStraightFour(ABC, Generic[Observation]):

    def __init__(
        self,
        first_player: ChoiceFunction | None = None
    ) -> None:
        self.first_player = first_player or ChoiceFunction(choices = ('O',))

    def get_players(self) -> tuple[Player, ...]:
        return (Token.O, Token.X)

    @abstractmethod
    def get_perspectives(self) -> Mapping[Player, BasePerspective[Observation]]:
        """Returns the players' perspectives"""

    def make_initial_state(self) -> State:
        first_player = self.first_player(self)
        return State([Token.NONE] * 49, first_player)

    def update(self, state: State, action: Action) -> State:
        row = next(row for row in reversed(range(7)) if state.board[7*row + action] == Token.NONE)
        state.board[7*row + action] = state.active_player
        if self._active_player_won(state, row, action):
            state.winner = state.active_player
        state.active_player = Token.O if state.active_player is Token.X else Token.X
        state.is_final = (
            state.winner is not None or
            all(s != Token.NONE for s in state.board)
        )
        if state.is_final:
            state.active_player = Token.NONE
        return state

    def _active_player_won(self, state: State, row: int, column: int) -> bool:
        lines = (
            ((i, column) for i in range(7)),
            ((row, i) for i in range(7)),
            ((row - column + i, i) for i in range(7) if 0 <= row - column + i <= 6),
            ((row + column - i, i) for i in range(7) if 0 <= row + column - i <= 6)
        )
        return any(self._any_consecutive_four(state, line) for line in lines)

    def _any_consecutive_four(self, state, positions):
        count = 0
        for (row, column) in positions:
            if state.board[row * 7 + column] == state.active_player:
                count += 1
            else:
                count = 0
            if count >= 4:
                return True
        return False


class StraightFour(BaseStraightFour[State]):

    def get_perspectives(self) -> Mapping[Player, Perspective]:
        return {
            player: Perspective(player)
            for player in self.get_players()
        }


class StraightFourStdin(InteractivePolicy):

    def __init__(self):
        self.last_observation = None

    def __call__(self, state, action_space):
        assert state == self.last_observation
        print("Choose - Choose an invalid action to stop")

        try:
            action = int(input())
        except ValueError as error:
            raise StopEvent("Invalid Action") from error

        if action not in action_space:
            raise StopEvent("Invalid Action")
        return action

    def notify_observation(self, observation) -> None:
        self.last_observation = observation
        print(f"\n--Active Player: {str(observation.active_player)}--")
        for row in range(7):
            print(f" {''.join(str(x) for x in observation.board[row * 7:(row + 1)*7])}")
        print("------")
        if observation.is_final:
            print("The game is over")
            print(f"The winner is {observation.winner}")

    def notify_action(self, player: Player, action: Action) -> None:
        print(f"Player {player} used action {action}")

    def notify_stop_event(self, stop_event: StopEvent) -> None:
        print("Stop event observed")
