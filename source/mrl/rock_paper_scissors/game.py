from enum import Enum
from dataclasses import dataclass
from mrl.game.game import InteractivePolicy, StopEvent


class Player(Enum):

    O = 'O'
    X = 'X'

    def __str__(self):
        return self.value


class Action(Enum):

    ROCK = 'rock'
    PAPER = 'paper'
    SCISSORS = 'scissors'

    def __str__(self) -> str:
        return self.value


@dataclass
class State:

    wins: dict[Player, int]
    step: int = 0
    is_final: bool = False

    @property
    def winner(self) -> Player | None:
        return Player.O if self.wins[Player.O] > self.wins[Player.X] else (
            Player.X if self.wins[Player.X] > self.wins[Player.O] else None
        )

    def __repr__(self) -> str:
        return f"S{self.step}: {self.wins[Player.O]} vs {self.wins[Player.X]}"


class Perspective:

    def __init__(self, player: Player):
        self.player = player

    def get_observation(self, state: State) -> State:
        return state

    def get_action_space(self, state: State) -> tuple[Action, ...]:
        _ = state
        return (Action.ROCK, Action.PAPER, Action.SCISSORS)

    def get_payoff(self, state: State) -> float:
        if state.step == 0:
            return 0.0
        return float(state.wins[self.player]) / float(state.step)


class RockPaperScissors:

    def __init__(
        self,
        number_of_rounds: int = 5
    ):
        self.number_of_rounds = number_of_rounds

    def get_players(self) -> tuple[Player, ...]:
        return (Player.O, Player.X)

    def get_perspectives(self) -> dict[Player, Perspective]:
        return {
            player: Perspective(player)
            for player in self.get_players()
        }

    def make_initial_state(self) -> State:
        return State({p: 0 for p in self.get_players()})

    def update(self, state: State, actions: tuple[Action | None, ...]) -> State:
        for player in (Player.O, Player.X):
            index, other = (0, 1) if player == Player.O else (1, 0)
            if (
                (actions[index] == Action.ROCK and actions[other] == Action.SCISSORS) or
                (actions[index] == Action.SCISSORS and actions[other] == Action.PAPER) or
                (actions[index] == Action.PAPER and actions[other] == Action.ROCK)
            ):
                state.wins[player] += 1
        state.step += 1
        state.is_final = state.step >= self.number_of_rounds
        return state


class RockPaperScissorsStdin(InteractivePolicy):
    _action_map = dict(enumerate(list(Action)))

    def __init__(self, player: Player):
        self.player = player
        self.last_state: State | None = None

    def __call__(self, state: State, action_space: tuple[Action, ...]) -> Action:
        assert state == self.last_state
        print("Choose - Choose an invalid action to stop")
        for (x, action) in self._action_map.items():
            print(action, x)

        try:
            action = self._action_map[int(input())]
        except (ValueError, KeyError) as error:
            raise StopEvent("Invalid Action") from error

        if action not in action_space:
            raise StopEvent("Invalid Action")
        return action

    def notify_observation(self, observation: State) -> None:
        self.last_state = observation
        print(f"\n-- You are Player {self.player}--")
        print(f"Current results {observation}")
        print("------")
        if observation.is_final:
            print("The game is over")
            print(f"The winner is Player {observation.winner}")

    def notify_actions(self, actions: tuple[Action, ...]) -> None:
        _actions = tuple(str(x) for x in actions)
        print(f"Actions played: {_actions}")

    def notify_stop_event(self, stop_event: StopEvent) -> None:
        print("Stop event observed")
