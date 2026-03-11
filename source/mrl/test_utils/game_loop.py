import itertools
import time
from typing import Iterator, Tuple, Sequence, Dict, Generic
import pytest
from mrl.game.game import (
    Game,
    Observation,
    Action,
    Policy,
    Observer,
    GlobalObserver,
    StopEvent,
    Player,
    State,
    FinalCheckable,
    ActionSpace
)
from mrl.game.game_loop import make_game_loop


class TestPolicy(Generic[Observation, Action, ActionSpace]):

    def __init__(self, actions: Iterator[Action]):
        super().__init__()
        self.actions = actions

    def __call__(self, observation: Observation, action_space: ActionSpace) -> Action:
        assert observation is not None
        try:
            next_action = next(self.actions)
        except StopIteration as error:
            raise StopEvent("Stop Event") from error
        assert next_action in action_space
        return next_action


class History(Observer, Generic[Observation, Action]):

    def __init__(self):
        self.records = []

    def notify(self, observation: Observation, action: Action) -> None:
        self.records.append(str(observation))
        self.records.append(str(action))

    def notify_final(self, observation: Observation) -> None:
        self.records.append(str(observation))
        self.records.append("End of episode")

    def notify_stop_event(self, stop_event: StopEvent) -> None:
        self.records.append(str(stop_event))

    def as_list(self):
        return self.records


class GlobalHistory(GlobalObserver, Generic[Player, State, Action]):

    def __init__(self):
        self.records = []

    def notify_state(self, state: State) -> None:
        assert isinstance(state, FinalCheckable)
        self.records.append(str(state))
        if state.is_final:
            self.records.append("End of episode")

    def notify_actions(self, actions: Tuple[Action, ...]) -> None:
        self.records.append(tuple(str(x) for x in actions))

    def notify_action(self, player: Player, action: Action) -> None:
        self.records.append((str(player), str(action)))

    def notify_stop_event(self, stop_event: StopEvent) -> None:
        self.records.append(str(stop_event))

    def as_list(self):
        return self.records


class GameRunner(Generic[Player]):

    def __init__(self) -> None:
        self.histories: Dict[Player, History] = {}
        self.global_history: GlobalHistory = GlobalHistory()

    def run_test(
        self,
        game: Game,
        actions: Dict[Player, Tuple[Action, ...]],
        number_of_times: int = 1
    ) -> None:
        self.histories = {
            player: History()
            for player in game.get_players()
        }
        policies: Dict[Player, Policy] = {
            player: TestPolicy(iter(player_actions))
            for (player, player_actions) in actions.items()
        }
        game_loop = make_game_loop(game, policies, self.histories, self.global_history)
        game_loop.run(number_of_times)

    def run_performance_test(
       self,
       game: Game,
       actions: Dict[Player, Tuple[Action, ...]],
       number_of_times: int = 1000
    ):
        policies: Dict[Player, Policy] = {
            player: TestPolicy(itertools.cycle(player_actions))
            for (player, player_actions) in actions.items()
        }
        game_loop = make_game_loop(game, policies)
        start_time = time.perf_counter()
        game_loop.run(number_of_times)
        end_time = time.perf_counter()
        return end_time - start_time

    def assert_histories(
        self,
        expected_histories: Dict[Player, Sequence[str | Tuple[str, ...]]],
        expected_global_history: Sequence[str | Tuple[str, ...]]
    ):
        assert len(self.histories) == len(expected_histories)
        for (player, expected_history) in expected_histories.items():
            assert player in self.histories
            self.assert_history_is(self.histories[player], expected_history, f"Player {player}'s")
        self.assert_history_is(self.global_history, expected_global_history, "Global")

    def assert_history_is(
        self,
        history: History | GlobalHistory,
        expected_history: Sequence[str | Tuple[str, ...]],
        context: str
    ) -> None:
        history_list = history.as_list()
        for (index, (record, expected_record)) in enumerate(zip(history_list, expected_history)):
            assert \
                record == expected_record, (
                    f"{context} history is unexpected at record N. {index}\n"
                    f"The expected record was {expected_record}\n"
                    f"The observed record was {record}"
                )
        assert len(history_list) == len(expected_history)


@pytest.fixture
def game_runner() -> GameRunner:
    return GameRunner()
