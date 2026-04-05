from dataclasses import dataclass
from typing import Sequence

import pytest

from mrl.test_utils.policies import AlphaBetaPolicy


class Player:
    X = "X"
    O = "O"


@dataclass(frozen = True)
class State:
    active_player: str
    name: str
    is_final: bool = False


class Perspective:

    def __init__(self, game: "Game", player: str) -> None:
        self.game = game
        self.player = player

    def get_observation(self, state: State) -> State:
        return state

    def get_action_space(self, state: State) -> tuple[str, ...]:
        if state.is_final:
            return ()
        if state.active_player == Player.X:
            return ("choice_1", "choice_2")
        return ("left", "right")

    def get_reward(self, state: State) -> float:
        return self.game.payoffs[self.player][state]


class Game:  # pylint: disable=too-many-instance-attributes

    def __init__(self) -> None:
        self.update_calls = 0
        self.choice_1_state = State(active_player = Player.O, name = "choice_1")
        self.choice_2_state = State(active_player = Player.O, name = "choice_2")
        self.choice_1_left = State(
            active_player = Player.X,
            name = "choice_1_left",
            is_final = True,
        )
        self.choice_1_right = State(
            active_player = Player.X,
            name = "choice_1_right",
            is_final = True,
        )
        self.choice_2_left = State(
            active_player = Player.X,
            name = "choice_2_left",
            is_final = True,
        )
        self.choice_2_right = State(
            active_player = Player.X,
            name = "choice_2_right",
            is_final = True,
        )
        self.payoffs = {
            Player.X: {
                self.choice_1_state: 0.7,
                self.choice_2_state: 1.0,
                self.choice_1_left: 0.8,
                self.choice_1_right: 0.7,
                self.choice_2_left: 1.0,
                self.choice_2_right: 0.0,
            },
            Player.O: {
                self.choice_1_state: -0.7,
                self.choice_2_state: -1.0,
                self.choice_1_left: -0.8,
                self.choice_1_right: -0.7,
                self.choice_2_left: -1.0,
                self.choice_2_right: 0.0,
            },
        }
        self.perspectives = {
            Player.X: Perspective(self, Player.X),
            Player.O: Perspective(self, Player.O),
        }

    def restore(self, observation: State) -> State:
        return observation

    def update(self, state: State, action: str) -> State:
        self.update_calls += 1
        if state.active_player == Player.X:
            return {
                "choice_1": self.choice_1_state,
                "choice_2": self.choice_2_state,
            }[action]
        return {
            ("choice_1", "left"): self.choice_1_left,
            ("choice_1", "right"): self.choice_1_right,
            ("choice_2", "left"): self.choice_2_left,
            ("choice_2", "right"): self.choice_2_right,
        }[(state.name, action)]

    def get_perspectives(self) -> dict[str, Perspective]:
        return self.perspectives


@pytest.fixture
def game() -> Game:
    return Game()


@pytest.fixture
def root_state() -> State:
    return State(active_player = Player.X, name = "root")


def test_alpha_beta_policy_selects_best_action(game: Game, root_state: State) -> None:
    policy: AlphaBetaPolicy[State, State, str, str, Sequence] = AlphaBetaPolicy(
        game,
        max_depth = 2,
    )

    action = policy(root_state, ("choice_1", "choice_2"))

    assert action == "choice_1"


def test_alpha_beta_policy_respects_max_depth(game: Game, root_state: State) -> None:
    shallow_policy: AlphaBetaPolicy[State, State, str, str, Sequence] = AlphaBetaPolicy(
        game,
        max_depth = 1,
    )
    deep_policy: AlphaBetaPolicy[State, State, str, str, Sequence] = AlphaBetaPolicy(
        game,
        max_depth = 2,
    )

    shallow_action = shallow_policy(root_state, ("choice_1", "choice_2"))
    deep_action = deep_policy(root_state, ("choice_1", "choice_2"))

    assert shallow_action == "choice_2"
    assert deep_action == "choice_1"


def test_alpha_beta_policy_reuses_root_state_cache(
    game: Game,
    root_state: State,
) -> None:
    policy: AlphaBetaPolicy[State, State, str, str, Sequence] = AlphaBetaPolicy(
        game,
        max_depth = 2,
        cache_size = 1000,
    )

    first_action = policy(root_state, ("choice_1", "choice_2"))
    first_update_calls = game.update_calls
    second_action = policy(root_state, ("choice_1", "choice_2"))

    assert first_action == "choice_1"
    assert second_action == "choice_1"
    assert first_update_calls > 0
    assert game.update_calls == first_update_calls
