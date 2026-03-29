from dataclasses import dataclass
from typing import Mapping
import pytest
from mrl.game.game import InvalidGame, Perspective, Policy
from mrl.game.game_loop import make_game_loop


@dataclass
class InvalidTurnBasedState:
    is_final: bool = False


class DummyPerspective(Perspective[InvalidTurnBasedState, InvalidTurnBasedState, tuple[int, ...]]):

    def get_observation(self, state: InvalidTurnBasedState) -> InvalidTurnBasedState:
        return state

    def get_action_space(self, state: InvalidTurnBasedState) -> tuple[int, ...]:
        _ = state
        return (0,)


class DummyPolicy(Policy[InvalidTurnBasedState, int, tuple[int, ...]]):

    def __call__(
        self,
        observation: InvalidTurnBasedState,
        action_space: tuple[int, ...]
    ) -> int:
        _ = observation
        return action_space[0]


class InvalidTurnBasedGame:

    def make_initial_state(self) -> InvalidTurnBasedState:
        return InvalidTurnBasedState()

    def get_players(self) -> tuple[str, ...]:
        return ('O',)

    def get_perspectives(self) -> Mapping[str, DummyPerspective]:
        return {'O': DummyPerspective()}

    def update(self, state: InvalidTurnBasedState, action: int) -> InvalidTurnBasedState:
        _ = action
        return state


@pytest.mark.quick
def test_make_game_loop_rejects_turn_based_state_without_active_player():
    with pytest.raises(InvalidGame) as error:
        make_game_loop(InvalidTurnBasedGame(), {'O': DummyPolicy()})
    assert str(error.value) == (
        "State class <class 'mrl.game.game_loop_test.InvalidTurnBasedState'> "
        "does not implement required protocol HasActivePlayer."
    )
