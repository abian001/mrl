import itertools
from abc import abstractmethod, ABC
from typing import Iterable, Tuple, Mapping, Generic
from dataclasses import dataclass, field
from mrl.configuration.player_utils import ChoiceFunction
from mrl.xiangqi.game import (
    Xiangqi as BaseXiangqi,
    Board as BaseBoard,
    Action as BaseAction,
    BaseState,
    XiangqiStdin as BaseXiangqiStdin,
    Player,
    Role,
    Cell,
    Position,
    BasePerspective,
    Observation,
    _make_token_actions
)


Action = int


@dataclass
class OrderCounter:
    general: int = 0
    advisor: int = 0
    elephant: int = 0
    horse: int = 0
    chariot: int = 0
    cannon: int = 0
    soldier: int = 0

    def get_and_incr(self, role):
        if role == Role.GENERAL:
            value = self.general
            self.general += 1
        elif role == Role.ADVISOR:
            value = self.advisor
            self.advisor += 1
        elif role == Role.ELEPHANT:
            value = self.elephant
            self.elephant += 1
        elif role == Role.HORSE:
            value = self.horse
            self.horse += 1
        elif role == Role.CHARIOT:
            value = self.chariot
            self.chariot += 1
        elif role == Role.CANNON:
            value = self.cannon
            self.cannon += 1
        elif role == Role.SOLDIER:
            value = self.soldier
            self.soldier += 1
        else:
            raise ValueError(f"Invalid role passed to get_and_incr: {role}.")
        return value


class Board(BaseBoard):

    def make_action_map(
        self,
        active_player: Player,
    ) -> dict[int, BaseAction]:
        active_general_pos = \
            self.red_general_pos if active_player == Player.RED else self.black_general_pos
        counter = OrderCounter()
        make_action = _make_red_action if active_player == Player.RED else _make_black_action
        return {
            _inverse_action_map[make_action(base_action, role, order)]: base_action
            for (base_action, role, order) in itertools.chain.from_iterable(
                zip(
                    _make_token_actions(cell.token, position, self),
                    itertools.repeat(cell.token.role),
                    itertools.repeat(counter.get_and_incr(cell.token.role))
                )
                for (position, cell) in self._symmetric_cell_iterator(active_player)
                if cell.token.color == active_player and cell.token.role != Role.NONE
            )
            if self._is_action_legal(base_action, active_player, active_general_pos)
        }

    def _symmetric_cell_iterator(self, active_player: Player) -> Iterable[tuple[Position, Cell]]:
        if active_player is Player.RED:
            return self._cell_iterator()
        return zip(
            reversed(self._cell_positions),
            itertools.chain(*(reversed(x) for x in reversed(self.board)))
        )

    def get_order(self, position: Position, role: Role, active_player: Player) -> int:
        order = 0
        for (pos, cell) in self._symmetric_cell_iterator(active_player):
            if pos == position:
                break
            if cell.token.role == role and cell.token.color == active_player:
                order += 1
        return order


@dataclass(repr = False)
class State(BaseState[Board, Action]):

    action_dictionary: dict[Action, BaseAction] = field(default_factory = dict)

    def _update_action_space(self) -> None:
        self.action_dictionary = self.board.make_action_map(self.active_player)
        self.action_space = tuple(self.action_dictionary.keys())


@dataclass(frozen = True)
class PositionalAction:
    role: Role
    order: int
    row: int | None
    column: int | None

    def __repr__(self):
        return str((
            str(self.role),
            self.order,
            self.row,
            self.column
        ))


@dataclass(frozen = True)
class DifferentialAction:
    role: Role
    order: int
    row_offset: int
    column_offset: int

    def __repr__(self):
        return str((
            str(self.role),
            self.order,
            self.row_offset,
            self.column_offset
        ))


_action_map = tuple(
    PositionalAction(role, order, row, None)
    for role in (Role.CHARIOT, Role.CANNON)
    for order in (0, 1)
    for row in range(10)
) + tuple(
    PositionalAction(role, order, None, column)
    for role in (Role.CHARIOT, Role.CANNON)
    for order in (0, 1)
    for column in range(9)
) + tuple(
    DifferentialAction(Role.HORSE, order, r, c)
    for order in (0, 1)
    for (r, c) in ((+2, +1), (+2, -1), (-2, +1), (-2, -1), (+1, +2), (-1, +2), (+1, -2), (-1, -2))
) + tuple(
    DifferentialAction(Role.ELEPHANT, order, r, c)
    for order in (0, 1)
    for (r, c) in ((+2, +2), (+2, -2), (-2, +2), (-2, -2))
) + tuple(
    DifferentialAction(Role.ADVISOR, order, r, c)
    for order in (0, 1)
    for (r, c) in ((+1, +1), (+1, -1), (-1, +1), (-1, -1))
) + tuple(
    DifferentialAction(Role.GENERAL, 0, r, c)
    for (r, c) in ((+1, 0), (-1, 0), (0, +1), (0, -1))
) + tuple(
    DifferentialAction(Role.SOLDIER, order, r, c)
    for order in range(5)
    for (r, c) in ((-1, 0), (0, +1), (0, -1))
)
_inverse_action_map = {
    a: i
    for (i, a) in enumerate(_action_map)
}
action_space_dimension = len(_action_map)


def _make_red_action(base_action, role, order):
    if role in (Role.CHARIOT, Role.CANNON):
        if base_action.origin[0] == base_action.destination[0]:
            return PositionalAction(role, order, None, base_action.destination[1])
        return PositionalAction(role, order, base_action.destination[0], None)
    return DifferentialAction(
        role,
        order,
        base_action.destination[0] - base_action.origin[0],
        base_action.destination[1] - base_action.origin[1]
    )


_max_row = 9
_max_column = 8
def _make_black_action(base_action, role, order):
    if role in (Role.CHARIOT, Role.CANNON):
        if base_action.origin[0] == base_action.destination[0]:
            return PositionalAction(role, order, None, _max_column - base_action.destination[1])
        return PositionalAction(role, order, _max_row - base_action.destination[0], None)
    return DifferentialAction(
        role,
        order,
        - base_action.destination[0] + base_action.origin[0],
        - base_action.destination[1] + base_action.origin[1]
    )


def base_action_to_extended_action(base_action: BaseAction, state: BaseState) -> Action | None:
    token = state.board[base_action.origin]
    order = state.board.get_order(base_action.origin, token.role, state.active_player)

    if state.active_player is Player.RED:
        action = _make_red_action(base_action, token.role, order)
    else:
        assert state.active_player is Player.BLACK
        action = _make_black_action(base_action, token.role, order)

    return _inverse_action_map.get(action)


class BaseEnumerableXiangqi(ABC, Generic[Observation]):

    def __init__(
        self,
        first_player: ChoiceFunction | None = None,
        first_state: str | None = None,
        step_limit: int = 200
    ) -> None:
        self.base_game = BaseXiangqi(first_player, first_state, step_limit)

    def get_players(self) -> Tuple[Player, ...]:
        return self.base_game.get_players()

    @abstractmethod
    def get_perspectives(self) -> Mapping[Player, BasePerspective[State, Observation, Action]]:
        """Returns the players' perspectives"""

    def make_initial_state(self) -> State:
        first_player = self.base_game.first_player(self)
        board = Board(self.base_game.first_state)
        action_dictionary = board.make_action_map(first_player)
        action_space = tuple(action_dictionary.keys())
        return State(board, first_player, action_space, action_dictionary = action_dictionary)

    def update(self, state: State, action: Action) -> State:
        _action = state.action_dictionary[action]
        state.update(_action, self.base_game.step_limit)
        return state


class Perspective(BasePerspective[State, State, Action]):

    def get_observation(self, state: State) -> State:
        return state


class Xiangqi(BaseEnumerableXiangqi):

    def get_perspectives(self) -> Mapping[Player, Perspective]:
        return {
            player: Perspective(player)
            for player in self.get_players()
        }


class XiangqiStdin(BaseXiangqiStdin):

    def _print_actions(self, state, actions):
        numbered_actions = dict(enumerate(actions))
        for (index, action) in numbered_actions.items():
            base_action = state.action_dictionary[action]
            action = _action_map[action]
            print(index, action, base_action)
        return numbered_actions
