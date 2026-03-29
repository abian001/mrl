from enum import Enum
from abc import abstractmethod, ABC
from dataclasses import dataclass
import itertools
import re
from typing import Tuple, Iterable, TypeVar, Generic, Mapping
from mrl.configuration.player_utils import ChoiceFunction
from mrl.game.game import InteractivePolicy, StopEvent, Observation


class Color(Enum):

    NONE = '0'
    RED = 'Red'
    BLACK = 'Black'

    def __str__(self):
        return self.value


Player = Color


class Role(Enum):
    NONE = 0
    GENERAL = 1
    ADVISOR = 2
    ELEPHANT = 3
    HORSE = 4
    CHARIOT = 5
    CANNON = 6
    SOLDIER = 7

    def __str__(self):
        return self.name


@dataclass(frozen = True)
class Token:
    color: Color
    role: Role


@dataclass
class Cell:
    token: Token


Position = Tuple[int, int]


@dataclass(frozen = True)
class Action:
    origin: Position
    destination: Position

    def __repr__(self):
        return str((
            self.origin[0],
            self.origin[1],
            self.destination[0],
            self.destination[1]
        ))


class Board:

    def __init__(self, initial_state: str | None) -> None:
        self.board = self._make_initial_board(initial_state)
        self.red_general_pos = self._get_position(_red_general)
        self.black_general_pos = self._get_position(_black_general)

    def __getitem__(self, position: Position) -> Token:
        return self.board[position[0]][position[1]].token

    def __setitem__(self, position: Position, token: Token) -> None:
        self.board[position[0]][position[1]].token = token

    def apply_action(self, action: Action) -> None:
        moving_token, _ = self._partial_apply_action(action)
        if moving_token is _red_general:
            self.red_general_pos = action.destination
        elif moving_token is _black_general:
            self.black_general_pos = action.destination

    def make_actions(
        self,
        active_player: Player,
    ) -> Tuple[Action, ...]:
        active_general_pos = \
            self.red_general_pos if active_player == Player.RED else self.black_general_pos
        return tuple(
            action
            for action in itertools.chain.from_iterable(
                _make_token_actions(cell.token, position, self)
                for (position, cell) in self._cell_iterator()
                if cell.token.color == active_player and cell.token.role != Role.NONE
            )
            if self._is_action_legal(action, active_player, active_general_pos)
        )

    def _get_cell(self, position: Position) -> Cell:
        return self.board[position[0]][position[1]]

    def _cell_iterator(self) -> Iterable[Tuple[Position, Cell]]:
        return zip(self._cell_positions, itertools.chain.from_iterable(self.board))

    def _partial_apply_action(self, action: Action) -> Tuple[Token, Token]:
        origin_cell = self._get_cell(action.origin)
        destination_cell = self._get_cell(action.destination)
        revert_token = destination_cell.token
        destination_cell.token = origin_cell.token
        origin_cell.token = _empty_token
        return destination_cell.token, revert_token

    def _partial_revert_action(self, action: Action, captured_token: Token) -> None:
        self[action.origin] = self[action.destination]
        self[action.destination] = captured_token

    def _get_position(self, token: Token) -> Position:
        return next(
            position for (position, cell) in self._cell_iterator()
            if cell.token is token
        )

    def _is_action_legal(
        self,
        action: Action,
        active_player: Player,
        active_general_pos: Position,
    ) -> bool:
        origin_token, captured_token = self._partial_apply_action(action)
        general_pos = \
            action.destination if origin_token.role == Role.GENERAL else active_general_pos
        is_illegal = (
            _generals_face_each_other(self, active_player, general_pos) or
            _chariot_or_cannon_attack_general(self, active_player, general_pos) or \
            _horse_attacks_general(self, active_player, general_pos) or \
            _soldier_attacks_general(self, active_player, general_pos)
        )
        self._partial_revert_action(action, captured_token)
        return not is_illegal

    def __repr__(self) -> str:
        return \
            ''.join(_token_to_string[c.token] for c in itertools.chain.from_iterable(self.board))

    @classmethod
    def _make_initial_board(cls, initial_state: str | None) -> Tuple[Tuple[Cell, ...], ...]:
        initial_state = initial_state or cls._initial_state
        if len(initial_state) != cls._size:
            raise ValueError(
                f"Invalid xiangqi initial state length {len(initial_state)}. "
                f"Expected {cls._size} characters."
            )
        state_iterator = iter(initial_state)
        return tuple(
            tuple(
                Cell(_string_to_token[next(state_iterator)])
                for column in range(cls._columns)
            )
            for row in range(cls._rows)
       )

    _rows = 10
    _columns = 9
    _size = _rows * _columns
    _cell_positions: Tuple[Position, ...] = tuple(
        (row, column)
        for (row, column) in itertools.product(range(_rows), range(_columns))
    )
    _initial_state = (
        "rheagaehr"
        "........."
        ".c.....c."
        "s.s.s.s.s"
        "........."
        "........."
        "S.S.S.S.S"
        ".C.....C."
        "........."
        "RHEAGAEHR"
    )


BoardType = TypeVar("BoardType", bound = Board)
StoredActionType = TypeVar("StoredActionType")


@dataclass
class BaseState(Generic[BoardType, StoredActionType]):

    board: BoardType
    active_player: Player
    action_space: tuple[StoredActionType, ...]
    winner: Player | None = None
    is_final: bool = False
    step: int = 0

    def __repr__(self) -> str:
        return str(self.board) + \
            ' ' + _player_to_string(self.active_player) + _player_to_string(self.winner)

    def update(self, action: Action, step_limit: int) -> None:
        self.board.apply_action(action)

        self.active_player = _get_opponent(self.active_player)
        self.step += 1
        self._update_action_space()

        self.is_final = (len(self.action_space) == 0)
        if self.is_final:
            self.winner = _get_opponent(self.active_player)
            self.active_player = Player.NONE
        else:
            self.is_final = (self.step >= step_limit)

    @abstractmethod
    def _update_action_space(self):
        """Updates the content of the action space"""


class State(BaseState[Board, Action]):

    def _update_action_space(self):
        self.action_space = self.board.make_actions(self.active_player)


StateContra = TypeVar('StateContra', contravariant = True, bound = BaseState)
ActionCo = TypeVar('ActionCo', covariant = True)


class BasePerspective(ABC, Generic[StateContra, Observation, ActionCo]):

    def __init__(self, player: Player):
        self.player = player

    @abstractmethod
    def get_observation(self, state: StateContra) -> Observation:
        """Returns the observation made by the player in the game"""

    def get_action_space(self, state: StateContra) -> tuple[ActionCo, ...]:
        return state.action_space

    def get_payoff(self, state: StateContra) -> float:
        if state.winner == self.player:
            return 1.0
        if state.is_final and state.winner is None:
            return 0.5
        return 0.0


class Perspective(BasePerspective[State, State, Action]):

    def get_observation(self, state: State) -> State:
        return state


class Xiangqi:

    def __init__(
        self,
        first_player: ChoiceFunction | None = None,
        first_state: str | None = None,
        step_limit: int = 200
    ) -> None:
        self.first_player = first_player or ChoiceFunction(choices = ('Red',))
        self.first_state = _strip_whitespaces(first_state)
        self.step_limit = step_limit

    def get_players(self) -> Tuple[Player, ...]:
        return (Color.RED, Color.BLACK)

    def get_perspectives(self) -> Mapping[Player, Perspective]:
        return {
            player: Perspective(player)
            for player in self.get_players()
        }

    def make_initial_state(self) -> State:
        first_player = self.first_player(self)
        board = Board(self.first_state)
        action_space = board.make_actions(first_player)
        return State(board, first_player, action_space)

    def update(self, state: State, action: Action) -> State:
        state.update(action, self.step_limit)
        return state


def _player_to_string(player: Player | None) -> str:
    if player == Player.RED:
        return 'R'
    if player == Player.BLACK:
        return 'B'
    return '.'


def _strip_whitespaces(text: str | None):
    if text is None:
        return text
    return re.sub(r"\s+", "", text)


_empty_token = Token(Color.NONE, Role.NONE)
_red_general = Token(Color.RED, Role.GENERAL)
_black_general = Token(Color.BLACK, Role.GENERAL)
_red_horse = Token(Color.RED, Role.HORSE)
_black_horse = Token(Color.BLACK, Role.HORSE)
_red_soldier = Token(Color.RED, Role.SOLDIER)
_black_soldier = Token(Color.BLACK, Role.SOLDIER)
_red_chariot = Token(Color.RED, Role.CHARIOT)
_black_chariot = Token(Color.BLACK, Role.CHARIOT)
_string_to_token = {
    ".": _empty_token,
    "G": _red_general,
    "A": Token(Color.RED, Role.ADVISOR),
    "E": Token(Color.RED, Role.ELEPHANT),
    "H": _red_horse,
    "R": _red_chariot,
    "C": Token(Color.RED, Role.CANNON),
    "S": _red_soldier,
    "g": _black_general,
    "a": Token(Color.BLACK, Role.ADVISOR),
    "e": Token(Color.BLACK, Role.ELEPHANT),
    "h": _black_horse,
    "r": _black_chariot,
    "c": Token(Color.BLACK, Role.CANNON),
    "s": _black_soldier
}


_token_to_string = {
    t: s
    for (s, t) in _string_to_token.items()
}


def _get_opponent(color: Color) -> Color:
    return Color.BLACK if color == Color.RED else Color.RED


def _make_cross_explore_ranges(position: Position) -> Tuple[Iterable[Position], ...]:
    row, column = position
    return (
        ((r, column) for r in range(row + 1, 10)),
        ((r, column) for r in range(row - 1, -1, -1)),
        ((row, c) for c in range(column + 1, 9)),
        ((row, c) for c in range(column - 1, -1, -1))
    )


def _chariot_or_cannon_attack_general(
    board: Board,
    active_player: Player,
    active_general_pos: Position
):
    return any(
        _chariot_or_cannon_attack_general_on_path(board, active_player, explore_range)
        for explore_range in _make_cross_explore_ranges(active_general_pos)
    )


def _chariot_or_cannon_attack_general_on_path(
    board: Board,
    active_player: Player,
    explore_range: Iterable[Position]
) -> bool:
    attacking_chariot = _black_chariot if active_player == Player.RED else _red_chariot
    token_on_path = False
    for position in explore_range:
        token = board[position]
        if token_on_path:
            if token is not _empty_token:
                return token.color != active_player and token.role == Role.CANNON
        else:
            token_on_path = token is not _empty_token
            if token is attacking_chariot:
                return True
    return False


def _make_chariot_actions(
    chariot_position: Position,
    active_player: Player,
    board: Board
) -> Iterable[Action]:
    return itertools.chain.from_iterable(
        _make_chariot_actions_on_path(chariot_position, active_player, board, explore_range)
        for explore_range in _make_cross_explore_ranges(chariot_position)
    )


def _make_cannon_actions(
    cannon_position: Position,
    active_player: Player,
    board: Board
) -> Iterable[Action]:
    return itertools.chain.from_iterable(
        _make_cannon_actions_on_path(cannon_position, active_player, board, explore_range)
        for explore_range in _make_cross_explore_ranges(cannon_position)
    )


def _make_chariot_actions_on_path(
    chariot_position: Position,
    active_player: Player,
    board: Board,
    explore_range: Iterable[Position]
) -> Iterable[Action]:
    for position in explore_range:
        if board[position].color == active_player:
            break
        yield Action(chariot_position, position)
        if board[position] is not _empty_token:
            break


def _make_cannon_actions_on_path(
    cannon_position: Position,
    active_player: Player,
    board: Board,
    explore_range: Iterable[Position]
) -> Iterable[Action]:
    token_on_the_way = False
    for position in explore_range:
        if token_on_the_way:
            token = board[position]
            if token is not _empty_token:
                if token.color == _get_opponent(active_player):
                    yield Action(cannon_position, position)
                break
        elif board[position] is _empty_token:
            yield Action(cannon_position, position)
        else:
            token_on_the_way = True


def _make_token_actions(token: Token, position: Position, board: Board):
    if token is _empty_token:
        return tuple()
    if token.role == Role.SOLDIER:
        return _make_soldier_actions(position, token.color, board)
    if token.role == Role.ADVISOR:
        return _make_advisor_actions(position, token.color, board)
    if token.role == Role.ELEPHANT:
        return _make_elephant_actions(position, token.color, board)
    if token.role == Role.HORSE:
        return _make_horse_actions(position, token.color, board)
    if token.role == Role.CHARIOT:
        return _make_chariot_actions(position, token.color, board)
    if token.role == Role.CANNON:
        return _make_cannon_actions(position, token.color, board)
    if token.role == Role.GENERAL:
        return _make_general_actions(position, token.color, board)
    raise ValueError(f"Invalid token role {token.role}.")


def _make_general_actions(general_pos: Position, active_player: Player, board: Board):
    general_r, general_c = general_pos
    min_row, max_row = (0, 2) if active_player == Color.BLACK else (7, 9)
    return (
        Action(general_pos, (r, c))
        for (dr, dc) in ((+1, 0), (-1, 0), (0, +1), (0, -1))
        if (
            min_row <= (r := general_r + dr) <= max_row and
            3 <= (c := general_c + dc) <= 5 and
            board[(r, c)].color != active_player
        )
    )


def _make_advisor_actions(advisor_pos: Position, active_player: Player, board: Board):
    advisor_r, advisor_c = advisor_pos
    min_row, max_row = (0, 2) if active_player == Color.BLACK else (7, 9)
    return (
        Action(advisor_pos, (r, c))
        for (dr, dc) in ((+1, +1), (+1, -1), (-1, +1), (-1, -1))
        if (
            min_row <= (r := advisor_r + dr) <= max_row and
            3 <= (c := advisor_c + dc) <= 5 and
            board[(r, c)].color != active_player
        )
    )


def _make_elephant_actions(elephant_pos: Position, active_player: Player, board: Board):
    elephant_r, elephant_c = elephant_pos
    min_row, max_row = (0, 4) if active_player == Color.BLACK else (5, 9)
    return (
        Action(elephant_pos, (r, c))
        for ((dr, dc), (block_dr, block_dc)) in (
            ((+2, +2), (+1, +1)),
            ((+2, -2), (+1, -1)),
            ((-2, +2), (-1, +1)),
            ((-2, -2), (-1, -1))
        )
        if (
            min_row <= (r := elephant_r + dr) <= max_row and
            0 <= (c := elephant_c + dc) <= 8 and
            board[(r, c)].color != active_player and
            board[(elephant_r + block_dr, elephant_c + block_dc)] is _empty_token
        )
    )


def _make_horse_actions(horse_pos: Position, active_player: Player, board: Board):
    horse_r, horse_c = horse_pos
    return (
       Action(horse_pos, (r, c))
       for ((dr, dc), (block_dr, block_dc)) in (
            ((+2, +1), (+1, 0)),
            ((+2, -1), (+1, 0)),
            ((-2, +1), (-1, 0)),
            ((-2, -1), (-1, 0)),
            ((+1, +2), (0, +1)),
            ((-1, +2), (0, +1)),
            ((+1, -2), (0, -1)),
            ((-1, -2), (0, -1))
        )
        if (
            0 <= (r := horse_r + dr) <= 9 and
            0 <= (c := horse_c + dc) <= 8 and
            board[(r, c)].color != active_player and
            board[(horse_r + block_dr, horse_c + block_dc)] is _empty_token
        )
    )


def _make_soldier_actions(soldier_pos: Position, active_player: Player, board: Board):
    soldier_r, soldier_c = soldier_pos
    direction, promoted = (-1, soldier_r < 5) if active_player == Color.RED else (+1, soldier_r > 4)
    displacements = ((direction, 0), (0, +1), (0, -1)) if promoted else ((direction, 0),)
    return (
        Action(soldier_pos, (r, c))
        for (dr, dc) in displacements
        if (
            0 <= (r := soldier_r + dr) <= 9 and
            0 <= (c := soldier_c + dc) <= 8 and
            board[(r, c)].color != active_player
        )
    )


def _generals_face_each_other(board: Board, active_player: Player, general_pos: Position):
    row, column = general_pos
    explore_range = range(row + 1, 10) if active_player == Color.BLACK else range(row - 1, -1, -1)
    token = next((
        token for r in explore_range
        if (token := board[(r, column)]) is not _empty_token
    ), _empty_token)
    return token.role == Role.GENERAL


def _horse_attacks_general(board: Board, active_player: Player, general_pos: Position):
    attacking_horse = _black_horse if active_player == Color.RED else _red_horse
    general_r, general_c = general_pos
    return any(
        board[(r, c)] is attacking_horse and
        board[(general_r + block_dr, general_c + block_dc)] is _empty_token
        for ((dr, dc), (block_dr, block_dc)) in (
            ((+2, +1), (+1, +1)),
            ((+2, -1), (+1, -1)),
            ((-2, +1), (-1, +1)),
            ((-2, -1), (-1, -1)),
            ((+1, +2), (+1, +1)),
            ((-1, +2), (-1, +1)),
            ((+1, -2), (+1, -1)),
            ((-1, -2), (-1, -1))
        )
        if (
            0 <= (r := general_r + dr) <= 9 and
            0 <= (c := general_c + dc) <= 8
        )
    )


def _soldier_attacks_general(board: Board, active_player: Player, general_pos: Position):
    attacking_soldier = _black_soldier if active_player == Color.RED else _red_soldier
    direction = +1 if attacking_soldier is _red_soldier else -1
    general_r, general_c = general_pos
    return any(
        board[(r, c)] is attacking_soldier
        for (dr, dc) in ((direction, 0), (0, +1), (0, -1))
        if (
            0 <= (r := general_r + dr) <= 9 and
            0 <= (c := general_c + dc) <= 8
        )
    )


class XiangqiStdin(InteractivePolicy):

    def __init__(self):
        self.last_state = None

    def __call__(self, state: State, action_space: tuple[Action, ...]) -> Action:
        assert state == self.last_state
        print("Choose - Choose an invalid action to stop")
        numbered_actions = self._print_actions(state, action_space)

        try:
            action = int(input())
        except ValueError as error:
            raise StopEvent("Invalid Action") from error

        if action not in numbered_actions:
            raise StopEvent("Invalid Action")
        return numbered_actions[action]

    def _print_actions(self, state, actions):
        numbered_actions = dict(enumerate(actions))
        for (index, action) in numbered_actions.items():
            print(index, action, state.board[action.origin].role)
        return numbered_actions

    def notify_observation(self, observation: State) -> None:
        self.last_state = observation
        print(f"\n--Player: {_player_to_string(observation.active_player)}--")
        print('\n'.join(
            ''.join(_token_to_string[c.token] for c in row)
            for row in observation.board.board
        ))
        print("------")
        if observation.is_final:
            print("The game is over")
            print(f"The winner is {_player_to_string(observation.winner)}")

    def notify_action(self, player: Player, action: Action) -> None:
        print(f"Player {player} used action {action}")

    def notify_stop_event(self, stop_event: StopEvent) -> None:
        print("Stop event observed")
