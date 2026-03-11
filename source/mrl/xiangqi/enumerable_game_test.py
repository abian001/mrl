from dataclasses import dataclass
from collections import Counter
import pytest
from mrl.test_utils.game_loop import GameRunner
from mrl.test_utils.game_loop import game_runner  # pylint: disable=unused-import
from mrl.xiangqi.enumerable_game import (
    Xiangqi,
    Player,
    Board,
    Role,
    _action_map,
    _inverse_action_map,
    DifferentialAction,
    PositionalAction
)
from mrl.xiangqi.game_test import _make_symmetric_board


@pytest.mark.quick
def test_large_play(game_runner: GameRunner):
    red_actions = tuple(_inverse_action_map[x] for x in(
        DifferentialAction(Role.SOLDIER, 2, -1, 0),
        DifferentialAction(Role.SOLDIER, 0, -1, 0),
        DifferentialAction(Role.SOLDIER, 0, 0, -1),
        DifferentialAction(Role.SOLDIER, 0, 0, -1)
    ))
    black_actions = tuple(_inverse_action_map[x] for x in(
        DifferentialAction(Role.SOLDIER, 2, -1, 0),
        PositionalAction(Role.CANNON, 1, None, 4),
        PositionalAction(Role.CANNON, 0, 8, None),
        PositionalAction(Role.CANNON, 1, None, 4)
    ))
    game_runner.run_test(
        game = Xiangqi(),
        actions = {
            Player.RED: red_actions,
            Player.BLACK: black_actions
        }
    )
    game_runner.assert_histories(
        expected_histories = {
            Player.RED: (
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
                " R.", str(red_actions[0]),
                "rheagaehr"
                "........."
                ".c.....c."
                "s.s...s.s"
                "....s...."
                "....S...."
                "S.S...S.S"
                ".C.....C."
                "........."
                "RHEAGAEHR"
                " R.", str(red_actions[1]),
                "rheagaehr"
                "........."
                "....c..c."
                "s.s...s.s"
                "....S...."
                "........."
                "S.S...S.S"
                ".C.....C."
                "........."
                "RHEAGAEHR"
                " R.", str(red_actions[2]),
                "rheagaehr"
                ".......c."
                "....c...."
                "s.s...s.s"
                "...S....."
                "........."
                "S.S...S.S"
                ".C.....C."
                "........."
                "RHEAGAEHR"
                " R.", str(red_actions[3]),
                "rheagaehr"
                "....c...."
                "....c...."
                "s.s...s.s"
                "..S......"
                "........."
                "S.S...S.S"
                ".C.....C."
                "........."
                "RHEAGAEHR"
                " .B", "End of episode"
            ),
            Player.BLACK: (
                "rheagaehr"
                "........."
                ".c.....c."
                "s.s.s.s.s"
                "........."
                "....S...."
                "S.S...S.S"
                ".C.....C."
                "........."
                "RHEAGAEHR"
                " B.", str(black_actions[0]),
                "rheagaehr"
                "........."
                ".c.....c."
                "s.s...s.s"
                "....S...."
                "........."
                "S.S...S.S"
                ".C.....C."
                "........."
                "RHEAGAEHR"
                " B.", str(black_actions[1]),
                "rheagaehr"
                "........."
                "....c..c."
                "s.s...s.s"
                "...S....."
                "........."
                "S.S...S.S"
                ".C.....C."
                "........."
                "RHEAGAEHR"
                " B.", str(black_actions[2]),
                "rheagaehr"
                ".......c."
                "....c...."
                "s.s...s.s"
                "..S......"
                "........."
                "S.S...S.S"
                ".C.....C."
                "........."
                "RHEAGAEHR"
                " B.", str(black_actions[3]),
                "rheagaehr"
                "....c...."
                "....c...."
                "s.s...s.s"
                "..S......"
                "........."
                "S.S...S.S"
                ".C.....C."
                "........."
                "RHEAGAEHR"
                " .B", "End of episode"
            )
        },
        expected_global_history = (
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
            " R.", ('Red', str(red_actions[0])),
            "rheagaehr"
            "........."
            ".c.....c."
            "s.s.s.s.s"
            "........."
            "....S...."
            "S.S...S.S"
            ".C.....C."
            "........."
            "RHEAGAEHR"
            " B.", ('Black', str(black_actions[0])),
            "rheagaehr"
            "........."
            ".c.....c."
            "s.s...s.s"
            "....s...."
            "....S...."
            "S.S...S.S"
            ".C.....C."
            "........."
            "RHEAGAEHR"
            " R.", ('Red', str(red_actions[1])),
            "rheagaehr"
            "........."
            ".c.....c."
            "s.s...s.s"
            "....S...."
            "........."
            "S.S...S.S"
            ".C.....C."
            "........."
            "RHEAGAEHR"
            " B.", ('Black', str(black_actions[1])),
            "rheagaehr"
            "........."
            "....c..c."
            "s.s...s.s"
            "....S...."
            "........."
            "S.S...S.S"
            ".C.....C."
            "........."
            "RHEAGAEHR"
            " R.", ('Red', str(red_actions[2])),
            "rheagaehr"
            "........."
            "....c..c."
            "s.s...s.s"
            "...S....."
            "........."
            "S.S...S.S"
            ".C.....C."
            "........."
            "RHEAGAEHR"
            " B.", ('Black', str(black_actions[2])),
            "rheagaehr"
            ".......c."
            "....c...."
            "s.s...s.s"
            "...S....."
            "........."
            "S.S...S.S"
            ".C.....C."
            "........."
            "RHEAGAEHR"
            " R.", ('Red', str(red_actions[3])),
            "rheagaehr"
            ".......c."
            "....c...."
            "s.s...s.s"
            "..S......"
            "........."
            "S.S...S.S"
            ".C.....C."
            "........."
            "RHEAGAEHR"
            " B.", ('Black', str(black_actions[3])),
            "rheagaehr"
            "....c...."
            "....c...."
            "s.s...s.s"
            "..S......"
            "........."
            "S.S...S.S"
            ".C.....C."
            "........."
            "RHEAGAEHR"
            " .B", "End of episode"
        )
    )

@pytest.mark.performance
def test_perfomance(game_runner: GameRunner):
    red_actions = tuple(_inverse_action_map[x] for x in(
        DifferentialAction(Role.SOLDIER, 2, -1, 0),
        DifferentialAction(Role.SOLDIER, 0, -1, 0),
        DifferentialAction(Role.SOLDIER, 0, 0, -1),
        DifferentialAction(Role.SOLDIER, 0, 0, -1)
    ))
    black_actions = tuple(_inverse_action_map[x] for x in(
        DifferentialAction(Role.SOLDIER, 2, -1, 0),
        PositionalAction(Role.CANNON, 1, None, 4),
        PositionalAction(Role.CANNON, 0, 8, None),
        PositionalAction(Role.CANNON, 1, None, 4)
    ))
    duration = game_runner.run_performance_test(
        game = Xiangqi(),
        actions = {
            Player.RED: red_actions,
            Player.BLACK: black_actions,
        },
        number_of_times = 200
    )
    assert duration <= 2.00


@dataclass
class MoveData:
    name: str
    board: str
    expected_actions: tuple[PositionalAction | DifferentialAction, ...]


black_move_data = (
    MoveData(
        name = 'general_moves',
        board = (
            "........."
            "....gS..."
            "........."
            "........."
            "........."
            "........."
            "........."
            "........."
            "....A...."
            "....G...."
        ),
        expected_actions = (
            DifferentialAction(Role.GENERAL, 0, 0, +1),
            DifferentialAction(Role.GENERAL, 0, 0, -1),
            DifferentialAction(Role.GENERAL, 0, +1, 0),
            DifferentialAction(Role.GENERAL, 0, -1, 0),
        )
    ),
    MoveData(
        name = 'elephant_moves',
        board = (
            "..e.g.e.."
            "........."
            "........."
            "........."
            "........."
            "........."
            "........."
            "........."
            "........."
            "...RRG..."
        ),
        expected_actions = (
            DifferentialAction(Role.ELEPHANT, 0, -2, +2),
            DifferentialAction(Role.ELEPHANT, 1, -2, -2)
        )
    ),
    MoveData(
        name = 'advisor_moves',
        board = (
            "...aga..."
            "........."
            "........."
            "........."
            "........."
            "........."
            "........."
            "....R...."
            "....A...."
            "....G...."
        ),
        expected_actions = (
            DifferentialAction(Role.ADVISOR, 1, -1, -1),
            DifferentialAction(Role.ADVISOR, 0, -1, +1)
        )
    ),
    MoveData(
        name = 'cannon_and_chariot_moves',
        board = (
            "r...g...."
            "r.......c"
            "........."
            ".......sc"
            "........."
            "........."
            "........."
            "........."
            "........."
            "...RRG..."
        ),
        expected_actions = (
            PositionalAction(Role.CANNON, 1, None, 4),
            PositionalAction(Role.CHARIOT, 0, None, 4)
        )
    ),
    MoveData(
        name = 'soldier_moves',
        board = (
            "....g...."
            "R........"
            "........."
            "..s.s.s.."
            "........."
            "........."
            "s........"
            "........."
            "........."
            "...R.G..s"
        ),
        expected_actions = (
            DifferentialAction(Role.SOLDIER, 0, 0, +1),
            DifferentialAction(Role.SOLDIER, 1, -1, 0),
            DifferentialAction(Role.SOLDIER, 1, 0, -1),
            DifferentialAction(Role.SOLDIER, 2, -1, 0),
            DifferentialAction(Role.SOLDIER, 3, -1, 0),
            DifferentialAction(Role.SOLDIER, 4, -1, 0)
        )
    )
)


@pytest.mark.parametrize('move_data', black_move_data)
@pytest.mark.quick
def test_black_action_map(move_data: MoveData):
    board = Board(move_data.board)
    actions = tuple(_action_map[x] for x in board.make_action_map(Player.BLACK))
    assert Counter(actions) == Counter(move_data.expected_actions)


@pytest.mark.parametrize('move_data', black_move_data)
@pytest.mark.quick
def test_red_action_map(move_data: MoveData):
    board = Board(_make_symmetric_board(move_data.board))
    actions = tuple(_action_map[x] for x in board.make_action_map(Player.RED))
    assert Counter(actions) == Counter(move_data.expected_actions)
