from dataclasses import dataclass
from collections import Counter
import pytest
from mrl.test_utils.game_loop import GameRunner
from mrl.test_utils.game_loop import game_runner  # pylint: disable=unused-import
from mrl.xiangqi.game import (
    Xiangqi,
    Player,
    Action,
    Board
)


@pytest.mark.quick
def test_large_play(game_runner: GameRunner):
    game_runner.run_test(
        game = Xiangqi(),
        actions = {
            Player.RED: (
                Action((6, 4), (5, 4)),
                Action((5, 4), (4, 4)),
                Action((4, 4), (4, 3)),
                Action((4, 3), (4, 2))
            ),
            Player.BLACK: (
                Action((3, 4), (4, 4)),
                Action((2, 7), (2, 4)),
                Action((2, 1), (1, 1)),
                Action((1, 1), (1, 4))
            )
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
                " R.", "(6, 4, 5, 4)",
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
                " R.", "(5, 4, 4, 4)",
                "rheagaehr"
                "........."
                ".c..c...."
                "s.s...s.s"
                "....S...."
                "........."
                "S.S...S.S"
                ".C.....C."
                "........."
                "RHEAGAEHR"
                " R.", "(4, 4, 4, 3)",
                "rheagaehr"
                ".c......."
                "....c...."
                "s.s...s.s"
                "...S....."
                "........."
                "S.S...S.S"
                ".C.....C."
                "........."
                "RHEAGAEHR"
                " R.", "(4, 3, 4, 2)",
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
                " B.", "(3, 4, 4, 4)",
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
                " B.", "(2, 7, 2, 4)",
                "rheagaehr"
                "........."
                ".c..c...."
                "s.s...s.s"
                "...S....."
                "........."
                "S.S...S.S"
                ".C.....C."
                "........."
                "RHEAGAEHR"
                " B.", "(2, 1, 1, 1)",
                "rheagaehr"
                ".c......."
                "....c...."
                "s.s...s.s"
                "..S......"
                "........."
                "S.S...S.S"
                ".C.....C."
                "........."
                "RHEAGAEHR"
                " B.", "(1, 1, 1, 4)",
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
            " R.", ('Red', '(6, 4, 5, 4)'),
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
            " B.", ('Black', '(3, 4, 4, 4)'),
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
            " R.", ('Red', '(5, 4, 4, 4)'),
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
            " B.", ('Black', '(2, 7, 2, 4)'),
            "rheagaehr"
            "........."
            ".c..c...."
            "s.s...s.s"
            "....S...."
            "........."
            "S.S...S.S"
            ".C.....C."
            "........."
            "RHEAGAEHR"
            " R.", ('Red', '(4, 4, 4, 3)'),
            "rheagaehr"
            "........."
            ".c..c...."
            "s.s...s.s"
            "...S....."
            "........."
            "S.S...S.S"
            ".C.....C."
            "........."
            "RHEAGAEHR"
            " B.", ('Black', '(2, 1, 1, 1)'),
            "rheagaehr"
            ".c......."
            "....c...."
            "s.s...s.s"
            "...S....."
            "........."
            "S.S...S.S"
            ".C.....C."
            "........."
            "RHEAGAEHR"
            " R.", ('Red', '(4, 3, 4, 2)'),
            "rheagaehr"
            ".c......."
            "....c...."
            "s.s...s.s"
            "..S......"
            "........."
            "S.S...S.S"
            ".C.....C."
            "........."
            "RHEAGAEHR"
            " B.", ('Black', '(1, 1, 1, 4)'),
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
    duration = game_runner.run_performance_test(
        game = Xiangqi(),
        actions = {
            Player.RED: (
                Action((6, 4), (5, 4)),
                Action((5, 4), (4, 4)),
                Action((4, 4), (4, 3)),
                Action((4, 3), (4, 2))
            ),
            Player.BLACK: (
                Action((3, 4), (4, 4)),
                Action((2, 7), (2, 4)),
                Action((2, 1), (1, 1)),
                Action((1, 1), (1, 4))
            )
        },
        number_of_times = 200
    )
    assert duration <= 2.00


@dataclass
class MoveData:
    name: str
    board: str
    expected_actions: tuple[Action, ...]


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
            Action((1, 4), (2, 4)),
            Action((1, 4), (0, 4)),
            Action((1, 4), (1, 3)),
            Action((1, 4), (1, 5))
        )
    ),
    MoveData(
        name = 'general_cannot_leave_fortress',
        board = (
            "........."
            "........."
            ".....g..."
            "........."
            "........."
            "........."
            "........."
            "........."
            "....A...."
            "....G...."
        ),
        expected_actions = (
            Action((2, 5), (1, 5)),
            Action((2, 5), (2, 4))
        )
    ),
    MoveData(
        name = 'generals_cannot_face_each_other',
        board = (
            "........."
            "........."
            ".....g..."
            "........."
            "........."
            "........."
            "........."
            "........."
            "........."
            "....G...."
        ),
        expected_actions = (
            Action((2, 5), (1, 5)),
        )
    ),
    MoveData(
        name = 'general_cannot_move_under_threat',
        board = (
            "R........"
            "....g...."
            ".H......."
            "........."
            "........."
            ".....C..."
            ".....C..."
            "........."
            "....A...."
            "....G...."
        ),
        expected_actions = (
            Action((1, 4), (2, 4)),
        )
    ),
    MoveData(
        name = 'advisor_moves',
        board = (
            "....gC..."
            "....a...."
            "........."
            "........."
            "........."
            "........."
            "........."
            "...R.R..."
            "....A...."
            "....G...."
        ),
        expected_actions = (
            Action((1, 4), (0, 3)),
            Action((1, 4), (0, 5)),
            Action((1, 4), (2, 3)),
            Action((1, 4), (2, 5))
        )
    ),
    MoveData(
        name = 'advisor_cannot_leave_the_fortress',
        board = (
            "....g...."
            "........R"
            ".....a..."
            "........."
            "........."
            "........."
            "........."
            "...R.C..."
            "....A...."
            "....G...."
        ),
        expected_actions = (
            Action((2, 5), (1, 4)),
        )
    ),
    MoveData(
        name = 'advisor_cannot_let_generals_face_each_other',
        board = (
            "....g...."
            "....a...."
            "........."
            "........."
            "........."
            "........."
            "........."
            "........."
            "........."
            "....G...."
        ),
        expected_actions = (
            Action((0, 4), (0, 3)),
            Action((0, 4), (0, 5))
        )
    ),
    MoveData(
        name = 'advisor_cannot_let_general_under_threat',
        board = (
            "...g....."
            "....aH..."
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
            Action((0, 3), (0, 4)),
            Action((0, 3), (1, 3))
        )
    ),
    MoveData(
        name = 'elephant_moves',
        board = (
            "....g.S.."
            "CC......."
            "....e...."
            "........."
            "........."
            "........."
            "........."
            "...R.R..."
            "....A...."
            "....G...."
        ),
        expected_actions = (
            Action((2, 4), (0, 2)),
            Action((2, 4), (0, 6)),
            Action((2, 4), (4, 2)),
            Action((2, 4), (4, 6))
        )
    ),
    MoveData(
        name = 'elephant_cannot_leave_own_side_of_river',
        board = (
            "....g...."
            "CC......."
            "........."
            "........."
            "......e.."
            "........."
            "........."
            "...R.R..."
            "....A...."
            "....G...."
        ),
        expected_actions = (
            Action((4, 6), (2, 4)),
            Action((4, 6), (2, 8)),
        )
    ),
    MoveData(
        name = 'horse_moves',
        board = (
            "....g...."
            "CC......."
            "........."
            "......S.."
            "....h...."
            "........."
            "........."
            "...R.R..."
            "....A...."
            "....G...."
        ),
        expected_actions = (
            Action((4, 4), (2, 3)),
            Action((4, 4), (2, 5)),
            Action((4, 4), (3, 2)),
            Action((4, 4), (3, 6)),
            Action((4, 4), (5, 2)),
            Action((4, 4), (5, 6)),
            Action((4, 4), (6, 3)),
            Action((4, 4), (6, 5))
        )
    ),
    MoveData(
        name = 'chariot_moves',
        board = (
            "....g...."
            "CC......."
            "........."
            "........."
            "....r.S.."
            "........."
            "........."
            "...R.R..."
            "....A...."
            "....G...."
        ),
        expected_actions = (
            Action((4, 4), (1, 4)),
            Action((4, 4), (2, 4)),
            Action((4, 4), (3, 4)),
            Action((4, 4), (5, 4)),
            Action((4, 4), (6, 4)),
            Action((4, 4), (7, 4)),
            Action((4, 4), (8, 4)),
            Action((4, 4), (4, 0)),
            Action((4, 4), (4, 1)),
            Action((4, 4), (4, 2)),
            Action((4, 4), (4, 3)),
            Action((4, 4), (4, 5)),
            Action((4, 4), (4, 6)),
        )
    ),
    MoveData(
        name = 'cannon_moves',
        board = (
            "....g...."
            "CC......."
            "........."
            "........."
            "....c.H.S"
            "........."
            "........."
            "...RER..."
            "....A...."
            "....G...."
        ),
        expected_actions = (
            Action((4, 4), (1, 4)),
            Action((4, 4), (2, 4)),
            Action((4, 4), (3, 4)),
            Action((4, 4), (5, 4)),
            Action((4, 4), (6, 4)),
            Action((4, 4), (8, 4)),
            Action((4, 4), (4, 0)),
            Action((4, 4), (4, 1)),
            Action((4, 4), (4, 2)),
            Action((4, 4), (4, 3)),
            Action((4, 4), (4, 5)),
            Action((4, 4), (4, 8)),
        )
    ),
    MoveData(
        name = 'unpromoted_soldier_moves',
        board = (
            "....g...."
            "CC......."
            "........."
            "........."
            "......s.."
            "........."
            "........."
            "...R.R..."
            "....A...."
            "....G...."
        ),
        expected_actions = (
            Action((4, 6), (5, 6)),
        )
    ),
    MoveData(
        name = 'promoted_soldier_moves',
        board = (
            "....g...."
            "CC......."
            "........."
            "........."
            "........."
            "........."
            "........."
            "...R.Rs.."
            "....A...."
            "....G...."
        ),
        expected_actions = (
            Action((7, 6), (7, 5)),
            Action((7, 6), (7, 7)),
            Action((7, 6), (8, 6)),
        )
    ),
    MoveData(
        name = 'cannot_must_capture_to_save_general',
        board = (
            "....R...."
            "....g.H.."
            "........."
            "........."
            "....c...."
            "........."
            "........."
            "...CER..."
            "...CA...."
            "....G...."
        ),
        expected_actions = (
            Action((4, 4), (0, 4)),
        )
    ),
    MoveData(
        name = 'cannon_cannot_take_the_soldier',
        board = (
            "........."
            "....a...r"
            "rchgS...."
            "........."
            "........."
            "........."
            "..S.S.S.S"
            ".C.....C."
            "........."
            "RHEAGAEHR"
        ),
        expected_actions = (
            Action((2, 3), (1, 3)),
            Action((2, 3), (2, 4))
        )
    )
)


@pytest.mark.parametrize('move_data', black_move_data)
@pytest.mark.quick
def test_black_single_move(move_data: MoveData):
    board = Board(move_data.board)
    actions = board.make_actions(Player.BLACK)
    assert \
        Counter(actions) == Counter(move_data.expected_actions), \
        f'Unexpected actions for test {move_data.name}'


@pytest.mark.parametrize('move_data', black_move_data)
@pytest.mark.quick
def test_red_single_move(move_data: MoveData):
    board = Board(_make_symmetric_board(move_data.board))
    actions = board.make_actions(Player.RED)
    assert \
        Counter(actions) == Counter(_make_symmetric_actions(move_data.expected_actions)), \
        f'Unexpected actions for test {move_data.name}'


def _make_symmetric_board(board: str) -> str:
    return ''.join(x.swapcase() for x in reversed(board))


def _make_symmetric_actions(actions: tuple[Action, ...]) -> tuple[Action, ...]:
    # pylint: disable=protected-access
    return tuple(
        Action(
            (Board._rows - 1 - a.origin[0], Board._columns - 1 - a.origin[1]),
            (Board._rows - 1 - a.destination[0], Board._columns - 1 - a.destination[1])
        )
        for a in actions
    )
