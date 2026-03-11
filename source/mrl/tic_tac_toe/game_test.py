import pytest
from mrl.test_utils.game_loop import GameRunner
from mrl.test_utils.game_loop import game_runner  # pylint: disable=unused-import
from mrl.tic_tac_toe.game import TicTacToe, Player


@pytest.mark.quick
def test_player_o_wins_diagonal_on_last_move(game_runner: GameRunner):
    game_runner.run_test(
        game = TicTacToe(),
        actions = {
            Player.O: (0, 1, 4, 5, 8),
            Player.X: (2, 3, 6, 7)
        }
    )
    game_runner.assert_histories(
        expected_histories = {
            Player.O: (
                "......... O.", "0",
                "O.X...... O.", "1",
                "OOXX..... O.", "4",
                "OOXXO.X.. O.", "5",
                "OOXXOOXX. O.", "8",
                "OOXXOOXXO .O",
                "End of episode"
            ),
            Player.X: (
                "O........ X.", "2",
                "OOX...... X.", "3",
                "OOXXO.... X.", "6",
                "OOXXOOX.. X.", "7",
                "OOXXOOXXO .O",
                "End of episode"
            )
        },
        expected_global_history = (
            "......... O.", ('O', '0'),
            "O........ X.", ('X', '2'),
            "O.X...... O.", ('O', '1'),
            "OOX...... X.", ('X', '3'),
            "OOXX..... O.", ('O', '4'),
            "OOXXO.... X.", ('X', '6'),
            "OOXXO.X.. O.", ('O', '5'),
            "OOXXOOX.. X.", ('X', '7'),
            "OOXXOOXX. O.", ('O', '8'),
            "OOXXOOXXO .O",
            "End of episode"
        )
    )


@pytest.mark.performance
def test_performance(game_runner: GameRunner):
    duration = game_runner.run_performance_test(
        game = TicTacToe(),
        actions = {
            Player.O: (0, 1, 4, 5, 8),
            Player.X: (2, 3, 6, 7)
        },
        number_of_times = 20000
    )
    assert duration <= 0.92
