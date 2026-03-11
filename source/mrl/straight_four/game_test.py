import pytest
from mrl.test_utils.game_loop import GameRunner
from mrl.test_utils.game_loop import game_runner  # pylint: disable=unused-import
from mrl.straight_four.game import StraightFour, Player


@pytest.mark.quick
def test_player_red_wins_diagonal(game_runner: GameRunner):
    game_runner.run_test(
        game = StraightFour(),
        actions = {
            Player.O: (0, 1, 3, 2, 4, 3),
            Player.X: (1, 2, 2, 3, 3)
        }
    )
    game_runner.assert_histories(
        expected_histories = {
            Player.O: (
                "................................................. O.", "0",
                "..........................................OX..... O.", "1",
                "....................................O.....OXX.... O.", "3",
                "....................................OX....OXXO... O.", "2",
                "..............................O.....OXX...OXXO... O.", "4",
                "..............................OX....OXX...OXXOO.. O.", "3",
                "........................O.....OX....OXX...OXXOO.. .O",
                "End of episode"
            ),
            Player.X: (
                "..........................................O...... X.", "1",
                "....................................O.....OX..... X.", "2",
                "....................................O.....OXXO... X.", "2",
                "..............................O.....OX....OXXO... X.", "3",
                "..............................O.....OXX...OXXOO.. X.", "3",
                "........................O.....OX....OXX...OXXOO.. .O",
                "End of episode"
            )
        },
        expected_global_history = (
            "................................................. O.", ('O', '0'),
            "..........................................O...... X.", ('X', '1'),
            "..........................................OX..... O.", ('O', '1'),
            "....................................O.....OX..... X.", ('X', '2'),
            "....................................O.....OXX.... O.", ('O', '3'),
            "....................................O.....OXXO... X.", ('X', '2'),
            "....................................OX....OXXO... O.", ('O', '2'),
            "..............................O.....OX....OXXO... X.", ('X', '3'),
            "..............................O.....OXX...OXXO... O.", ('O', '4'),
            "..............................O.....OXX...OXXOO.. X.", ('X', '3'),
            "..............................OX....OXX...OXXOO.. O.", ('O', '3'),
            "........................O.....OX....OXX...OXXOO.. .O",
            "End of episode"
        )
    )

@pytest.mark.performance
def test_perfomance(game_runner: GameRunner):
    duration = game_runner.run_performance_test(
        game = StraightFour(),
        actions = {
            Player.O: (0, 1, 3, 2, 4, 3),
            Player.X: (1, 2, 2, 3, 3)
        },
        number_of_times = 20000
    )
    assert duration <= 1.34
