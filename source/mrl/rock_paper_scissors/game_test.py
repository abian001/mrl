import pytest
from mrl.test_utils.game_loop import GameRunner
from mrl.test_utils.game_loop import game_runner  # pylint: disable=unused-import
from mrl.rock_paper_scissors.game import RockPaperScissors, Action, Player


@pytest.mark.quick
def test_rock_paper_scissors(game_runner: GameRunner):
    game_runner.run_test(
        game = RockPaperScissors(),
        actions = {
            Player.O: (Action.ROCK, Action.PAPER, Action.PAPER, Action.ROCK, Action.PAPER),
            Player.X: (Action.ROCK, Action.SCISSORS, Action.ROCK, Action.SCISSORS, Action.PAPER)
        }
    )
    game_runner.assert_histories(
        expected_histories = {
            Player.O: (
                "S0: 0 vs 0", "rock",
                "S1: 0 vs 0", "paper",
                "S2: 0 vs 1", "paper",
                "S3: 1 vs 1", "rock",
                "S4: 2 vs 1", "paper",
                "S5: 2 vs 1",
                "End of episode"
            ),
            Player.X: (
                "S0: 0 vs 0", "rock",
                "S1: 0 vs 0", "scissors",
                "S2: 0 vs 1", "rock",
                "S3: 1 vs 1", "scissors",
                "S4: 2 vs 1", "paper",
                "S5: 2 vs 1",
                "End of episode"
            )
        },
        expected_global_history = (
            "S0: 0 vs 0", ("rock", "rock"),
            "S1: 0 vs 0", ("paper", "scissors"),
            "S2: 0 vs 1", ("paper", "rock"),
            "S3: 1 vs 1", ("rock", "scissors"),
            "S4: 2 vs 1", ("paper", "paper"),
            "S5: 2 vs 1",
            "End of episode"
        )
    )


@pytest.mark.performance
def test_performance(game_runner: GameRunner):
    duration = game_runner.run_performance_test(
        game = RockPaperScissors(),
        actions = {
            Player.O: (Action.ROCK, Action.PAPER, Action.PAPER, Action.ROCK, Action.PAPER),
            Player.X: (Action.ROCK, Action.SCISSORS, Action.ROCK, Action.SCISSORS, Action.PAPER)
        },
        number_of_times = 20000
    )
    assert duration <= 0.17
