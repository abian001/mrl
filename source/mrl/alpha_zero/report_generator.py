from typing import Generic

from mrl.alpha_zero.context import ReportGeneratorContext
from mrl.alpha_zero.mcts import MCTSGame
from mrl.game.game import Player, State
from mrl.game.game_loop import make_game_loop
from mrl.game.trackers import MultiResultTracker


class ReportGenerator(Generic[State, Player]):

    def __init__(
        self,
        game: MCTSGame[State, Player],
        context: ReportGeneratorContext,
    ):
        self.number_of_tests = context.number_of_tests
        self.result_tracker = self._make_result_tracker(game, context)
        self.game_loop = make_game_loop(
            game = game,
            policies = context.policies,
            global_observer = self.result_tracker,
        )

    def _make_result_tracker(
        self,
        game: MCTSGame[State, Player],
        context: ReportGeneratorContext,
    ) -> MultiResultTracker:
        observed_perspectives = {
            player: perspective
            for (player, perspective) in game.get_perspectives().items()
            if player in context.observed_players
        }
        if context.buckets is None:
            raise TypeError("ReportGeneratorContext.buckets must not be None.")
        return MultiResultTracker(observed_perspectives, context.buckets)

    def __call__(self) -> None:
        self.result_tracker.clear()
        self.game_loop.run(number_of_times = self.number_of_tests)
        self.result_tracker.print_results()
