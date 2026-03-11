from typing import Generic
from mrl.game.game import Player, State
from mrl.game.trackers import MultiResultTracker
from mrl.game.game_loop import make_game_loop
from mrl.alpha_zero.oracle import DeterministicOraclePolicy
from mrl.alpha_zero.mcts import MCTSGame
from mrl.test_utils.policies import RandomPolicy


class EvaluationLoop(Generic[State, Player]):

    def __init__(self, game: MCTSGame[State, Player], model, config):
        self.number_of_tests = config.number_of_tests
        self.result_tracker = self._make_result_tracker(game, config)
        self.game_loop = self._make_game_loop(game, model, config, self.result_tracker)

    def _make_result_tracker(self, game: MCTSGame[State, Player], config):
        model_led_perspectives = {
            player: perspective
            for (player, perspective) in game.get_perspectives().items()
            if player in config.oracle_led_players
        }
        return MultiResultTracker(model_led_perspectives, config.buckets)

    def _make_game_loop(self, game: MCTSGame[State, Player], model, config, result_tracker):
        random_policy: RandomPolicy = RandomPolicy()
        alpha_zero_policy = DeterministicOraclePolicy(model)
        return make_game_loop(
            game = game,
            policies = {
                player: alpha_zero_policy if player in config.oracle_led_players else random_policy
                for player in game.get_players()
            },
            global_observer = result_tracker
        )

    def __call__(self):
        self.result_tracker.clear()
        self.game_loop.run(number_of_times = self.number_of_tests)
        self.result_tracker.print_results()
