from enum import Enum
import os
from typing import Generic
import argparse
import yaml
from mrl.game.game import Player
from mrl.game.game_loop import make_game_loop
from mrl.game.trackers import MultiResultTracker
from mrl.configuration.game_runner_configuration import (
    GameRunnerSpecification,
    make_game_runner_specification
)
from mrl.test_utils.policies import ManualPolicy
from mrl.test_utils.exit_handler import try_main
from mrl.alpha_zero.mcts_observation import PayoffPerspective


def main():
    try_main(_run)


def _run():
    arguments = _make_arguments()
    runner = GameRunner(arguments.config, arguments.mode)
    runner.run()


def _make_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        type = _parse_configuration,
        help = "Path to the game runner configuration."
    )
    parser.add_argument(
        '--mode',
        type = _parse_test_mode,
        choices = list(TestMode),
        required = True,
        help = (
            "Mode of execution:\n"
            "(evaluate) evaluate policies against each other,\n"
            "(terminal) play against policies in terminal,\n"
            "(gui) play against policies in gui."
        )
    )
    return parser.parse_args()


class TestMode(Enum):
    EVALUATE = 'evaluate'  # autonomous play
    TERMINAL = 'terminal'  # manual play using standard input
    GUI = 'gui'  # manual play using GUI

    def __str__(self):
        return self.value


def _parse_test_mode(value: str) -> TestMode:
    try:
        return TestMode(value.lower())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f'--mode must be one of {[str(x) for x in list(TestMode)]}'
        ) from error


def _parse_configuration(path: str) -> GameRunnerSpecification:
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(
            f'Input file {path} does not exist.'
        )

    try:
        with open(path, 'r', encoding = 'utf-8') as yaml_file:
            config_data = yaml.safe_load(yaml_file)
    except Exception as error:
        raise argparse.ArgumentTypeError(
            f'Failed to access the file {path} with error: "{error}".'
        ) from error

    try:
        return make_game_runner_specification(config_data)
    except Exception as error:
        raise argparse.ArgumentTypeError(
            'Invalid game runner configuration in input. '
            f'Parsing failed with error: "{error}"'
        ) from error


class GameRunner(Generic[Player]):

    def __init__(self, specification: GameRunnerSpecification[Player], mode: TestMode):
        self.specification = specification
        self.mode: TestMode = mode
        self.manual_players = tuple((
            player for (player, policy) in self.specification.policies.items()
            if isinstance(policy, ManualPolicy)
        ))

    def run(self):
        if self.mode == TestMode.EVALUATE:
            if len(self.manual_players) > 0:
                raise TypeError(
                    "EVALUATE mode was requested but some players use the manual policy "
                    "Please use autonomous policies for all players."
                )
            self._run_evaluation()
        elif self.mode == TestMode.TERMINAL:
            if len(self.manual_players) == 0:
                raise TypeError(
                    "TERMINAL mode was requested but no player uses the manual policy. "
                    "Please use the ManualPolicy for at least one player. "
                    "NOTE: works best with one manual player only, as the output "
                    "stream is replicated once for each player."
                )
            self._run_terminal_play()
        elif self.mode == TestMode.GUI:
            if len(self.manual_players) == 0:
                raise TypeError(
                    "GUI mode was requested but no player uses the manual policy "
                    "Please use the ManualPolicy for at least one player."
                )
            self._run_gui_play()

    def _run_evaluation(self) -> None:
        observed_perspectives: dict[Player, PayoffPerspective] = {}
        for (player, perspective) in self.specification.game.get_perspectives().items():
            if player in self.specification.evaluation.observed_players:
                if not isinstance(perspective, PayoffPerspective):
                    raise TypeError(
                        f"EVALUATE mode was requested, but the player {player}'s "
                        "perspective does not implemet the get_payoff method. "
                        "Please use a game whose observed players' perspectives "
                        "implement get_payoff(self, state) -> float. "
                    )
                observed_perspectives[player] = perspective

        tracker = MultiResultTracker(observed_perspectives, self.specification.evaluation.buckets)
        game_loop = make_game_loop(
            game = self.specification.game,
            policies = self.specification.policies,
            global_observer = tracker
        )
        game_loop.run(self.specification.evaluation.number_of_tests)
        tracker.print_results()

    def _run_terminal_play(self):
        game_loop = make_game_loop(
            game = self.specification.game,
            policies = self.specification.policies_with_manual_replaced
        )
        game_loop.run()

    def _run_gui_play(self):
        self.specification.gui.set_policies({
            player: policy
            for (player, policy) in self.specification.policies.items()
            if not isinstance(policy, ManualPolicy)
        })
        self.specification.gui.run()


if __name__ == "__main__":
    main()
