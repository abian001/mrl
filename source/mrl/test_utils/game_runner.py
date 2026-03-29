from enum import Enum
import os
from typing import Generic
import argparse
import yaml
from mrl.game.game import Player, Policy
from mrl.game.game_loop import make_game_loop
from mrl.game.trackers import MultiResultTracker
from mrl.configuration.game_runner_configuration import GameRunnerConfiguration
from mrl.configuration.game_runner_factory import GameRunnerFactory
from mrl.test_utils.policies import ManualPolicy
from mrl.test_utils.error_handler import try_main, describe_exception
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


def _parse_configuration(path: str) -> GameRunnerConfiguration:
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(
            f'Input file {path} does not exist.'
        )

    try:
        with open(path, 'r', encoding = 'utf-8') as yaml_file:
            config_data = yaml.safe_load(yaml_file)
    except Exception as error:
        raise argparse.ArgumentTypeError(
            f'Failed to access the file {path}. {describe_exception(error)}'
        ) from error

    if not isinstance(config_data, dict):
        raise argparse.ArgumentTypeError(
            f'Configuration file {path} must contain a YAML mapping at the top level.'
        )

    try:
        return GameRunnerConfiguration.model_validate(config_data)
    except Exception as error:
        raise argparse.ArgumentTypeError(
            'Invalid game runner configuration in input. '
            f'Parsing failed with {describe_exception(error)}'
        ) from error


class GameRunner(Generic[Player]):  # pylint: disable=too-many-instance-attributes

    def __init__(self, configuration: GameRunnerConfiguration, mode: TestMode):
        self.configuration = configuration
        self.mode: TestMode = mode
        factory = GameRunnerFactory()
        self.game = factory.make_runner_game(self.configuration)
        self.policies = factory.make_policies(self.configuration, self.game)
        self._factory = factory
        self._terminal_policies: dict[Player, Policy] | None = None
        self._gui = None
        self._observed_players: tuple[Player, ...] | None = None
        self.manual_players = tuple((
            player for (player, policy) in self.policies.items()
            if isinstance(policy, ManualPolicy)
        ))

    @property
    def terminal_policies(self) -> dict[Player, Policy]:
        if self._terminal_policies is None:
            self._terminal_policies = self._factory.make_terminal_policies(
                self.configuration,
                self.policies,
            )
        return self._terminal_policies

    @property
    def gui(self):
        if self._gui is None:
            self._gui = self._factory.make_runner_gui(self.configuration)
        return self._gui

    @property
    def observed_players(self) -> tuple[Player, ...]:
        if self._observed_players is None:
            self._observed_players = self._factory.make_observed_players(
                self.configuration,
                self.game,
            )
        return self._observed_players

    @property
    def number_of_tests(self) -> int:
        return self.configuration.evaluation_config.number_of_tests

    @property
    def buckets(self) -> tuple[tuple[float, float], ...]:
        return self.configuration.evaluation_config.buckets

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
        for (player, perspective) in self.game.get_perspectives().items():
            if player in self.observed_players:
                if not isinstance(perspective, PayoffPerspective):
                    raise TypeError(
                        f"EVALUATE mode was requested, but the player {player}'s "
                        "perspective does not implemet the get_payoff method. "
                        "Please use a game whose observed players' perspectives "
                        "implement get_payoff(self, state) -> float. "
                    )
                observed_perspectives[player] = perspective

        tracker = MultiResultTracker(observed_perspectives, self.buckets)
        game_loop = make_game_loop(
            game = self.game,
            policies = self.policies,
            global_observer = tracker
        )
        game_loop.run(self.number_of_tests)
        tracker.print_results()

    def _run_terminal_play(self):
        game_loop = make_game_loop(
            game = self.game,
            policies = self.terminal_policies
        )
        game_loop.run()

    def _run_gui_play(self):
        if self.gui is None:
            raise TypeError("GUI mode was requested but no GUI is available.")
        self.gui.set_policies({
            player: policy
            for (player, policy) in self.policies.items()
            if not isinstance(policy, ManualPolicy)
        })
        self.gui.run()


if __name__ == "__main__":
    main()
