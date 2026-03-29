import argparse
import os
import time
from enum import Enum
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import yaml

from mrl.alpha_zero.context import AlphaZeroContext
from mrl.alpha_zero.model_evaluation import EvaluationLoop
from mrl.configuration.alpha_zero_runner_configuration import AlphaZeroRunnerConfiguration
from mrl.configuration.alpha_zero_runner_factory import AlphaZeroRunnerFactory
from mrl.game.game_loop import make_game_loop
from mrl.test_utils.error_handler import try_main, describe_exception


def main():
    try_main(_run)


def _run():
    arguments = _make_arguments()
    runner = AlphaZeroRunner(arguments.config, arguments.mode)
    runner.run()


def _make_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config',
        type = _parse_configuration,
        help = "Path to the alpha zero runner configuration."
    )
    parser.add_argument(
        '--mode',
        type = _parse_test_mode,
        choices = list(TestMode),
        required = True,
        help = (
            "Mode of execution:\n"
            "(train) train a model protect against overwrite,\n"
            "(overwrite) train a model and overwrite as necessary,\n"
            "(resume) resume training an existing model,\n"
            "(evaluate) evaluate model against a random policy,\n"
            "(terminal) play against a trained model in terminal,\n"
            "(gui) play against a trained model in gui."
        )
    )
    return parser.parse_args()


class TestMode(Enum):
    TRAIN = 'train'  # train from zero, error if model file exists
    OVERWRITE = 'overwrite' # train from zero, overwrite previous model
    RESUME = 'resume'  # train from saved state
    EVALUATE = 'evaluate'  # evaluate saved model
    TERMINAL = 'terminal'  # play against saved model in terminal
    GUI = 'gui'  # play against saved model in GUI

    def __str__(self):
        return self.value


def _parse_test_mode(value: str) -> TestMode:
    try:
        return TestMode(value.lower())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f'--mode must be one of {[str(x) for x in list(TestMode)]}'
        ) from error


def _parse_configuration(path: str) -> AlphaZeroRunnerConfiguration:
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

    if 'config_file_path' in config_data:
        config_file_path = Path(config_data['config_file_path'])
        if not config_file_path.exists() or not config_file_path.samefile(Path(path)):
            raise argparse.ArgumentTypeError(
                f"There is a Mismatch between the input file path {path}, "
                f"and the file path mentioned in the configuration {config_file_path}. "
                "Please, ensure they are the same, or do not include config_file_path "
                "in the configuration file."
            )

    config_data['config_file_path'] = path
    try:
        return AlphaZeroRunnerConfiguration.model_validate(config_data)
    except Exception as error:
        raise argparse.ArgumentTypeError(
            'Invalid Alpha Zero configuration in input. '
            f'Parsing failed with {describe_exception(error)}'
        ) from error


@runtime_checkable
class Loadable(Protocol):

    def load(self, file_path) -> None:
        """Load model state from disk."""


class AlphaZeroRunner:  # pylint: disable=too-many-instance-attributes

    def __init__(self, config: AlphaZeroRunnerConfiguration, mode: TestMode):
        self.config = config
        self.mode: TestMode = mode
        self.factory = AlphaZeroRunnerFactory()
        self.alpha_zero_context: AlphaZeroContext = self.factory.make_context(
            self.config.alpha_zero
        )
        self._manual_player = None
        self._autonomous_policy = None
        self._stdin_policy = None
        self._gui = None

    @property
    def manual_player(self):
        if self._manual_player is None:
            self._manual_player = self.factory.make_manual_player(
                self.config,
                self.alpha_zero_context,
            )
        return self._manual_player

    @property
    def autonomous_policy(self):
        if self._autonomous_policy is None:
            self._autonomous_policy = self.factory.make_autonomous_policy(
                self.config,
                self.alpha_zero_context,
            )
        return self._autonomous_policy

    @property
    def stdin_policy(self):
        if self._stdin_policy is None:
            self._stdin_policy = self.factory.make_stdin_policy_for_runner(
                self.config,
                self.alpha_zero_context,
                self.manual_player,
            )
        return self._stdin_policy

    @property
    def gui(self):
        if self._gui is None:
            self._gui = self.factory.make_manual_gui(
                self.config,
                self.alpha_zero_context,
            )
        return self._gui

    def run(self):
        if self.mode == TestMode.TRAIN:
            if os.path.exists(self.alpha_zero_context.oracle_file_path):
                raise FileExistsError(
                    f"Model file {self.alpha_zero_context.oracle_file_path} already exists. "
                    "Use --mode overwrite to replace it or --mode resume to continue training."
                )
        if self.mode == TestMode.OVERWRITE:
            if os.path.exists(self.alpha_zero_context.oracle_file_path):
                os.remove(self.alpha_zero_context.oracle_file_path)
        if self.mode in (TestMode.TRAIN, TestMode.OVERWRITE, TestMode.RESUME):
            alpha_zero = self.factory.make_alpha_zero(self.alpha_zero_context)
            self._run_train(alpha_zero, resume = self.mode == TestMode.RESUME)

        elif self.mode == TestMode.EVALUATE:
            self._run_evaluation()
        elif self.mode == TestMode.TERMINAL:
            self._run_terminal_play()
        elif self.mode == TestMode.GUI:
            self._run_gui_play()

    def _run_train(self, alpha_zero, resume: bool = False):
        start_time = time.perf_counter()
        alpha_zero.train(resume = resume)
        end_time = time.perf_counter()
        print(f"Training took {end_time - start_time} seconds")

    def _run_evaluation(self) -> None:
        oracle: Loadable = cast(Loadable, self.alpha_zero_context.oracle)
        oracle.load(self.alpha_zero_context.oracle_file_path)  # pylint: disable=no-member
        report_generator = EvaluationLoop(
            self.alpha_zero_context.game,
            oracle,
            self.alpha_zero_context.report_generator
        )
        report_generator()

    def _run_terminal_play(self) -> None:
        oracle: Loadable = cast(Loadable, self.alpha_zero_context.oracle)
        oracle.load(self.alpha_zero_context.oracle_file_path)  # pylint: disable=no-member
        game_loop = make_game_loop(
            game = self.alpha_zero_context.game,
            policies = {
                player: (
                    self.stdin_policy
                    if player == self.manual_player
                    else self.autonomous_policy
                )
                for player in self.alpha_zero_context.game.get_players()
            }
        )
        game_loop.run()

    def _run_gui_play(self) -> None:
        oracle: Loadable = cast(Loadable, self.alpha_zero_context.oracle)
        oracle.load(self.alpha_zero_context.oracle_file_path)  # pylint: disable=no-member
        if self.gui is None:
            raise TypeError("GUI mode was requested but no GUI is available.")
        self.gui.set_policies({
            player: self.autonomous_policy
            for player in self.alpha_zero_context.game.get_players()
            if player != self.manual_player
        })
        self.gui.run()


if __name__ == "__main__":
    main()
