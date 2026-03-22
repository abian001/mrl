from enum import Enum
from pathlib import Path
import os
import time
import argparse
import yaml
from mrl.game.game_loop import make_game_loop
from mrl.alpha_zero.model_evaluation import EvaluationLoop
from mrl.configuration.alpha_zero_factories import make_alpha_zero
from mrl.configuration.alpha_zero_runner_configuration import (
    make_alpha_zero_runner_specification,
    AlphaZeroRunnerSpecification
)
from mrl.test_utils.exit_handler import try_main


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


def _parse_configuration(path: str) -> AlphaZeroRunnerSpecification:
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
        return make_alpha_zero_runner_specification(config_data)
    except Exception as error:
        raise argparse.ArgumentTypeError(
            'Invalid Alpha Zero configuration in input. '
            f'Parsing failed with error: "{error}"'
        ) from error


class AlphaZeroRunner:

    def __init__(self, config: AlphaZeroRunnerSpecification, mode: TestMode):
        self.config = config
        self.mode: TestMode = mode

    def run(self):
        if self.mode == TestMode.TRAIN:
            assert not os.path.exists(self.config.oracle_file_path), (
                f"{self.config.oracle_file_path} already exists"
            )
        if self.mode == TestMode.OVERWRITE:
            if os.path.exists(self.config.oracle_file_path):
                os.remove(self.config.oracle_file_path)
        if self.mode in (TestMode.TRAIN, TestMode.OVERWRITE, TestMode.RESUME):
            alpha_zero = make_alpha_zero(self.config)
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

    def _run_evaluation(self):
        self.config.oracle.load(self.config.oracle_file_path)
        report_generator = EvaluationLoop(
            self.config.game,
            self.config.oracle,
            self.config.alpha_zero.report_generator
        )
        report_generator()

    def _run_terminal_play(self):
        assert self.config.manual_play is not None
        self.config.oracle.load(self.config.oracle_file_path)
        game_loop = make_game_loop(
            game = self.config.game,
            policies = {
                player: (
                    self.config.stdin_policy
                    if player == self.config.manual_player
                    else self.config.autonomous_policy
                )
                for player in self.config.game.get_players()
            }
        )
        game_loop.run()

    def _run_gui_play(self):
        assert self.config.manual_play is not None
        self.config.oracle.load(self.config.oracle_file_path)
        self.config.gui.set_policies({
            player: self.config.autonomous_policy
            for player in self.config.game.get_players()
            if player != self.config.manual_player
        })
        self.config.gui.run()


if __name__ == "__main__":
    main()
