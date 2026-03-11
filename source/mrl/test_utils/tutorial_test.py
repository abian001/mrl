from pathlib import Path
from dataclasses import dataclass
import os
import random
import subprocess
import shutil
import pytest
from mrl.test_utils.game_runner import main as game_runner
from mrl.test_utils.alpha_zero import main as alpha_zero
from mrl.test_utils.get_examples import main as get_examples


example_directory = Path('test_example_directory').absolute()


@pytest.fixture(scope = "module", autouse = True)
def example_suite():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr('sys.argv', [
        'get_examples',
        str(example_directory), '--overwrite'
    ])
    monkeypatch.syspath_prepend(str(example_directory))
    monkeypatch.setenv(
        "PYTHONPATH",
        str(example_directory) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    get_examples()

    yield

    if example_directory.exists():
        shutil.rmtree(example_directory)
    monkeypatch.undo()


@pytest.fixture
def configuration_file_path(file_name: str) -> str:
    return _get_configuration_file_path(file_name)


def _get_configuration_file_path(file_name: str) -> str:
    file_path = next((
        item for item in example_directory.iterdir()
        if item.name == file_name
    ), None)
    assert file_path is not None
    return str(file_path)


@dataclass
class EvaluationTestData:
    config_file_name: str
    expected_output: str


evaluation_test_data = [
    EvaluationTestData(
        config_file_name = 'tic_tac_toe_auto.yaml',
        expected_output = (
            "\nEvaluation report:\n"
            "Total plays as player O: N. 100\n"
            "Mean Payoff: 0.37\n"
            "Payoff distribution in buckets:\n"
            "(-inf, 0.25): 57 (57%)\n"
            "(0.25, 0.75): 12 (12%)\n"
            "(0.75, inf): 31 (31%)\n"
        )
    ),
    EvaluationTestData(
        config_file_name = 'tic_tac_toe_custom_policy.yaml',
        expected_output = (
            "\nEvaluation report:\n"
            "Total plays as player O: N. 100\n"
            "Mean Payoff: 0.735\n"
            "Payoff distribution in buckets:\n"
            "(-inf, 0.25): 16 (16%)\n"
            "(0.25, 0.75): 21 (21%)\n"
            "(0.75, inf): 63 (63%)\n"
        )
    ),
    EvaluationTestData(
        config_file_name = 'straight_four_auto.yaml',
        expected_output = (
            "\nEvaluation report:\n"
            "Total plays as player O: N. 100\n"
            "Mean Payoff: 0.41\n"
            "Payoff distribution in buckets:\n"
            "(-inf, 0.25): 59 (59%)\n"
            "(0.25, 0.75): 0 (0%)\n"
            "(0.75, inf): 41 (41%)\n"
        )
    ),
    EvaluationTestData(
        config_file_name = 'xiangqi_auto.yaml',
        expected_output = (
            "\nEvaluation report:\n"
            "Total plays as player O: N. 10\n"
            "Mean Payoff: 0.45\n"
            "Payoff distribution in buckets:\n"
            "(-inf, 0.25): 5 (50%)\n"
            "(0.25, 0.75): 1 (10%)\n"
            "(0.75, inf): 4 (40%)\n"
        )
    ),
    EvaluationTestData(
        config_file_name = 'rock_paper_scissors_auto.yaml',
        expected_output = (
            "\nEvaluation report:\n"
            "Total plays as player O: N. 100\n"
            "Mean Payoff: 0.32\n"
            "Payoff distribution in buckets:\n"
            "(-inf, 0.2): 10 (10%)\n"
            "(0.2, 0.4): 40 (40%)\n"
            "(0.4, 0.6): 35 (35%)\n"
            "(0.6, 0.8): 10 (10%)\n"
            "(0.8, inf): 5 (5%)\n"
        )
    ),
    EvaluationTestData(
        config_file_name = 'coordination_auto.yaml',
        expected_output = (
            "\nEvaluation report:\n"
            "Total plays as player O: N. 10\n"
            "Mean Payoff: 0.52\n"
            "Payoff distribution in buckets:\n"
            "(-inf, 0.5): 4 (40%)\n"
            "(0.5, inf): 6 (60%)\n"
        )
    )
]


@pytest.fixture
def file_name(data: EvaluationTestData) -> str:
    return data.config_file_name


@pytest.mark.parametrize('data', evaluation_test_data)
@pytest.mark.quick
def test_evaluation(
    data: EvaluationTestData,
    configuration_file_path: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str]
):
    _run_game(configuration_file_path, monkeypatch, mode = 'evaluate')
    assert str(capsys.readouterr().out) == data.expected_output


@pytest.mark.parametrize('file_name', [
    'tic_tac_toe_manual.yaml',
    'straight_four_manual.yaml',
    'xiangqi_manual.yaml',
    'xiangqi_mcts_manual.yaml',
    'rock_paper_scissors_manual.yaml',
    'coordination_manual.yaml'
])
@pytest.mark.manual
def test_terminal(
    configuration_file_path: str,
    monkeypatch: pytest.MonkeyPatch
):
    _run_game(configuration_file_path, monkeypatch, mode = 'terminal')


@pytest.mark.parametrize('file_name', [
    'tic_tac_toe_manual.yaml',
    'straight_four_manual.yaml',
    'xiangqi_manual.yaml',
    'xiangqi_mcts_manual.yaml',
])
@pytest.mark.gui
def test_gui(configuration_file_path: str):
    subprocess.run(['run_game', configuration_file_path, '--mode', 'gui'], check = True)


def _run_game(configuration_file_path: str, monkeypatch: pytest.MonkeyPatch, mode: str):
    random.seed(42)
    monkeypatch.setattr('sys.argv', [
        'game_runner',
        configuration_file_path,
        '--mode', mode
    ])
    game_runner()


@pytest.fixture
def _clear_workspace():
    yield
    if os.path.exists('test_workspace'):
        shutil.rmtree('test_workspace')


@pytest.mark.parametrize('file_name', [
    'tic_tac_toe_alpha_zero.yaml',
    'straight_four_alpha_zero.yaml',
    'xiangqi_alpha_zero.yaml',
    'centipede_alpha_zero.yaml',
    'tic_tac_toe_alpha_zero_custom_oracle.yaml'
])
@pytest.mark.slow
def test_alpha_zero(
    configuration_file_path: str,
    monkeypatch: pytest.MonkeyPatch,
    _clear_workspace: None
):
    _run_alpha_zero(configuration_file_path, monkeypatch, mode = 'train')


@pytest.mark.parametrize('file_name', [
    'tic_tac_toe_alpha_zero.yaml',
    'straight_four_alpha_zero.yaml',
    'xiangqi_alpha_zero.yaml',
])
@pytest.mark.gui
def test_play_alpha_zero(
    configuration_file_path: str,
    monkeypatch: pytest.MonkeyPatch,
    _clear_workspace: None
):
    _run_alpha_zero(configuration_file_path, monkeypatch, mode = 'train')
    subprocess.run(['run_alpha_zero', configuration_file_path, '--mode', 'gui'], check = True)


@pytest.mark.slow
def test_evaluate_policies_with_trained_oracles(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    _clear_workspace: None
):
    config_path_1 = _get_configuration_file_path('tic_tac_toe_alpha_zero.yaml')
    config_path_2 = _get_configuration_file_path('tic_tac_toe_alpha_zero_auto.yaml')
    _run_alpha_zero(config_path_1, monkeypatch, mode = 'train')
    _run_game(config_path_2, monkeypatch, mode = 'evaluate')
    assert 'Payoff distribution in buckets' in str(capsys.readouterr().out)


def _run_alpha_zero(configuration_file_path: str, monkeypatch: pytest.MonkeyPatch, mode: str):
    random.seed(42)
    monkeypatch.setattr('sys.argv', [
        'alpha_zero',
        configuration_file_path,
        '--mode', mode
    ])
    alpha_zero()
