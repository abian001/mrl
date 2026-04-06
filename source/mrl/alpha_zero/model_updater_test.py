from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest
from trueskill import TrueSkill, Rating  # type: ignore[import-untyped]

from mrl.alpha_zero.context import EvaluationContext, EvaluationPolicies
from mrl.alpha_zero.model_updater import ModelUpdater, Scorebook


@pytest.mark.quick
def test_scorebook_restores_scores_from_yaml(oracle_file_path: Path) -> None:
    old_model_path = Path(f"{oracle_file_path}_0")
    old_model_path.touch()
    scorebook = Scorebook(oracle_file_path)
    scorebook[str(old_model_path)] = Rating(mu = 0.5, sigma = 0.3)
    scorebook.save()

    restored = Scorebook(oracle_file_path)

    assert str(old_model_path) in restored
    assert restored[str(old_model_path)].mu == 0.5
    assert restored[str(old_model_path)].sigma == 0.3


@pytest.fixture
def oracle_file_path(tmp_path: Path) -> Path:
    return tmp_path / "oracle"


@pytest.mark.quick
def test_scorebook_ignores_missing_model_files(oracle_file_path: Path) -> None:
    existing_path = Path(f"{oracle_file_path}_0")
    missing_path = Path(f"{oracle_file_path}_999")
    existing_path.touch()
    scorebook = Scorebook(oracle_file_path)
    scorebook[str(existing_path)] = Rating(mu = 0.5, sigma = 0.3)
    scorebook[str(missing_path)] = Rating(mu = 10.0, sigma = 0.1)
    scorebook.save()

    restored = Scorebook(oracle_file_path)

    assert str(existing_path) in restored
    assert str(missing_path) not in restored


@pytest.mark.quick
def test_scorebook_add_challenger_uses_next_free_index(oracle_file_path: Path) -> None:
    scorebook = Scorebook(oracle_file_path)

    challenger_0 = scorebook.add_challenger()
    challenger_1 = scorebook.add_challenger()

    assert challenger_0 == f"{oracle_file_path}_0"
    assert challenger_1 == f"{oracle_file_path}_1"


@pytest.mark.quick
def test_scorebook_pop_worse_uses_conservative_rating_value(oracle_file_path: Path) -> None:
    scorebook = Scorebook(oracle_file_path)
    best_path = f"{oracle_file_path}_0"
    worse_path = f"{oracle_file_path}_1"
    scorebook[best_path] = Rating(mu = 10.0, sigma = 1.0)
    scorebook[worse_path] = Rating(mu = 10.0, sigma = 2.0)

    removed = scorebook.pop_worse()

    assert removed == worse_path
    assert best_path in scorebook
    assert worse_path not in scorebook


@pytest.mark.quick
def test_model_updater_load_or_initialize_model_loads_saved_model(game: "SimpleGame") -> None:
    model = MockOracle()
    oracle_path = MockPath("oracle", exists = True)
    evaluation_context = EvaluationContext(
        episodes = 1,
        max_models = 2,
        oracles = cast(list[Any], [MockOracle()]),
        policies = EvaluationPolicies(
            lead = cast(Any, FirstActionPolicy()),
            opponents = cast(list[Any], [FirstActionPolicy()]),
        ),
    )
    updater = ModelUpdater(cast(Any, game), evaluation_context, cast(Path, oracle_path))

    updater.load_or_initialize_model(cast(Any, model), resume = True)

    model.load.assert_called_once_with(oracle_path)
    model.save.assert_not_called()


@pytest.fixture
def game() -> "SimpleGame":
    return SimpleGame()


@pytest.mark.quick
def test_model_updater_load_or_initialize_model_saves_model_when_not_resuming(
    game: "SimpleGame",
) -> None:
    model = MockOracle()
    oracle_path = MockPath("oracle", exists = False)
    evaluation_context = EvaluationContext(
        episodes = 1,
        max_models = 2,
        oracles = cast(list[Any], [MockOracle()]),
        policies = EvaluationPolicies(
            lead = cast(Any, FirstActionPolicy()),
            opponents = cast(list[Any], [FirstActionPolicy()]),
        ),
    )
    updater = ModelUpdater(cast(Any, game), evaluation_context, cast(Path, oracle_path))

    updater.load_or_initialize_model(cast(Any, model), resume = False)

    model.save.assert_called_once_with(oracle_path)
    model.load.assert_not_called()


@pytest.mark.quick
def test_model_updater_prunes_old_model_and_updates_best_symlink(
    three_model_context: "TestContext",
) -> None:
    model_is_accepted = three_model_context.updater.save_if_accepted(
        cast(Any, three_model_context.model)
    )

    assert model_is_accepted is True
    assert three_model_context.updater.best_model_was_updated
    three_model_context.model.save.assert_called_once_with("oracle_2")
    three_model_context.remove.assert_called_once_with("oracle_0")
    three_model_context.save_scorebook.assert_called_once_with()
    three_model_context.oracle_path.unlink.assert_called_once_with()
    three_model_context.oracle_path.symlink_to.assert_called_once_with("oracle_1")


@pytest.fixture
def three_model_context(
    game: "SimpleGame",
    monkeypatch: pytest.MonkeyPatch,
) -> "TestContext":
    scenario = TestContextInput(
        existing_model_paths = {
            "oracle_0": Rating(),
            "oracle_1": Rating(),
        },
        rewards_by_path = {
            "oracle_0": 0.0,
            "oracle_1": 1.0,
            "oracle_2": 0.5,
        },
        max_models = 2,
        rated_models = [
            Rating(mu = 1.0, sigma = 0.1),
            Rating(mu = 5.0, sigma = 0.1),
            Rating(mu = 3.0, sigma = 0.1),
        ],
    )
    return TestContext.make(game, scenario, monkeypatch)


@pytest.mark.quick
def test_model_updater_keeps_last_challenger_when_it_is_pruned(
    incumbent_and_challenger_context: "TestContext",
) -> None:
    model_is_accepted = incumbent_and_challenger_context.updater.save_if_accepted(
        cast(Any, incumbent_and_challenger_context.model)
    )

    assert model_is_accepted is False
    assert not incumbent_and_challenger_context.updater.best_model_was_updated
    incumbent_and_challenger_context.move.assert_called_once_with(
        "oracle_0",
        incumbent_and_challenger_context.updater.last_challenger_path,
    )
    incumbent_and_challenger_context.remove.assert_not_called()
    incumbent_and_challenger_context.save_scorebook.assert_called_once_with()
    incumbent_and_challenger_context.oracle_path.unlink.assert_called_once_with()
    incumbent_and_challenger_context.oracle_path.symlink_to.assert_called_once_with("oracle_1")


@pytest.fixture
def incumbent_and_challenger_context(
    game: "SimpleGame",
    monkeypatch: pytest.MonkeyPatch,
) -> "TestContext":
    scenario = TestContextInput(
        existing_model_paths = {
            "oracle_1": Rating(),
        },
        rewards_by_path = {
            "oracle_1": 1.0,
            "oracle_0": 0.0,
        },
        max_models = 1,
        rated_models = [
            Rating(mu = 5.0, sigma = 0.1),
            Rating(mu = 1.0, sigma = 0.1),
        ],
    )
    return TestContext.make(game, scenario, monkeypatch)


Player = str
Action = float


@dataclass
class SimpleState:
    active_player: Player
    lead_reward: float = 0.0
    is_final: bool = False


class SimplePerspective:

    def __init__(self, player: Player) -> None:
        self.player = player

    def get_observation(self, state: SimpleState) -> SimpleState:
        return state

    def get_action_space(self, _state: SimpleState) -> tuple[Action, ...]:
        return (0.0, 0.5, 1.0)

    def get_reward(self, state: SimpleState) -> float:
        if self.player == "lead":
            return state.lead_reward
        return 1.0 - state.lead_reward


class SimpleGame:

    def __init__(self) -> None:
        self.players = ("lead", "opponent")
        self.perspectives = {
            player: SimplePerspective(player)
            for player in self.players
        }

    def make_initial_state(self) -> SimpleState:
        return SimpleState(active_player = "lead")

    def restore(self, observation: SimpleState) -> SimpleState:
        return observation

    def get_players(self) -> tuple[Player, ...]:
        return self.players

    def get_perspectives(self) -> dict[Player, SimplePerspective]:
        return self.perspectives

    def update(self, state: SimpleState, action: Action) -> SimpleState:
        return SimpleState(
            active_player = state.active_player,
            lead_reward = action,
            is_final = True,
        )


class ModelDrivenPolicy:

    def __init__(self, model: "MockOracle", rewards_by_path: dict[str, float]) -> None:
        self.model = model
        self.rewards_by_path = rewards_by_path

    def __call__(
        self,
        _observation: SimpleState,
        action_space: tuple[Action, ...],
    ) -> Action:
        assert self.model.current_path is not None
        reward = self.rewards_by_path[self.model.current_path]
        assert reward in action_space
        return reward


class FirstActionPolicy:

    def __call__(
        self,
        _observation: SimpleState,
        action_space: tuple[Action, ...],
    ) -> Action:
        return action_space[0]


class MockOracle:

    def __init__(self) -> None:
        self.current_path: str | None = None
        self.save = Mock()
        self.load = Mock(side_effect = self._load)

    def _load(self, file_path: str | Path) -> None:
        self.current_path = str(file_path)


class StubRatingSystem(TrueSkill):

    def __init__(self, ratings: list[Rating]) -> None:  # pylint: disable=super-init-not-called
        self.ratings = ratings

    def rate(self, rating_groups: object, ranks: object, *args, **kwargs) -> list[tuple[Rating]]:
        del rating_groups
        del args
        del kwargs
        assert ranks is not None
        return [(rating,) for rating in self.ratings]


class MockPath:

    def __init__(self, path: str, exists: bool = False) -> None:
        self.path = path
        self.exists_value = exists
        self.unlink = Mock()
        self.symlink_to = Mock()

    def exists(self) -> bool:
        return self.exists_value

    def __str__(self) -> str:
        return self.path


@dataclass
class TestContextInput:
    existing_model_paths: dict[str, Rating]
    rewards_by_path: dict[str, float]
    max_models: int
    rated_models: list[Rating]


@dataclass
class TestContext:
    model: MockOracle
    oracle_path: MockPath
    save_scorebook: Mock
    remove: Mock
    move: Mock
    updater: ModelUpdater
    scenario: TestContextInput

    @classmethod
    def make(
        cls,
        game: SimpleGame,
        scenario: TestContextInput,
        monkeypatch: pytest.MonkeyPatch,
    ) -> "TestContext":
        model = MockOracle()
        opponent_oracle = MockOracle()
        oracle_path = MockPath("oracle")
        save_scorebook = Mock()
        remove = Mock()
        move = Mock()
        monkeypatch.setattr(Scorebook, "_load", lambda self: scenario.existing_model_paths)
        monkeypatch.setattr(Scorebook, "save", lambda self: save_scorebook())
        monkeypatch.setattr("mrl.alpha_zero.model_updater.os.remove", remove)
        monkeypatch.setattr("mrl.alpha_zero.model_updater.shutil.move", move)
        evaluation_context = EvaluationContext(
            episodes = 1,
            max_models = scenario.max_models,
            oracles = cast(list[Any], [opponent_oracle]),
            policies = EvaluationPolicies(
                lead = cast(Any, ModelDrivenPolicy(model, scenario.rewards_by_path)),
                opponents = cast(list[Any], [FirstActionPolicy()]),
            ),
            true_skill = StubRatingSystem(scenario.rated_models)
        )
        updater = ModelUpdater(cast(Any, game), evaluation_context, cast(Path, oracle_path))
        return cls(
            model = model,
            oracle_path = oracle_path,
            save_scorebook = save_scorebook,
            remove = remove,
            move = move,
            updater = updater,
            scenario = scenario,
        )
