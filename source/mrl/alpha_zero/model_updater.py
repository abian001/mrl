import copy
import os
import shutil
from collections import UserDict
import yaml
import numpy as np
from mrl.alpha_zero.context import UnionAlphaZeroContext
from mrl.game.game import GlobalObserver
from mrl.game.game_loop import make_game_loop
from mrl.alpha_zero.mcts import MCTSGame


class ModelUpdater:

    def __init__(self, game: MCTSGame, context: UnionAlphaZeroContext):
        self.game = game
        self.evaluation = context.evaluation
        self.model_path = context.oracle_file_path
        self.scores_path = f"{self.model_path}_scores.yaml"
        self.old_models: list[str] = self._get_old_models()
        self.scores = ScoreTracker(self.scores_path, self.old_models)

    def save_if_better(self, model):
        is_best, scores, new_scores = self._evaluate()
        if is_best:
            old_path = f"{self.model_path}_old_{len(self.old_models)}"
            self.old_models.append(old_path)
            scores[old_path] = new_scores
            self.scores = scores
            shutil.move(self.model_path, old_path)
            model.save(self.model_path)

        while len(self.old_models) > self.evaluation.max_old_models:
            self._remove_oldest_model()
        self.scores.save(self.old_models)
        return is_best

    def _evaluate(self):
        scores = copy.deepcopy(self.scores)
        new_scores = AverageScore()
        for file_path in self.old_models:
            self.evaluation.oracle.load(file_path)
            for _ in range((self.evaluation.episodes // 2) + 1):
                payoff_1 = self._evaluate_once(
                    self.evaluation.policies.lead,
                    self.evaluation.policies.opponent,
                )
                payoff_2 = self._evaluate_once(
                    self.evaluation.policies.opponent,
                    self.evaluation.policies.lead,
                )
                scores[file_path].append(payoff_2)
                new_scores.append(payoff_1)
        new_average_score = new_scores.average_score
        is_best = all(
            scores[old_model].average_score < new_average_score
            for old_model in self.old_models
        )
        return is_best, scores, new_scores

    def _evaluate_once(self, lead_policy, opponent_policy):
        perspectives = self.game.get_perspectives()
        policy_player = np.random.choice(list(perspectives.keys()))
        score_observer = ScoreObserver(perspectives[policy_player])
        game_loop = make_game_loop(
            game = self.game,
            policies = {
                player: lead_policy if player is policy_player else opponent_policy
                for player in perspectives.keys()
            },
            global_observer = score_observer
        )
        game_loop.run()
        return score_observer.get_payoff()

    def _get_old_models(self) -> list[str]:
        if not self.model_path.parent.exists():
            return []

        def get_index(name: str) -> int:
            try:
                return int(name.split("_")[-1])
            except (TypeError, IndexError):
                return 0

        old_models = sorted([
            str(x) for x in self.model_path.parent.iterdir()
            if x.is_file() and x.name.startswith(f"{self.model_path.name}_old_")
        ], key = get_index)

        for (index, old_model) in enumerate(old_models):
            expected_path = f"{self.model_path}_old_{index}"
            if old_model != expected_path:
                print(f"The old model {old_model} was renamed {expected_path}.")
                shutil.move(old_model, expected_path)
            old_models[index] = expected_path

        return old_models

    def _remove_oldest_model(self):
        removed_model = self.old_models.pop(0)
        self.scores.pop(removed_model, None)
        for (index, model_path) in enumerate(self.old_models):
            expected_path = f"{self.model_path}_old_{index}"
            if model_path != expected_path:
                shutil.move(model_path, expected_path)
                self.scores[expected_path] = self.scores.pop(model_path)
                self.old_models[index] = expected_path


class ScoreTracker(UserDict[str, "AverageScore"]):

    def __init__(self, file_path: str, model_paths: list[str]):
        self.file_path = file_path
        super().__init__(self._load(model_paths))

    def save(self, model_paths: list[str]):
        self.data = {
            model_path: self.get(model_path, AverageScore())
            for model_path in model_paths
        }
        with open(self.file_path, "w", encoding = "utf-8") as yaml_file:
            yaml.safe_dump(
                {
                    model_path: score.to_data()
                    for (model_path, score) in self.items()
                },
                yaml_file
            )

    def _load(self, model_paths: list[str]) -> dict[str, "AverageScore"]:
        if not os.path.exists(self.file_path):
            return {
                model_path: AverageScore()
                for model_path in model_paths
            }

        with open(self.file_path, "r", encoding = "utf-8") as yaml_file:
            data = yaml.safe_load(yaml_file) or {}

        return {
            model_path: AverageScore.from_data(data.get(model_path, {}))
            for model_path in model_paths
        }


class AverageScore:

    def __init__(self):
        self.average_score = 0.0
        self.count = 0

    def append(self, item):
        self.average_score = (self.average_score * self.count + item) / (self.count + 1)
        self.count += 1

    def to_data(self) -> dict[str, float | int]:
        return {
            "average_score": self.average_score,
            "count": self.count
        }

    @classmethod
    def from_data(cls, data: dict) -> "AverageScore":
        tracker = cls()
        tracker.average_score = float(data.get("average_score", 0.0))
        tracker.count = int(data.get("count", 0))
        return tracker


class ScoreObserver(GlobalObserver):

    def __init__(self, perspective):
        self.perspective = perspective
        self.payoff = 0.0

    def notify_state(self, state) -> None:
        if state.is_final:
            self.payoff += self.perspective.get_reward(state)

    def get_payoff(self):
        return self.payoff
