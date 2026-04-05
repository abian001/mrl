import os
import shutil
from collections import UserDict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from trueskill import Rating  # type: ignore[import-untyped]

from mrl.alpha_zero.context import EvaluationContext
from mrl.alpha_zero.mcts import MCTSGame
from mrl.alpha_zero.oracle import TrainableOracle
from mrl.game.game import GlobalObserver, Policy, RewardPerspective
from mrl.game.game_loop import make_game_loop


class ModelUpdater:

    def __init__(
        self,
        game: MCTSGame,
        context: EvaluationContext,
        oracle_file_path: Path,
    ):
        self.game = game
        self.context = context
        self.oracle_file_path = oracle_file_path
        self.last_challenger_path = f"{oracle_file_path}_last"
        self.scorebook = Scorebook(
            oracle_file_path,
            context.uncertainty_penalty_coefficient,
        )
        self.rating_system = context.true_skill

    def load_or_initialize_model(
        self,
        model: TrainableOracle,
        resume: bool = True,
    ) -> None:
        if resume and self.oracle_file_path.exists():
            model.load(self.oracle_file_path)
            return
        model.save(self.oracle_file_path)

    def save_if_better(self, model: TrainableOracle) -> bool:
        challenger_path = self.scorebook.add_challenger()
        model.save(challenger_path)
        if len(self.scorebook) > 1:
            self.scorebook = self._evaluate(model, challenger_path)
        best_model_path = self.scorebook.best_model_path
        challenger_model_is_best = best_model_path == challenger_path

        while len(self.scorebook) > self.context.max_old_models:
            worse_model = self.scorebook.pop_worse()
            if worse_model == challenger_path:
                shutil.move(challenger_path, self.last_challenger_path)
            else:
                os.remove(worse_model)

        self.scorebook.save()
        self.oracle_file_path.unlink()
        self.oracle_file_path.symlink_to(Path(best_model_path).name)
        return challenger_model_is_best

    def _evaluate(
        self,
        model: TrainableOracle,
        challenger_path: str,
    ) -> "Scorebook":
        players = self.game.get_players()
        perspectives = self.game.get_perspectives()
        model_paths = list(self.scorebook.keys())
        ratings = [(self.scorebook[model_path],) for model_path in model_paths]

        # Ranking strategy: evaluate all models in the same sampled scenarios,
        # then derive their ranking from the resulting payoffs.
        for _ in range((self.context.episodes // len(self.scorebook)) + 1):
            lead_player = np.random.choice(players)
            for oracle in self.context.oracles:
                opponent_model = np.random.choice(model_paths)
                oracle.load(opponent_model)

            opponent_players = (player for player in players if player != lead_player)
            policies = dict(
                zip(opponent_players, self.context.policies.opponents)
            ) | {
                lead_player: self.context.policies.lead,
            }
            reverse_rewards = []
            for model_path in model_paths:
                model.load(model_path)
                reverse_rewards.append(
                    -self._evaluate_once(perspectives[lead_player], policies)
                )
            ratings = self.rating_system.rate(ratings, ranks = reverse_rewards)

        for (index, model_path) in enumerate(model_paths):
            self.scorebook[model_path] = ratings[index][0]
        model.load(challenger_path)
        return self.scorebook

    def _evaluate_once(
        self,
        lead_perspective: RewardPerspective[Any, Any],
        policies: dict[Any, Policy],
    ) -> float:
        score_observer = ScoreObserver(
            lead_perspective,
            self.context.discount_factor,
        )
        game_loop = make_game_loop(
            game = self.game,
            policies = policies,
            global_observer = score_observer,
        )
        game_loop.run()
        return score_observer.get_reward()


class Scorebook(UserDict[str, Rating]):

    def __init__(
        self,
        oracle_file_path: Path,
        uncertainty_penalty_coefficient: float = 3.0,
    ) -> None:
        self.baseline_root = f"{oracle_file_path}_"
        self.scores_path = f"{oracle_file_path}_scores.yaml"
        self.uncertainty_penalty_coefficient = uncertainty_penalty_coefficient
        super().__init__(self._load())
        self.next_free_index = self._get_next_free_index()

    @property
    def best_model_path(self) -> str:
        return max(
            self.data.items(),
            key = lambda model_rating: self._get_rating_value(model_rating[1]),
        )[0]

    def add_challenger(self) -> str:
        challenger_path = f"{self.baseline_root}{self.next_free_index}"
        self.data[challenger_path] = Rating()
        self.next_free_index = self._get_next_free_index()
        return challenger_path

    def _get_next_free_index(self) -> int:
        indices = []
        for path in self.keys():
            try:
                indices.append(int(path.replace(self.baseline_root, "")))
            except ValueError:
                continue
        return next(index for index in range(len(indices) + 1) if index not in indices)

    def save(self) -> None:
        with open(self.scores_path, "w", encoding = "utf-8") as yaml_file:
            yaml.safe_dump(
                [
                    [model_path, {"mu": rating.mu, "sigma": rating.sigma}]
                    for (model_path, rating) in self.items()
                ],
                yaml_file,
            )

    def pop_worse(self) -> str:
        worse_model, _ = min(
            self.items(),
            key = lambda model_rating: self._get_rating_value(model_rating[1]),
        )
        del self.data[worse_model]
        self.next_free_index = self._get_next_free_index()
        return worse_model

    def _get_rating_value(self, rating: Rating) -> float:
        return rating.mu - self.uncertainty_penalty_coefficient * rating.sigma

    def _load(self) -> dict[str, Rating]:
        if not os.path.exists(self.scores_path):
            return {}

        with open(self.scores_path, "r", encoding = "utf-8") as yaml_file:
            data = yaml.safe_load(yaml_file) or []

        return {
            path: Rating(**rating)
            for [path, rating] in data
            if os.path.exists(path)
        }


class ScoreObserver(GlobalObserver):

    def __init__(
        self,
        perspective: RewardPerspective[Any, Any],
        discount_factor: float = 1.0,
    ):
        self.perspective = perspective
        self.reward = 0.0
        self.discount_factor = discount_factor

    def notify_state(self, state) -> None:
        self.reward = self.perspective.get_reward(state) + self.discount_factor * self.reward

    def get_reward(self) -> float:
        return self.reward
