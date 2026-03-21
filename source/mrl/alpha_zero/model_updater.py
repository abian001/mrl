import copy
import shutil
import numpy as np
from mrl.game.game import GlobalObserver
from mrl.game.game_loop import make_game_loop
from mrl.alpha_zero.mcts import MCTSGame
from mrl.configuration.game_factories import make_policy
from mrl.configuration.alpha_zero_configuration import (
    UnionAlphaZeroConfiguration,
    make_new_oracle_clone
)


class ModelUpdater:

    def __init__(self, game: MCTSGame, config: UnionAlphaZeroConfiguration):
        self.game = game
        self.evaluation = config.evaluation
        self.model_path = config.oracle_file_path
        self.clone_model = make_new_oracle_clone(config)
        self.old_models: list[str] = self._get_old_models()
        self.scores: dict[str, ScoreTracker] = {}

    def save_if_better(self, model):
        is_best, scores, new_scores = self._evaluate(model)
        if is_best:
            old_path = f"{self.model_path}_old_{len(self.old_models)}"
            self.old_models.append(old_path)
            scores[old_path] = new_scores
            self.scores = scores
            shutil.move(self.model_path, old_path)
            model.save(self.model_path)

        while len(self.old_models) > self.evaluation.max_old_models:
            self._remove_oldest_model()
        return is_best

    def _evaluate(self, model):
        scores = copy.deepcopy(self.scores)
        new_scores = ScoreTracker()
        for file_path in self.old_models:
            self.clone_model.load(file_path)
            for _ in range((self.evaluation.episodes // 2) + 1):
                payoff_1 = self._evaluate_once(model, self.clone_model)
                payoff_2 = self._evaluate_once(self.clone_model, model)
                scores[file_path].append(payoff_2)
                new_scores.append(payoff_1)
        new_average_score = new_scores.average_score
        is_best = all(
            scores[old_model].average_score < new_average_score
            for old_model in self.old_models
        )
        return is_best, scores, new_scores

    def _evaluate_once(self, lead, opponent):
        policy = self._make_evaluation_policy(lead)
        opponent = self._make_evaluation_policy(opponent)
        perspectives = self.game.get_perspectives()
        policy_player = np.random.choice(list(perspectives.keys()))
        score_observer = ScoreObserver(perspectives[policy_player])
        game_loop = make_game_loop(
            game = self.game,
            policies = {
                player: policy if player is policy_player else opponent
                for player in perspectives.keys()
            },
            global_observer = score_observer
        )
        game_loop.run()
        return score_observer.get_payoff()

    def _make_evaluation_policy(self, oracle):
        return make_policy(self.evaluation.policy_data | {
            'game': self.game,
            'oracle': oracle
        })

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
        self.old_models = self.old_models[1:]
        for (index, model_path) in enumerate(self.old_models):
            expected_path = f"{self.model_path}_old_{index}"
            shutil.move(model_path, expected_path)
            self.old_models[index] = expected_path
            self.scores[expected_path] = self.scores[model_path]


class ScoreTracker:

    def __init__(self):
        self.average_score = 0.0
        self.count = 0

    def append(self, item):
        self.average_score = (self.average_score * self.count + item) / (self.count + 1)
        self.count += 1


class ScoreObserver(GlobalObserver):

    def __init__(self, perspective):
        self.perspective = perspective
        self.payoff = 0.0

    def notify_state(self, state) -> None:
        if state.is_final:
            self.payoff += self.perspective.get_payoff(state)

    def get_payoff(self):
        return self.payoff
