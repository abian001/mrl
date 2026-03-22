import os
from typing import Callable
import torch
from mrl.alpha_zero.oracle import Oracle
from mrl.alpha_zero.experience_collector import (
    LocalExperienceCollector,
    SharedProcessCollector,
    SingleBufferCollector
)
from mrl.alpha_zero.model_trainer import ModelTrainer
from mrl.alpha_zero.model_evaluation import EvaluationLoop
from mrl.alpha_zero.model_updater import ModelUpdater
from mrl.configuration.alpha_zero_configuration import InMemoryAlphaZeroConfiguration


class AlphaZero:

    def __init__(self, config: InMemoryAlphaZeroConfiguration):
        self.model_path = config.oracle_file_path
        game = config.game

        self.model = config.oracle

        if not isinstance(self.model, torch.nn.Module):
            raise TypeError(
                f"Oracle class {type(self.model)} is not a torch Module."
            )
        self.model_trainer = ModelTrainer(self.model, config.trainer)
        self.model_updater = ModelUpdater(config.game, config)
        self.number_of_epochs = config.number_of_epochs

        self.report_generator: Callable[[], None]
        if config.report_generator is None:
            self.report_generator = lambda: None
        else:
            self.report_generator = EvaluationLoop(game, self.model, config.report_generator)

        self.collector: LocalExperienceCollector
        assert isinstance(self.model, Oracle)
        if config.collector.number_of_processes > 1:
            self.collector = SharedProcessCollector(game, self.model, config.collector)
        else:
            self.collector = SingleBufferCollector(game, self.model, config.collector)

    def train(self, resume: bool = True):
        if resume and os.path.exists(self.model_path):
            self.model.load(self.model_path)
        else:
            self.model.save(self.model_path)
        for _ in range(self.number_of_epochs):
            buffer, buffer_ready = self.collector.collect()
            if buffer_ready:
                self.model_trainer.train(buffer)
                better_model = self.model_updater.save_if_better(self.model)
                if better_model:
                    self.report_generator()

    def save_training_data(self, file_path: str):
        self.collector.save_training_data(file_path)

    def load_training_data(self, file_path: str):
        self.collector.load_training_data(file_path)
