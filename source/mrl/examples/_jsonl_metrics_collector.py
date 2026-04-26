from pathlib import Path
import random
import json
from mrl.alpha_zero.model_trainer import TrainingMetrics


class JsonlMetricsCollector:

    def __init__(
        self,
        file_path: Path,
        workspace_path: Path | None = None,
        sample_chance: float | None = None
    ):
        if not file_path.is_absolute() and workspace_path is not None:
            file_path = workspace_path / file_path
        self.file_path = file_path
        self.file_path.parent.mkdir(parents = True, exist_ok = True)
        self.sample_chance = sample_chance

    def collect(self, metrics: TrainingMetrics, is_last_batch_in_epoch: bool):
        if not self._is_sample_batch(metrics.batch_index, is_last_batch_in_epoch):
            return
        with open(self.file_path, "a", encoding = "utf-8") as metrics_file:
            metrics_file.write(json.dumps({
                'alpha_zero_epoch': metrics.alpha_zero_epoch_index,
                'training_epoch': metrics.training_epoch_index,
                'batch': metrics.batch_index,
                'policy_loss': metrics.policy_loss,
                'value_loss': metrics.value_loss,
                'total_loss': metrics.total_loss,
                'policy_entropy': metrics.policy_entropy,
            }) + "\n")

    def _is_sample_batch(self, batch_count: int, is_last_batch: bool):
        return is_last_batch or (
            self.sample_chance is not None and
            random.random() <= self.sample_chance
        )
