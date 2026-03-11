from typing import Generic, Mapping
from mrl.game.game import (
    GlobalObserver,
    Player,
    PayoffPerspective
)


class MultiResultTracker(GlobalObserver, Generic[Player]):

    def __init__(
        self,
        perspectives: Mapping[Player, PayoffPerspective],
        buckets: tuple[tuple[float, float], ...]
    ):
        self.trackers = {
            ResultTracker(player, perspective, buckets)
            for (player, perspective) in perspectives.items()
        }

    def clear(self):
        for tracker in self.trackers:
            tracker.clear()

    def notify_state(self, state) -> None:
        for tracker in self.trackers:
            tracker.notify_state(state)

    def print_results(self):
        for tracker in self.trackers:
            tracker.print_results()


class ResultTracker(GlobalObserver, Generic[Player]):

    def __init__(
        self,
        model_led_player: Player,
        model_led_perspective: PayoffPerspective,
        buckets: tuple[tuple[float, float], ...] | None = None
    ):
        self.buckets = buckets or tuple()
        self.payoff = 0.0
        self.statistics = [0] * len(self.buckets)
        self.total = 0
        self.model_led_player = model_led_player
        self.model_led_perspective = model_led_perspective

    @property
    def average_payoff(self):
        return 0.0 if self.total <= 1e-7 else self.payoff / self.total

    def clear(self):
        self.payoff = 0
        self.statistics = [0] * len(self.buckets)
        self.total = 0

    def notify_state(self, state) -> None:
        if not state.is_final:
            return
        self.total += 1
        payoff = self.model_led_perspective.get_payoff(state)
        self.payoff += payoff
        bucket_index = next(
            i for (i, bucket) in enumerate (self.buckets)
            if bucket[0] <= payoff < bucket[1]
        )
        self.statistics[bucket_index] += 1

    def print_results(self):
        print("\nEvaluation report:")
        print(f"Total plays as player {self.model_led_player}: N. {self.total}")
        print(f"Mean Payoff: {self.average_payoff:.3g}")
        print("Payoff distribution in buckets:")
        print("\n".join(
            f"{bucket}: {self.statistics[index]} ({100 * self.statistics[index] / self.total:.3g}%)"
            for (index, bucket) in enumerate(self.buckets)
        ))
