from typing import Generic, Any
import random as python_random
from mrl.game.game import Action, ActionSpace, DiscreteActionSpace


class RandomPolicy(Generic[Action, DiscreteActionSpace]):

    def __call__(
        self,
        observation: Any,
        action_space: DiscreteActionSpace
    ) -> Action:
        return python_random.choice(action_space)


class FirstChoicePolicy(Generic[Action, DiscreteActionSpace]):

    def __call__(
        self,
        observation: Any,
        action_space: DiscreteActionSpace
    ) -> Action:
        return action_space[0]


class ManualPolicy(Generic[Action, ActionSpace]):
    """Marker class that tells the runner to use manual inputs as a policy"""

    def __call__(self, observation: Any, action_space: ActionSpace) -> Action:
        raise TypeError("An attempt to call the marker manual policy was made.")
