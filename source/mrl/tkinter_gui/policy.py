import threading
from typing import Generic
from mrl.game.game import StopEvent, Observation, Action, ActionSpace


class _StopAction:
    pass


_stop_action = _StopAction()


class TkinterPolicy(Generic[Observation, Action, ActionSpace]):
    """A policy to be used with the Tkinter GUI framework"""

    def __init__(self) -> None:
        self.action: Action | _StopAction | None = None
        self.action_condition = threading.Condition()

    def __call__(self, observation: Observation, action_space: ActionSpace) -> Action:
        action = self._wait_for_action()
        if action is _stop_action:
            raise StopEvent()
        assert not isinstance(action, _StopAction)
        return action

    def send_stop_action(self):
        self.send_action(_stop_action)

    def send_action(self, action: Action | _StopAction) -> None:
        self.action = action
        with self.action_condition:
            self.action_condition.notify()

    def _wait_for_action(self) -> Action | _StopAction:
        with self.action_condition:
            while self.action is None:
                self.action_condition.wait()
            action = self.action
            self.action = None
        return action
