from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Protocol, runtime_checkable, Container, Mapping, Sequence

State = TypeVar("State")
Action = TypeVar("Action")
Player = TypeVar("Player")
Observation = TypeVar("Observation")
ActionSpace = TypeVar("ActionSpace", bound = Container)
DiscreteActionSpace = TypeVar("DiscreteActionSpace", bound = Sequence)

PlayerCo = TypeVar("PlayerCo", covariant = True)
StateCo = TypeVar("StateCo", covariant = True)
ActionCo = TypeVar("ActionCo", covariant = True)
ObservationCo = TypeVar("ObservationCo", covariant = True)
ActionSpaceCo = TypeVar("ActionSpaceCo", bound = Container, covariant = True)

StateContra = TypeVar("StateContra", contravariant = True)
ObservationContra = TypeVar("ObservationContra", contravariant = True)
ActionContra = TypeVar("ActionContra", contravariant = True)
ActionSpaceContra = TypeVar("ActionSpaceContra", bound = Container, contravariant = True)


class InvalidGame(Exception):
    """Exception raised when the game definition is invalid"""


class StopEvent(Exception):
    """An exception raised when we want to stop a game in progress"""


class Perspective(Protocol[StateContra, ObservationCo, ActionSpaceCo]):
    """The state of the game as observed by one player"""

    @abstractmethod
    def get_observation(self, state: StateContra) -> ObservationCo:
        """Returns an observation made by a player under the perspective"""

    @abstractmethod
    def get_action_space(self, state: StateContra) -> ActionSpaceCo:
        """Returns the space of actions the player is allowed to make in the state"""


@runtime_checkable
class Game(Protocol[StateCo, Player]):
    """The game handles the state initialisation, player specifications
       and state update rules.
       For the update rules you need to implement either the TurnBased
       or the DiscreteTime protocols below.
    """

    @abstractmethod
    def make_initial_state(self) -> StateCo:
        """Create the initial state of the game"""

    @abstractmethod
    def get_players(self) -> tuple[Player, ...]:
        """Returns the player identifiers.
           If this a DiscreteTime game, order is important.
           All players actions are provided in this order in the
           update method. See DiscreteTime protocol below.
        """

    @abstractmethod
    def get_perspectives(self) -> Mapping[Player, Perspective]:
        """Return the player perspectives."""


@runtime_checkable
class TurnBased(Protocol[State, ActionContra]):
    """This is a protocol for a game."""

    @abstractmethod
    def update(self, state: State, action: ActionContra) -> State:
        """Updates the game state according to the action of the active player"""


@runtime_checkable
class DiscreteTime(Protocol[State, ActionContra]):
    """This is a protocol for a game."""

    @abstractmethod
    def update(self, state: State, actions: tuple[ActionContra, ...]) -> State:
        """Updates the game state according to the action of all players"""


@runtime_checkable
class FinalCheckable(Protocol):
    """This is a protocol for a game state.
       Every game state needs to implement this protocol.
    """

    @property
    @abstractmethod
    def is_final(self) -> bool:
        """Tells whether the state is final."""


@runtime_checkable
class HasActivePlayer(Protocol[PlayerCo]):
    """This is a protocol for a game state.
       A TurnBased game state needs to implement this protocol
    """

    @property
    @abstractmethod
    def active_player(self) -> PlayerCo:
        """Tells the player that is active in the state."""


@runtime_checkable
class Restorable(Protocol[StateCo, ObservationContra]):
    """This is a protocol for a game."""

    @abstractmethod
    def restore(self, observation: ObservationContra) -> StateCo:
        """Returns a state that is compatible with the given observation"""


@runtime_checkable
class HasActionSpaceDimension(Protocol):
    """This is a protocol for a perspective."""

    @property
    @abstractmethod
    def action_space_dimension(self) -> int:
        """Returns the number of distinct actions allowed in the action space."""


Reward = float


@runtime_checkable
class RewardPerspective(Perspective, Protocol[StateContra, ObservationCo]):
    """Protocol for a game perspective."""

    @abstractmethod
    def get_reward(self, state: StateContra) -> Reward:
        """Returns observed reward in the given state"""


class RewardObservable(Protocol[StateContra, Player, ObservationCo]):
    """Protocol for a game"""

    @abstractmethod
    def get_perspectives(self) -> Mapping[Player, RewardPerspective[StateContra, ObservationCo]]:
        """Return the player payoff perspectives."""


class Observer(ABC, Generic[Observation, Action]):
    """Observes the game from the perspective of one player only.
       It collects observation as the game was a one-player game,
       it ignore actions from all other players and observes
       the final state change.
    """

    def notify(self, observation: Observation, action: Action) -> None:
        """Act on the received notification: this is the observation
           and the action taken as a result of the observation.
        """

    def notify_final(self, observation: Observation) -> None:
        """Act on the received final notification of the game"""

    def notify_stop_event(self, stop_event: StopEvent) -> None:
        """Act on the received stop event"""


class GlobalObserver(ABC, Generic[Player, State, Action]):
    """Observes the full state of the game and all actions of all players"""

    def notify_state(self, state: State) -> None:
        """Act on received state notification"""

    def notify_actions(self, actions: tuple[Action, ...]) -> None:
        """Act on received action notification, for games in which players
           act simultaneously.
        """

    def notify_action(self, player: Player, action: Action) -> None:
        """Act on received action notification, for turn-based games."""

    def notify_stop_event(self, stop_event: StopEvent) -> None:
        """Act on the received stop event"""


@runtime_checkable
class Policy(Protocol[ObservationContra, ActionCo, ActionSpaceContra]):
    """Determines the actions to be taken given the input observation"""

    @abstractmethod
    def __call__(self, observation: ObservationContra, action_space: ActionSpaceContra) -> ActionCo:
        """Chose an action to perform on the game"""


class InteractivePolicy(
    ABC,
    Policy[Observation, Action, ActionSpaceContra],
    Generic[Observation, Action, ActionSpaceContra, Player]
):
    """A user supplies the value to the __call__function.
       The notification messages should be used to construct a view for the user.
    """

    def notify_observation(self, observation: Observation) -> None:
        """Act on the received observation"""

    def notify_actions(self, actions: tuple[Action, ...]) -> None:
        """Act on the received actions, for games in which players
           act simultaneously.
        """

    def notify_action(self, player: Player, action: Action) -> None:
        """Act on the received action, for turn-based games."""

    def notify_stop_event(self, stop_event: StopEvent) -> None:
        """Act on the received stop event"""
