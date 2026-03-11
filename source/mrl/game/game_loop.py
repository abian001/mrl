from abc import ABC, abstractmethod
from typing import Mapping, Generic, Protocol, Any, runtime_checkable, TypeVar
import inspect
from mrl.game.game import (
    Game,
    Perspective,
    Player,
    Policy,
    Observer,
    GlobalObserver,
    State,
    StopEvent,
    Action,
    FinalCheckable,
    HasActivePlayer,
    TurnBased,
    DiscreteTime,
    InvalidGame,
    ActionContra,
    PlayerCo,
    InteractivePolicy
)
from mrl.game.protocols import ProtocolChecker


class GameLoop(ABC, Generic[State]):

    @abstractmethod
    def run(self, number_of_times: int = 1) -> None:
        """Runs a game a give number of times"""

    @abstractmethod
    def run_once_from_state(self, state: State) -> None:
        """Run the game loop once starting from the input state"""


def make_game_loop(
    game: Game,
    policies: Mapping[Player, Policy],
    observers: Mapping[Player, Observer] | None = None,
    global_observer: GlobalObserver | None = None
) -> GameLoop:
    if any(isinstance(policy, InteractivePolicy) for policy in policies.values()):
        return _make_interactive_game_loop(game, policies, observers, global_observer)
    return _make_non_interactive_game_loop(game, policies, observers, global_observer)


def _make_non_interactive_game_loop(
    game: Game,
    policies: Mapping[Player, Policy],
    observers: Mapping[Player, Observer] | None = None,
    global_observer: GlobalObserver | None = None
) -> GameLoop:
    if ProtocolChecker.isinstance(game, TurnBased):
        _assert_state_supports_turn_based(game)
        return TurnBasedGameLoop(game, policies, observers, global_observer)
    if ProtocolChecker.isinstance(game, DiscreteTime):
        _assert_state_support_discrete_time(game)
        return DiscreteTimeGameLoop(game, policies, observers, global_observer)
    raise InvalidGame("Game is neither turn based nor discrete time")


def _make_interactive_game_loop(
    game: Game,
    policies: Mapping[Player, Policy],
    observers: Mapping[Player, Observer] | None = None,
    global_observer: GlobalObserver | None = None
) -> GameLoop:
    interactive_policies: dict[Player, InteractivePolicy] = {
        p: x for (p, x) in policies.items()
        if isinstance(x, InteractivePolicy)
    }
    dispatcher: NotificationDispatcher = NotificationDispatcher(
        global_observer = global_observer or GlobalObserver(),
        policies = interactive_policies,
        perspectives = game.get_perspectives()
    )
    return _make_non_interactive_game_loop(game, policies, observers, dispatcher)


class _GameLoop(GameLoop[State], Generic[State, Player]):

    def __init__(
        self,
        game: Game[State, Player],
        policies: Mapping[Player, Policy],
        observers: Mapping[Player, Observer] | None = None,
        global_observer: GlobalObserver | None = None
    ):
        self.game = game
        observers = observers or {}
        null_observer: Observer = Observer()
        self.actors : dict[Player, _Actor] = {
            player: _Actor(
                perspective,
                policies[player],
                observers.get(player, null_observer)
            )
            for (player, perspective) in game.get_perspectives().items()
        }
        self.global_observer = global_observer or GlobalObserver()

    def run(self, number_of_times: int = 1) -> None:
        try:
            for _ in range(number_of_times):
                state = self.game.make_initial_state()
                self.run_once_from_state(state)
        except StopEvent as event:
            self._notify_stop_event(event)

    def _notify_final_state(self, state: State) -> None:
        self.global_observer.notify_state(state)
        for actor in self.actors.values():
            actor.notify_final_state(state)

    def _notify_stop_event(self, event: StopEvent) -> None:
        self.global_observer.notify_stop_event(event)
        for actor in self.actors.values():
            actor.notify_stop_event(event)


class _Actor(Generic[State, Action]):

    def __init__(self, perspective: Perspective, policy: Policy, observer: Observer):
        self.perspective = perspective
        self.policy = policy
        self.observer = observer

    def choose_action(self, state: State) -> Action:
        observation = self.perspective.get_observation(state)
        action_space = self.perspective.get_action_space(state)
        action = self.policy(observation, action_space)
        self.observer.notify(observation, action)
        return action

    def notify_final_state(self, state: State) -> None:
        self.observer.notify_final(
            observation = self.perspective.get_observation(state),
        )

    def notify_stop_event(self, event: StopEvent) -> None:
        self.observer.notify_stop_event(event)


class TurnBasedGame(
   Game[State, Player],
   TurnBased[State, ActionContra],
   Protocol[State, Player, ActionContra]
):
    pass


class TurnBasedState(FinalCheckable, HasActivePlayer[PlayerCo], Protocol[PlayerCo]):
    pass


class TurnBasedGameLoop(_GameLoop):

    game: TurnBasedGame

    def run_once_from_state(self, state: TurnBasedState) -> None:
        while not state.is_final:
            self.global_observer.notify_state(state)
            active_actor = self.actors[state.active_player]
            action = active_actor.choose_action(state)
            self.global_observer.notify_action(state.active_player, action)
            state = self.game.update(state, action)
        self._notify_final_state(state)


class DiscreteTimeGame(
    Game[State, Player],
    DiscreteTime[State, ActionContra],
    Protocol[State, Player, ActionContra]
):
    pass


class DiscreteTimeGameLoop(_GameLoop):

    game: DiscreteTimeGame

    def run_once_from_state(self, state: FinalCheckable) -> None:
        while not state.is_final:
            self.global_observer.notify_state(state)
            actions = tuple(
                player.choose_action(state)
                for player in self.actors.values()
            )
            self.global_observer.notify_actions(actions)
            state = self.game.update(state, actions)
        self._notify_final_state(state)


def _assert_state_supports_turn_based(game: Game) -> None:
    _assert_state_implements_protocols(game, (FinalCheckable, HasActivePlayer))


def _assert_state_support_discrete_time(game: Game) -> None:
    _assert_state_implements_protocols(game, (FinalCheckable,))


def _assert_state_implements_protocols(game: Game, protocols: tuple[type, ...]) -> None:
    state_annotation = inspect.signature(game.make_initial_state).return_annotation
    if (
        state_annotation is not inspect.Signature.empty and
        _class_implements_protocols(state_annotation, protocols)
    ):
        return
    state = game.make_initial_state()
    _assert_instance_implements_protocols(state, protocols)


def _class_implements_protocols(state_class: type, protocols: tuple[type, ...]) -> bool:
    return all(ProtocolChecker.issubclass(state_class, protocol) for protocol in protocols)


def _assert_instance_implements_protocols(state_instance: Any, protocols: tuple[type, ...]) -> None:
    for protocol in protocols:
        assert \
            ProtocolChecker.isinstance(state_instance, protocol), \
            f"The state class {type(state_instance)} does not implement the protocol {protocol}"


PlayerContra = TypeVar("PlayerContra", contravariant = True)


@runtime_checkable
class ActionFilter(Protocol[PlayerContra, Action]):
    """This is a protocol for a perspective"""

    @abstractmethod
    def filter_actions(self, actions: tuple[Action, ...]) -> tuple[Action, ...]:
        """Removes the actions that are not allowed."""

    @abstractmethod
    def filter_action(self, player: PlayerContra, action: Action) -> Action:
        """Removes the actions that are not allowed."""


class NotificationDispatcher(GlobalObserver, Generic[State, Player, Action]):

    def __init__(
        self,
        global_observer: GlobalObserver,
        policies: dict[Player, InteractivePolicy],
        perspectives: Mapping[Player, Perspective]
    ):
        self.global_observer = global_observer
        self.matchings = tuple(
            (policy, perspectives[p])
            for (p, policy) in policies.items()
        )

    def notify_state(self, state: State) -> None:
        self.global_observer.notify_state(state)
        for (policy, perspective) in self.matchings:
            policy.notify_observation(perspective.get_observation(state))

    def notify_actions(self, actions: tuple[Action, ...]) -> None:
        self.global_observer.notify_actions(actions)
        for (policy, perspective) in self.matchings:
            if isinstance(perspective, ActionFilter):
                actions = perspective.filter_actions(actions)
            policy.notify_actions(actions)

    def notify_action(self, player: Player, action: Action) -> None:
        self.global_observer.notify_action(player, action)
        for (policy, perspective) in self.matchings:
            if isinstance(perspective, ActionFilter):
                action = perspective.filter_action(player, action)
            policy.notify_action(player, action)

    def notify_stop_event(self, stop_event: StopEvent) -> None:
        self.global_observer.notify_stop_event(stop_event)
        for (policy, _) in self.matchings:
            policy.notify_stop_event(stop_event)
