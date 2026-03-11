from typing import Dict, Generic, Mapping, Protocol, runtime_checkable
from abc import abstractmethod
from dataclasses import dataclass, field
import math
import numpy as np
from mrl.game.game import (
    StopEvent,
    Restorable,
    State,
    Player,
    InteractivePolicy,
    TurnBased,
    Game,
    PayoffObservable
)
from mrl.game.game_loop import TurnBasedState
from mrl.alpha_zero.oracle import Oracle, DiscreteActionSpace
from mrl.alpha_zero.mcts_observation import MCTSObservation, MCTSPerspective


Action = int


class OracleEvent(StopEvent):

    def __init__(self, payoff):
        self.payoff = payoff


@runtime_checkable
class MCTSGame(
    PayoffObservable[State, Player, MCTSObservation],
    Game[State, Player],
    TurnBased[State, Action],
    Restorable[State, MCTSObservation],
    Protocol[State, Player]
):

    @abstractmethod
    def get_perspectives(self) -> Mapping[Player, MCTSPerspective[State, Action]]:
        """Return the player payoff perspectives."""



@dataclass
class StateNode:
    visits: int = 0
    actions: Dict[Action, 'ActionNode'] = field(default_factory = dict)


@dataclass
class ActionNode:
    visits: int = 0
    payoff: float = 0.0
    states: Dict[bytes, StateNode] = field(default_factory = dict)


class PUCBPolicy:

    def __init__(self, exploration_weight: float, oracle: Oracle) -> None:
        self.state_node: StateNode = StateNode()
        self.action_node: ActionNode | None = None
        self.exploration_weight = exploration_weight
        self.oracle = oracle

    def reset(self, root: StateNode) -> None:
        self.state_node = root
        self.action_node = None

    def __call__(
        self,
        observation: MCTSObservation,
        action_space: DiscreteActionSpace
    ) -> tuple[Action, StateNode, ActionNode]:
        if self.action_node is not None:
            self.state_node = self.action_node.states.setdefault(
                observation.core.tobytes(),
                StateNode()
            )

        if len(self.state_node.actions) == 0:
            self.state_node.actions = {
                a: ActionNode()
                for (a, legal) in enumerate(observation.legal_mask)
                if legal
            }

        action = self._pucb_choice(observation)
        self.action_node = self.state_node.actions[action]
        return action, self.state_node, self.action_node

    def _pucb_choice(self, observation: MCTSObservation) -> Action:
        priors = self.oracle.get_probabilities(observation, observation.legal_mask)
        return max(
            self.state_node.actions.items(),
            key = lambda x: self._compute_pucb(x[1], priors[x[0]])
        )[0]

    def _compute_pucb(self, action_node: ActionNode, prior: float) -> float:
        payoff = 0.0 if action_node.visits == 0 else action_node.payoff / action_node.visits
        return payoff + (
            self.exploration_weight * prior *
            (math.sqrt(self.state_node.visits + 1) / (action_node.visits + 1))
        )


@dataclass
class MCTSConfiguration:
    number_of_simulations: int = 25
    pucb_constant: float = 1.0
    discount_factor: float = 1.0
    temperature: float = 1.0


class MCTSPolicy(Generic[Player]):

    def __init__(
        self,
        game: MCTSGame,
        oracle: Oracle,
        mcts: MCTSConfiguration
    ):
        self.game = game
        self.pucb_policy = PUCBPolicy(mcts.pucb_constant, oracle)
        self.oracle = oracle
        self.discount_factor = mcts.discount_factor
        self.number_of_simulations = mcts.number_of_simulations
        self.perspectives = game.get_perspectives()

    def __call__(
        self,
        observation: MCTSObservation,
        action_space: DiscreteActionSpace
    ) -> Action:
        root = self._simulate(observation)
        return max(
            root.actions.items(),
            key = lambda x: x[1].visits
        )[0]

    def _simulate(self, observation: MCTSObservation) -> StateNode:
        root = StateNode()
        for _ in range(self.number_of_simulations):
            state = self.game.restore(observation)
            self._run_once_from_state(state, root)
        return root

    def _run_once_from_state(self, state: TurnBasedState, root: StateNode) -> None:
        buffers: dict[Player, list] = {p: [] for p in self.perspectives}
        self.pucb_policy.reset(root)
        while not state.is_final:
            observation = self.perspectives[state.active_player].get_observation(state)
            action_space = self.perspectives[state.active_player].get_action_space(state)
            action, state_node, action_node = self.pucb_policy(observation, action_space)
            buffers[state.active_player].append((observation.payoff, state_node, action_node))
            if action_node.visits == 0:
                break
            state = self.game.update(state, action)

        last_payoffs = self._get_last_payoffs(state)
        self._update_nodes(buffers, last_payoffs)

    def _update_nodes(self, buffers: dict[Player, list], last_payoffs: dict[Player, float]):
        for (p, buffer) in buffers.items():
            cumulated_payoff = last_payoffs[p]
            for (payoff, state_node, action_node) in reversed(buffer):
                action_node.payoff += cumulated_payoff
                action_node.visits += 1
                state_node.visits += 1
                cumulated_payoff = payoff + self.discount_factor * cumulated_payoff

    def _get_last_payoffs(self, state: TurnBasedState):
        if state.is_final:
            return {
                p: perspective.get_observation(state).payoff
                for (p, perspective) in self.perspectives.items()
            }
        return {
            p: self.oracle.get_value(perspective.get_observation(state))
            for (p, perspective) in self.perspectives.items()
        }


class NonDeterministicMCTSPolicy(MCTSPolicy):

    _empty_action_node = ActionNode()

    def __init__(
        self,
        game: MCTSGame,
        oracle: Oracle,
        mcts: MCTSConfiguration
    ):
        super().__init__(game, oracle, mcts)
        any_perspective = next(iter(self.game.get_perspectives().values()))
        # All perspectives should have the same action space size for an MCTSGame
        self.full_action_space = tuple(range(any_perspective.action_space_dimension))
        self.temperature = mcts.temperature

    def __call__(
        self,
        observation: MCTSObservation,
        action_space: DiscreteActionSpace
    ) -> Action:
        if self.temperature < 1e-9:
            return super().__call__(observation, action_space)
        return self.get_action_and_probabilities(observation, self.temperature)[0]

    def get_action_and_probabilities(
        self,
        observation: MCTSObservation,
        temperature: float
    ) -> tuple[Action, np.ndarray]:
        root = self._simulate(observation)
        temperature_inverse = 1.0 / temperature
        probabilities = np.fromiter((
            (
                root.actions.get(a, self._empty_action_node).visits / root.visits
            ) ** temperature_inverse
            for a in self.full_action_space
        ), dtype = float)
        action = np.random.choice(self.full_action_space, p = probabilities)
        return action, probabilities


class MemoryfullMCTSPolicy(MCTSPolicy, InteractivePolicy, Generic[Player]):

    def __init__(
        self,
        game: MCTSGame,
        oracle: Oracle,
        mcts: MCTSConfiguration
    ):
        super().__init__(game, oracle, mcts)
        self.root = StateNode()
        self.state_node = self.root
        self.action_node: ActionNode | None = None

    def _simulate(self, observation: MCTSObservation) -> StateNode:
        for _ in range(self.number_of_simulations):
            state = self.game.restore(observation)
            self._run_once_from_state(state, self.state_node)
        return self.state_node

    def notify_observation(self, observation: MCTSObservation) -> None:
        if observation.is_final:
            self.state_node = self.root
            self.action_node = None
        elif self.action_node is not None:
            self.state_node = self.action_node.states.setdefault(
                observation.core.tobytes(),
                StateNode()
            )

    def notify_action(self, player: Player, action: Action) -> None:
        self.action_node = self.state_node.actions.setdefault(action, ActionNode())
