import copy
import itertools
from typing import Iterable, Mapping
from abc import abstractmethod
import numpy as np
from mrl.alpha_zero.mcts_observation import MCTSObservation
from mrl.xiangqi.game import Token
from mrl.xiangqi.enumerable_game import (
    BaseEnumerableXiangqi,
    Player,
    Role,
    State,
    Action,
    action_space_dimension,
    BasePerspective
)


class Perspective(BasePerspective[State, MCTSObservation, Action]):

    def __init__(self, player: Player) -> None:
        super().__init__(player)
        self.opponent = Player.RED if player == Player.BLACK else Player.BLACK

    def get_observation(self, state: State) -> MCTSObservation:
        return MCTSObservation(state, self)

    def get_core(self, state: State) -> np.ndarray:
        return np.fromiter(
           (
               t.role == role and t.color == player
               for player in (self.player, self.opponent)
               for role in (
                   Role.GENERAL, Role.ADVISOR, Role.ELEPHANT,
                   Role.HORSE, Role.CHARIOT, Role.CANNON, Role.SOLDIER
               )
               for t in self._token_iterator(state)
           ), dtype = bool
        )

    @classmethod
    @abstractmethod
    def _token_iterator(cls, state: State) -> Iterable[Token]:
        """Iterates through the board tokens"""

    @property
    def action_space_dimension(self):
        return action_space_dimension


class RedPerspective(Perspective):

    def __init__(self):
        super().__init__(Player.RED)

    @classmethod
    def _token_iterator(cls, state: State) -> Iterable[Token]:
        return (cell.token for cell in itertools.chain(*state.board.board))


class BlackPerspective(Perspective):

    def __init__(self):
        super().__init__(Player.BLACK)

    @classmethod
    def _token_iterator(cls, state: State) -> Iterable[Token]:
        return (
            cell.token for cell in itertools.chain(*(
                reversed(x) for x in reversed(state.board.board)
            ))
        )


class MCTSXiangqi(BaseEnumerableXiangqi):

    def restore(self, observation: MCTSObservation) -> State:
        return copy.deepcopy(observation.state)

    def get_perspectives(self) -> Mapping[Player, Perspective]:
        return {
            p: RedPerspective() if p == Player.RED else BlackPerspective()
            for p in self.get_players()
        }
