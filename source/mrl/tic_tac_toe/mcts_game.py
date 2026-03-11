import copy
from typing import Mapping
import numpy as np
from mrl.tic_tac_toe.game import (
    BaseTicTacToe,
    Player,
    State,
    BasePerspective
)
from mrl.alpha_zero.mcts_observation import MCTSObservation


class Perspective(BasePerspective[MCTSObservation]):

    def __init__(self, player: Player):
        super().__init__(player)
        self.opponent = Player.O if player == Player.X else Player.X

    def get_observation(self, state: State) -> MCTSObservation:
        return MCTSObservation(state, self)

    def get_core(self, state: State) -> np.ndarray:
        return np.fromiter((
            cell == symbol
            for symbol in (self.player, self.opponent)
            for cell in state.board
        ), dtype = bool)

    @property
    def action_space_dimension(self) -> int:
        return 9


class MCTSTicTacToe(BaseTicTacToe):

    def restore(self, observation: MCTSObservation) -> State:
        return copy.deepcopy(observation.state)

    def get_perspectives(self) -> Mapping[Player, Perspective]:
        return {
            p: Perspective(p) for p in self.get_players()
        }
