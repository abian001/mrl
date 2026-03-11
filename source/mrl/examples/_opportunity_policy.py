from mrl.tic_tac_toe.game import State, Action, Player, TicTacToe
from mrl.test_utils.policies import RandomPolicy
from mrl.game.game import Game


class OpportunityPolicy:

    def __init__(
        self,
        player: Player,
        game: Game,
        check_winning: bool = False,
        check_losing: bool = False
    ):
        self.player = player
        self.other_player = next(p for p in game.get_players() if p != player)
        self.check_winning = check_winning
        self.check_losing = check_losing
        self.random_policy: RandomPolicy = RandomPolicy()

    def __call__(self, observation: State, action_space: tuple[Action, ...]) -> Action:
        index = None
        if self.check_winning:
            index = self._find_free_cell_on_almost_complete_line(observation, self.player)
        if index is None and self.check_losing:
            index = self._find_free_cell_on_almost_complete_line(observation, self.other_player)
        if index is None:
            index = self.random_policy(observation, action_space)
        assert index in action_space
        return index

    @staticmethod
    def _find_free_cell_on_almost_complete_line(
        observation: State,
        player: Player
    ) -> int | None:
        for line in TicTacToe.winning_lines:
            player_cells = tuple((x for x in line if observation.board[x] == player))
            empty_cells = tuple((x for x in line if observation.board[x] == Player.EMPTY))
            if len(player_cells) == 2 and len(empty_cells) == 1:
                return empty_cells[0]
        return None
