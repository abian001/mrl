from typing import Generic
from abc import ABC, abstractmethod
from functools import partial
import tkinter as tk
from mrl.game.game import Action as ActionType
from mrl.xiangqi.game import Role, Player, _token_to_string, Action, BaseState
from mrl.xiangqi.enumerable_game import (
    base_action_to_extended_action,
    Action as EnumerableAction
)
from mrl.xiangqi.mcts_game import MCTSXiangqi
from mrl.tkinter_gui.gui import View, Model, Gui, Event


def make_gui(game_data: dict):
    model = XiangqiModel(game_data, _help_text)
    gui = Gui(model)

    gui.add_view(XiangqiView(gui))
    return gui


_help_text = """
    The objective of the game is to checkmate or stalemate the opponent's General.
    Players take turns moving their pieces on the board according to their movement rules.
    See rules at https://www.xiangqi.com/how-to-play

    To move a piece, click on the piece and then click the destination square.
    If a move is invalid, nothing will happen. You can check the game log
    for additional information, but the reason for the invalid move
    will not be shown (for example, your general may be in check).
    Apologies! This GUI is for quick manual testing.

    Lengend of pieces:
    Red side:
        (S) soldier, (G) general, (A) advisor, (E) elephant,
        (H) horse, (C) cannon, (R) chariot.
    Black side:
        (s) soldier, (g) general, (a) advisor, (e) elephant,
        (h) horse, (c) cannon, (r) chariot.
"""


class XiangqiModel(Model):
    selection: tuple[int, int] | None = None


class BaseXiangqiController(ABC, Generic[ActionType]):

    def __init__(self, gui):
        self.model = gui.model
        self.observer = gui.observer
        self.controller = gui.controller

    def make_move(self, row, col):
        assert isinstance(self.model.game_state, BaseState)
        if not self.model.game_is_running or self.model.game_state is None:
            return
        if self.model.selection is None:
            token = self.model.game_state.board[(row, col)]
            if token.role == Role.NONE or token.color != self.model.game_state.active_player:
                return
            self.model.selection = (row, col)
            self.observer.notify_message(f"Selected {token.role} at ({row}, {col})")
        else:
            action = self._make_action(row, col)
            active_perspective = self.model.perspectives[self.model.game_state.active_player]
            if action in active_perspective.get_action_space(self.model.game_state):
                self.controller.send_action(action)
            else:
                self.observer.notify_message("Invalid move. Select again.")
            self.model.selection = None

    @abstractmethod
    def _make_action(self, row: int, col: int) -> ActionType | None:
        """Construct an action as expected from the game"""


class XiangqiController(BaseXiangqiController[Action]):

    def _make_action(self, row: int, col: int) -> Action:
        return Action(self.model.selection, (row, col))


class MCTSXiangqiController(BaseXiangqiController[EnumerableAction]):

    def _make_action(self, row: int, col: int) -> EnumerableAction | None:
        base_action = Action(self.model.selection, (row, col))
        return base_action_to_extended_action(base_action, self.model.game_state)


class XiangqiView(View):

    def __init__(self, gui: Gui):
        self.root = gui.tk_root
        self.model = gui.model
        self.observer = gui.observer
        self.root.title("Xiangqi - Tkinter")
        self._create_board(self._create_controller(gui))
        self.status_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.status_label.grid(row=10, column=0, columnspan=9)

    def _create_controller(self, gui: Gui):
        if self.model.game_data.get('name') == MCTSXiangqi.__name__:
            return MCTSXiangqiController(gui)
        return XiangqiController(gui)

    def _create_board(self, controller: BaseXiangqiController):
        self.buttons: list[list[tk.Button | None]] = [[None for _ in range(9)] for _ in range(10)]
        for r in range(10):
            for c in range(9):
                btn = tk.Button(
                    self.root,
                    text="",
                    font=("Arial", 20),
                    width=4,
                    height=1,
                    command=partial(controller.make_move, r, c)
                )
                btn.grid(row=r, column=c)
                self.buttons[r][c] = btn

    def update(self, event: Event):
        if event != Event.STATE:
            return
        assert isinstance(self.model.game_state, BaseState)
        state = self.model.game_state
        for row in range(10):
            for col in range(9):
                token = state.board[(row, col)]
                value = "" if token.role == Role.NONE else _token_to_string[token]
                button = self.buttons[row][col]
                assert button is not None
                button.config(text=value)
        if state.is_final:
            if state.winner == Player.NONE:
                message = "It is a draw"
            else:
                message = f"Player {state.winner} wins"
        else:
            message = f"It is Player {state.active_player}'s turn"
        self.status_label.config(text = message)
        self.observer.notify_message(message)
