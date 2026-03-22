from functools import partial
import tkinter as tk
from mrl.configuration.factory import ObjectConfiguration
from mrl.straight_four.game import Token, Player
from mrl.tkinter_gui.gui import View, Model, Gui, Event, GameController


def make_gui(game_configuration: ObjectConfiguration):
    model: Model[Player] = Model(game_configuration, _help_text)
    gui = Gui(model)

    gui.add_view(StraightFourView(gui))
    return gui


_help_text = """
    The objective of the game is to create a line of four tokens of the same type.
    Players take turns placing a token into a column by clicking the top square of that column.
    The token will fall to the lowest available space.
    The first player to form a horizontal, vertical, or diagonal line of four tokens wins.
"""


class StraightFourView(View):

    def __init__(self, gui):

        self.root = gui.tk_root
        self.model = gui.model
        self.game_controller = gui.controller
        self.observer = gui.observer
        self.root.title("Straight Four - Tkinter")
        self.create_board(self.game_controller)
        self.status_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.status_label.grid(row=8, column=0, columnspan=7)
        self.log_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.log_label.grid(row=9, column=0, columnspan=7)

    def create_board(self, controller: GameController[int]):
        self.buttons: list[list[tk.Button | None]] = [[None for _ in range(7)] for _ in range(7)]
        for r in range(7):
            for c in range(7):
                btn = tk.Button(
                    self.root,
                    text="",
                    font=("Arial", 20),
                    width=5,
                    height=2,
                    command=partial(controller.send_action, 7 * r + c)
                )
                btn.grid(row=r, column=c)
                self.buttons[r][c] = btn

    def update(self, event: Event):
        if event != Event.STATE:
            return
        state = self.model.game_state
        for row in range(7):
            for col in range(7):
                symbol = state.board[7*row+col]
                value = "" if symbol == Token.NONE else str(symbol)
                button = self.buttons[row][col]
                assert button is not None
                button.config(text=value)
        if state.is_final:
            if state.winner == Token.NONE:
                message = "It is a draw"
            else:
                message = f"Player {state.winner} wins"
        elif self.model.game_is_running:
            message = f"It is Player {state.active_player}'s turn"
        else:
            message = ''
        self.status_label.config(text = message)
        self.log_label.config(text="" if len(self.model.log) == 0 else self.model.log[-1])
