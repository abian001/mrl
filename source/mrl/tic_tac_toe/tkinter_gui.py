from functools import partial
import tkinter as tk
import tkinter.font as tkfont
from mrl.configuration.factory import ObjectConfiguration
from mrl.tkinter_gui.gui import (
    Gui,
    View,
    SingleChoiceDropdown,
    Model,
    SettingsView,
    Event,
    GameController
)
from mrl.tic_tac_toe.game import (
    Symbol,
    Player
)


def make_gui(game_configuration: ObjectConfiguration):
    model: Model[Player] = Model(game_configuration, _help_text)
    gui = Gui(model)

    gui.add_view(TicTacToeView(gui))
    gui.add_view(TicTacToeSettings(gui))
    return gui


_help_text = """
    The goal of the game is to create a line of three identical symbols.
    Players take turns placing their symbol in one of the nine squares on the grid.
    The first player to form a horizontal, vertical, or diagonal line of three symbols wins.
"""


class TicTacToeView(View):

    def __init__(self, gui):
        self.root = gui.tk_root
        self.model = gui.model
        self.game_controller = gui.controller
        self.observer = gui.observer
        self.root.title("Tic Tac Toe - Tkinter")
        self.create_board(self.game_controller)
        self.status_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.status_label.grid(row=3, column=0, columnspan=3)

    def create_board(self, controller: GameController[int]):
        self.buttons: list[list[tk.Button | None]] = [[None for _ in range(3)] for _ in range(3)]
        for r in range(3):
            for c in range(3):
                btn = tk.Button(
                    self.root,
                    text="",
                    font=("Arial", 20),
                    width=5,
                    height=2,
                    command=partial(controller.send_action, 3 * r + c)
                )
                btn.grid(row=r, column=c)
                self.buttons[r][c] = btn

    def update(self, event: Event):
        if event != Event.STATE:
            return
        state = self.model.game_state
        for row in range(3):
            for col in range(3):
                symbol = state.board[3*row+col]
                value = "" if symbol == Symbol.EMPTY else str(symbol)
                button = self.buttons[row][col]
                assert button is not None
                button.config(text=value)
        if state.is_final:
            if state.winner == Symbol.EMPTY:
                message = "It is a draw"
            else:
                message = f"Player {state.winner} wins"
        elif self.model.game_is_running:
            message = f"It is Player {state.active_player}'s turn"
        else:
            message = ''
        self.status_label.config(text = message)
        self.observer.notify_message(message)


class SettingsController:

    def __init__(self, model, observer):
        self.model = model
        self.observer = observer

    def save_settings(self, settings_window, dropdown):
        configuration_data = self.model.game_configuration.to_data()
        try:
            game_configuration = ObjectConfiguration.model_validate(
                configuration_data | {'first_player': dropdown.get_choice()}
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            self.observer.notify_message(f"Invalid settings: {error}")
        else:
            self.model.game_configuration = game_configuration
        settings_window.destroy()

    def close_settings(self, settings_window):
        settings_window.destroy()


class TicTacToeSettings(SettingsView):

    def __init__(self, gui):
        self.model = gui.model
        self.root = gui.tk_root
        self.controller = SettingsController(self.model, gui.observer)

    def create_settings(self):
        font = tkfont.Font(family = "Helvetica", size = 12)
        window = tk.Toplevel(self.root)
        window.title("Settings")
        window.geometry("500x300")

        configuration_data = self.model.game_configuration.to_data()
        dropdown = SingleChoiceDropdown(
            tk_parent = window,
            label = "first player",
            choices = (str(Player.X), str(Player.O), "random"),
            default = configuration_data["first_player"],
            font = font
        )
        dropdown.pack(pady = 20)

        button = tk.Button(
            window,
            text = "Save",
            width = 12,
            command = lambda window=window, dropdown=dropdown: \
                self.controller.save_settings(window, dropdown),
            font = font
        )
        button.pack(pady = (10, 10))

        button = tk.Button(
            window,
            text = "Close",
            width = 12,
            command = lambda window=window: self.controller.close_settings(window),
            font = font
        )
        button.pack(pady = (10, 10))
