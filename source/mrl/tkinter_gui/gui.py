from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from typing import Protocol, Generic, Mapping
from functools import partial
from collections import deque
import threading
import tkinter as tk
import tkinter.font  # pylint: disable=unused-import
from mrl.game.game import (
    GlobalObserver,
    Policy,
    Player,
    Action,
    FinalCheckable,
    StopEvent,
    Perspective,
    HasActivePlayer
)
from mrl.game.game_loop import make_game_loop
from mrl.configuration.factory import make_object
from mrl.tkinter_gui.policy import TkinterPolicy


class QueryableState(FinalCheckable, HasActivePlayer, Protocol):
    """A state that can be queried by the GUI"""


def _make_default_deque():
    return deque(maxlen = 100)


@dataclass
class Model(Generic[Player]):
    game_data: dict
    help_text: str
    game_state: QueryableState | None = None
    game_is_running: bool = False
    log: deque[str] = field(default_factory = _make_default_deque)
    policies: dict[Player, Policy] = field(default_factory = dict)
    perspectives: Mapping[Player, Perspective] = field(default_factory = dict)


class Event(Enum):

    STATE = 0
    ACTION = 1
    STOP = 2
    MESSAGE = 4


class View(ABC):

    @abstractmethod
    def update(self, event: Event):
        """Update the view with the recent version of the model"""


class SettingsView(ABC):

    @abstractmethod
    def create_settings(self):
        """Create a new window with the settings"""


@dataclass
class Gui(Generic[Player]):
    model: Model[Player]
    tk_root: tk.Tk
    observer: "GameObserver"
    controller: "GameController"
    views: list[View]

    def __init__(self, model: Model[Player]):
        self.model = model
        self.tk_root = tk.Tk()
        self.views = []
        self.observer = GameObserver(self.model, self.tk_root, self.views)
        self.controller = GameController(self.model, self.tk_root, self.observer)
        self.tk_root.protocol("WM_DELETE_WINDOW", self.controller.exit)

        self._main_view = MainView(self.tk_root, self.model, self.controller)
        self.add_view(self._main_view)

    def add_view(self, view: View | SettingsView):
        if isinstance(view, SettingsView):
            self._main_view.attach_settings_view(view)
        if isinstance(view, View):
            self.views.append(view)

    def run(self):
        self._main_view.create_menu()
        self.tk_root.mainloop()
        self.tk_root.destroy()

    def set_policies(self, policies: dict[Player, Policy]):
        self.model.policies = policies


class GameObserver(GlobalObserver, Generic[Player, Action]):

    def __init__(self, model: Model, tk_root: tk.Tk, views: list[View]):
        self.model = model
        self.tk_root = tk_root
        self.views = views
        self._accept_notifications = True
        self._log_lock = threading.Lock()
        self._notification_lock = threading.Lock()

    @property
    def accept_notifications(self) -> bool:
        with self._notification_lock:
            return self._accept_notifications

    @accept_notifications.setter
    def accept_notifications(self, value: bool) -> None:
        with self._notification_lock:
            self._accept_notifications = value

    def notify_state(self, state: QueryableState):
        # Method called only by the game loop thread
        self.model.game_state = state
        self.model.game_is_running = not state.is_final
        self._update_views(Event.STATE)

    def notify_action(self, player: Player, action: Action):
        # Method called only by the game loop thread
        self._append_to_log(f"Player {player} played action {action}")
        self._update_views(Event.ACTION)

    def notify_stop_event(self, stop_event: StopEvent):
        # Method called only by the game loop thread
        if self.accept_notifications:
            self.model.game_is_running = False
            self._append_to_log("The game was stopped")
            self._update_views(Event.STOP)

    def notify_message(self, message: str):
        # Method called by both the game loop thread and the main gui thread
        if self.accept_notifications:
            self._append_to_log(message)
            self._update_views(Event.MESSAGE)

    def notify_exit(self):
        # Method called only by the main gui thread
        self.accept_notifications = False

    def _append_to_log(self, message: str):
        with self._log_lock:
            self.model.log.append(message)

    def _update_views(self, event: Event):
        for view in self.views:
            self.tk_root.after(0, partial(view.update, event = event))


class GameController(Generic[Action]):

    def __init__(self, model: Model, tk_root: tk.Tk, observer: GameObserver):
        self.model = model
        self.policy: TkinterPolicy = TkinterPolicy()
        self.observer = observer
        self.tk_root = tk_root
        self.game_thread = None

    def new_game(self):
        if self.model.game_is_running:
            return
        self.game_thread = threading.Thread(target = self._run_game_loop)
        self.game_thread.start()
        self.model.game_is_running = True
        self.observer.notify_message("New game started")

    def _run_game_loop(self):
        try:
            self._make_game_loop().run()
        except Exception as error:  # pylint: disable=broad-exception-caught
            self.observer.notify_message(f"Error observed: {error}")
        else:
            self.observer.notify_message("The game is over.")

    def stop_game(self):
        if not self.model.game_is_running:
            return
        self.policy.send_stop_action()

    def exit(self):
        if self.model.game_is_running:
            self.stop_game()
        self.tk_root.quit()
        self.observer.notify_exit()

    def send_action(self, action: Action) -> None:
        try:
            if not self.model.game_is_running or self.model.game_state is None:
                return
            active_perspective = self.model.perspectives[self.model.game_state.active_player]
            if action not in active_perspective.get_action_space(self.model.game_state):
                return
            self.policy.send_action(action)
        except Exception as error:  # pylint: disable=broad-exception-caught
            self.observer.notify_message(
                f"Error observed during execution of action {action}. "
                f"The error message was: {error}"
            )

    def _make_game_loop(self):
        game = make_object(self.model.game_data)
        self.model.perspectives = game.get_perspectives()
        return make_game_loop(
            game = game,
            policies = {
                player: self.policy for player in game.get_players()
            } | self.model.policies,
            global_observer = self.observer
        )

    def save_settings(self):
        pass


class MainView(View):

    def __init__(self, tk_root: tk.Tk, model: Model, controller: GameController):
        self.root = tk_root
        self.model = model
        self.controller = controller
        self.log_window = None
        self.help_window = None
        self.settings_view: SettingsView | None = None

    def update(self, event: Event):
        self.update_log()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        game_menu = tk.Menu(menubar, tearoff=0)
        game_menu.add_command(label="New Game", command=self.controller.new_game)
        game_menu.add_command(label="Stop Game", command=self.controller.stop_game)
        if self.settings_view is not None:
            game_menu.add_command(label="Settings", command = self.settings_view.create_settings)
        game_menu.add_command(label="Log", command = self.create_log_window)
        game_menu.add_command(label="Help", command = self.create_help_window)
        game_menu.add_command(label="Exit", command=self.controller.exit)
        menubar.add_cascade(label="Game", menu=game_menu)
        self.root.config(menu=menubar)

    def attach_settings_view(self, view: SettingsView):
        self.settings_view = view

    def create_log_window(self):
        if self.log_window is None:
            self.log_window = LogWindow(self.root, 10)
            self.log_window.protocol("WM_DELETE_WINDOW", self.close_log_window)
            self.update_log()

    def create_help_window(self):
        if self.help_window is None:
            self.help_window = HelpWindow(self.root, self.model.help_text)
            self.help_window.protocol("WM_DELETE_WINDOW", self.close_help_window)

    def update_log(self):
        if self.log_window is not None:
            self.log_window.log_update(self.model.log)

    def close_log_window(self):
        if self.log_window is not None:
            self.log_window.destroy()
            self.log_window = None

    def close_help_window(self):
        if self.help_window is not None:
            self.help_window.destroy()
            self.help_window = None


class HelpWindow(tk.Toplevel):

    def __init__(self, tk_parent, help_text, **kwargs):
        super().__init__(tk_parent, **kwargs)
        font = tk.font.Font(family = "Helvetica", size = 12)
        self.label = tk.Label(
            self,
            text = help_text,
            anchor = "w",
            justify = "left",
            font = font
        )
        self.label.pack()


class LogWindow(tk.Toplevel):

    def __init__(self, tk_parent, number_of_entries, **kwargs):
        super().__init__(tk_parent, **kwargs)
        font = tk.font.Font(family = "Helvetica", size = 12)
        self.entries = [
            tk.Label(self, text = "", width = 60, anchor = "w", font = font)
            for _ in range(number_of_entries)
        ]
        for entry in self.entries:
            entry.pack()

    def log_update(self, log: deque[str]):
        log_split = self._get_log_split(log)
        number_of_entries = min(len(self.entries), len(log_split))
        for i in range(number_of_entries):
            self.entries[number_of_entries - i - 1].config(
                text = log_split[len(log_split) - i - 1]
            )

    def _get_log_split(self, log: deque[str]):
        split = []
        for x in log:
            for s in range(0, len(x), 60):
                split.append(x[s:s+60])
        return split


class SingleChoiceDropdown(tk.Frame):

    # pylint: disable=too-many-positional-arguments
    def __init__(self, tk_parent, label, choices, default, font, **kwargs):
        super().__init__(tk_parent, **kwargs)
        self.choice = tk.StringVar(value = default)

        self.label = tk.Label(
            self,
            text = label,
            width = 20,
            anchor = 'w',
            font = font
        )
        self.label.grid(row=0, column=0, padx=(0, 10))

        self.button = tk.Button(
            self,
            textvariable = self.choice,
            width = 12,
            command = self._show_dropdown,
            font = font
        )
        self.button.grid(row=0, column=1)

        self.dropdown_frame = tk.Frame(self)
        for choice in choices:
            rb = tk.Radiobutton(
                self.dropdown_frame,
                text=choice,
                variable=self.choice,
                value=choice,
                command=self._hide_dropdown,
                font = font
            )
            rb.pack(anchor="w")

    def _show_dropdown(self):
        self.dropdown_frame.grid(row=0, column=2, padx=(10, 0))

    def _hide_dropdown(self):
        self.dropdown_frame.grid_forget()

    def get_choice(self):
        return self.choice.get()
