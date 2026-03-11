from mrl.configuration.factory import Configuration, make_object
from mrl.configuration.predefined import predefined_guis
from mrl.tkinter_gui.gui import Gui


def make_gui(game_data: dict, gui_data: dict | None = None) -> Gui:
    if gui_data is None:
        gui_data = _get_predefined_gui_data(game_data)

    gui = make_object(gui_data | {'game_data': game_data})
    if not isinstance(gui, Gui):
        raise TypeError(
            f"Invalid gui class {type(gui)}. "
            "Gui classes should derive from tkinter_gui.gui.Gui."
        )
    return gui


def _get_predefined_gui_data(game_data: dict) -> dict:
    base_game_config = Configuration.model_validate(game_data)
    predefined_gui = predefined_guis.get(base_game_config.name)
    if predefined_gui is None:
        raise TypeError(
            f"No predefined gui for game {base_game_config.name}."
        )
    return {'name': predefined_gui[0], 'module': predefined_gui[1]}
