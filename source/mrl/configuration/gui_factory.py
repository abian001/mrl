from mrl.configuration.factory import ObjectConfiguration, make_object
from mrl.configuration.predefined import predefined_guis
from mrl.tkinter_gui.gui import Gui


def make_gui(
    game_configuration: ObjectConfiguration,
    gui_configuration: ObjectConfiguration | None = None
) -> Gui:
    if gui_configuration is None:
        gui_configuration = _get_predefined_gui_configuration(game_configuration)

    gui = make_object(
        gui_configuration,
        extra_arguments = {'game_configuration': game_configuration}
    )
    if not isinstance(gui, Gui):
        raise TypeError(
            f"Invalid gui class {type(gui)}. "
            "Gui classes should derive from tkinter_gui.gui.Gui."
        )
    return gui


def _get_predefined_gui_configuration(
    game_configuration: ObjectConfiguration
) -> ObjectConfiguration:
    predefined_gui = predefined_guis.get(game_configuration.name)
    if predefined_gui is None:
        raise TypeError(
            f"No predefined gui for game {game_configuration.name}."
        )
    return ObjectConfiguration(name = predefined_gui[0], module = predefined_gui[1])
