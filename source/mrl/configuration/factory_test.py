from enum import Enum
from dataclasses import dataclass
import pytest
from mrl.configuration.factory import make_object
from mrl.configuration.gui_factory import make_gui
from mrl.tkinter_gui.gui import Gui


class Player(Enum):

    RED = 'red'
    BLACK = 'black'

    def __str__(self):
        return self.value


class ObjectWithNoConstructor:
    pass

class ObjectWithNoParameters:

    def __init__(self):
        pass


@dataclass
class Configuration:
    test_value: int
    first_player: Player


class ObjectWithConfiguration:

    def __init__(self, config: Configuration):
        self.config = config


class ObjectWithParameters:

    def __init__(
        self,
        test_value: int,
        first_player: Player = Player.BLACK,
        second_player: Player = Player.RED
    ):
        self.test_value = test_value
        self.first_player = first_player
        self.second_player = second_player


@pytest.mark.parametrize('object_class', [ObjectWithNoConstructor, ObjectWithNoParameters])
@pytest.mark.quick
def test_make_game(object_class: type):
    instance = make_object({
        'name': object_class.__name__,
        'module': 'mrl.configuration.factory_test'
    })
    assert isinstance(instance, object_class)


@pytest.mark.quick
def test_make_game_with_configuration():
    instance = make_object({
        'name': 'ObjectWithConfiguration',
        'module': 'mrl.configuration.factory_test',
        'config': {
            'test_value': '3',
            'first_player': 'red'
        }
    })
    assert isinstance(instance, ObjectWithConfiguration)
    assert instance.config.test_value == 3
    assert instance.config.first_player == Player.RED


@pytest.mark.quick
def test_make_game_with_parameters():
    instance = make_object({
        'name': 'ObjectWithParameters',
        'module': 'mrl.configuration.factory_test',
        'test_value': '2',
        'second_player': 'black'
    })
    assert isinstance(instance, ObjectWithParameters)
    assert instance.test_value == 2
    assert instance.first_player == Player.BLACK
    assert instance.second_player == Player.BLACK


@pytest.mark.parametrize('module', ['missing_module', 'mrl.configuration.factory_test'])
@pytest.mark.quick
def test_import_error(module: str):
    with pytest.raises(TypeError) as error:
        make_object({
            'name': 'Missing',
            'module': module
        })
    if module == 'missing_module':
        specific_error = f"\"No module named '{module}'\""
    else:
        specific_error = f"\"module '{module}' has no attribute 'Missing'\""
    assert str(error.value) == (
        f"Failed to import Missing from module {module}. "
        f"Is the module accessible? Import error was: {specific_error}"
    )


@pytest.mark.quick
def test_import_predefined_error():
    with pytest.raises(TypeError) as error:
        make_object({
            'name': 'Missing'
        })
    assert str(error.value) == (
        'No predefined module available for object Missing. '
        'Did you forget to include a "module" field in the configuration?'
    )


class TestGui(Gui):

    def __init__(self, game_data: dict):  # pylint: disable=super-init-not-called
        self.game_data = game_data


@pytest.mark.quick
def test_make_gui():
    game_data = {
        'name': 'ObjectWithNoConstructor',
        'module': 'mrl.configuration.factory_test'
    }
    instance = make_gui(
        game_data = game_data,
        gui_data = {
            'name': 'TestGui',
            'module': 'mrl.configuration.factory_test'
        }
    )
    assert instance.game_data == game_data


def _make_bad_gui(game_data: dict) -> dict:
    return game_data


@pytest.mark.quick
def test_make_bad_gui():
    game_data = {
        'name': 'ObjectWithNoConstructor',
        'module': 'mrl.configuration.factory_test'
    }
    with pytest.raises(TypeError) as error:
        make_gui(
            game_data = game_data,
            gui_data = {
                'name': '_make_bad_gui',
                'module': 'mrl.configuration.factory_test'
            }
        )
    assert str(error.value) == (
        "Invalid gui class <class 'dict'>. "
        "Gui classes should derive from tkinter_gui.gui.Gui."
    )


@pytest.mark.parametrize('module', ['missing_module', 'mrl.configuration.factory_test'])
@pytest.mark.quick
def test_import_gui_error(module: str):
    game_data = {
        'name': 'ObjectWithNoConstructor',
        'module': 'mrl.configuration.factory_test'
    }
    with pytest.raises(TypeError) as error:
        make_gui(
            game_data = game_data,
            gui_data = {
                'name': 'Missing',
                'module': module
            }
        )
    if module == 'missing_module':
        specific_error = f"\"No module named '{module}'\""
    else:
        specific_error = f"\"module '{module}' has no attribute 'Missing'\""
    assert str(error.value) == (
        f"Failed to import Missing from module {module}. "
        f"Is the module accessible? Import error was: {specific_error}"
    )


@pytest.mark.quick
def test_import_gui_no_module_error():
    game_data = {
        'name': 'ObjectWithNoConstructor',
        'module': 'mrl.configuration.factory_test'
    }
    with pytest.raises(TypeError) as error:
        make_gui(
            game_data = game_data,
            gui_data = {
                'name': 'Missing'
            }
        )
    assert str(error.value) == (
        'No predefined module available for object Missing. '
        'Did you forget to include a "module" field in the configuration?'
    )


@pytest.mark.quick
def test_import_gui_game_error():
    game_data = {
        'name': 'Missing',
        'module': 'mrl.configuration.factory_test'
    }
    with pytest.raises(TypeError) as error:
        make_gui(
            game_data = game_data,
            gui_data = None
        )
    assert str(error.value) == "No predefined gui for game Missing."
