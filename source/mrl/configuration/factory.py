from typing import get_type_hints, Any, Callable, Type
from types import FunctionType
import importlib
import inspect
from pydantic import BaseModel, create_model, ConfigDict
from mrl.configuration.predefined import predefined_modules


class Configuration(BaseModel):
    name: str
    module: str | None = None


def make_object(data: dict) -> Any:
    base_config = Configuration.model_validate(data)
    constructor = _import(base_config)

    configuration_class = _get_configuration_class(constructor)
    if configuration_class is None:
        return constructor()

    configuration = configuration_class.model_validate(data)
    return constructor(**{
        key: getattr(configuration, key)
        for key in configuration_class.model_fields
    })


def _import(config: Configuration) -> Callable:
    if config.module is None:
        module = predefined_modules.get(config.name)
        if module is None:
            raise TypeError(
                f'No predefined module available for object {config.name}. '
                'Did you forget to include a "module" field in the configuration?'
            )
    else:
        module = config.module

    try:
        return getattr(
            importlib.import_module(module),
            config.name
        )
    except (ImportError, AttributeError) as error:
        raise TypeError(
            f'Failed to import {config.name} from module '
            f'{config.module}. Is the module accessible? '
            f'Import error was: "{error}"'
        ) from error


def _get_configuration_class(constructor: Callable) -> Type[BaseModel] | None:
    constructor_function = _get_constructor_function(constructor)
    if constructor_function is None:
        return None

    hints = get_type_hints(constructor_function)
    parameters: dict[str, Any] = {
        key: (
            hints.get(key, Any),
            ... if parameter.default is inspect.Parameter.empty else parameter.default
        )
        for (key, parameter) in inspect.signature(constructor_function).parameters.items()
        if (
            key != 'self' and
            parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        )
    }

    if len(parameters) == 0:
        return None
    return create_model(
        f"{constructor.__name__}Model",
        __config__ = ConfigDict(arbitrary_types_allowed = True),
        **parameters
    )


def _get_constructor_function(constructor: Callable) ->  Callable[..., Any] | None:
    if isinstance(constructor, FunctionType):
        return constructor
    if isinstance(constructor, type):
        return _get_constructor_method(constructor)
    raise TypeError(
        f"Constructor {constructor} is invalid. It should be either "
        "the class constructed itself, or a function"
    )


def _get_constructor_method(cls) -> Callable[..., Any] | None:
    for base in cls.__mro__:
        if "__init__" in base.__dict__:
            if base.__dict__["__init__"] is not object.__init__:
                return base.__dict__["__init__"]
    return None
