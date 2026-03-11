from typing import Any
import inspect


class ProtocolChecker:

    @classmethod
    def isinstance(cls, any_object: Any, protocol: type) -> bool:
        return (
            isinstance(any_object, protocol) and
            cls._has_all_valid_signatures(type(any_object), protocol) and
            cls._has_all_instance_attributes(any_object, protocol)
        )

    @classmethod
    def issubclass(cls, any_class: type, protocol: type) -> bool:
        return (
            cls._has_all_valid_signatures(any_class, protocol) and
            cls._has_all_valid_properties(any_class, protocol)
        )

    @classmethod
    def _has_all_valid_signatures(cls, any_class: type, protocol: type) -> bool:
        methods = cls._get_all_protocol_methods(protocol)
        for method in methods:
            parameters = set(inspect.signature(getattr(protocol, method)).parameters.keys())
            if not cls._has_signature(any_class, method, parameters):
                return False
        return True

    @staticmethod
    def _get_all_protocol_methods(protocol: type) -> tuple[str, ...]:
        return tuple(
            name for (name, element) in protocol.__dict__.items()
            if callable(element) and element.__qualname__.startswith(protocol.__name__)
        )

    @staticmethod
    def _get_all_protocol_properties(protocol: type) -> tuple[str, ...]:
        return tuple(
            name for (name, element) in protocol.__dict__.items()
            if isinstance(element, property)
        )

    @staticmethod
    def _has_signature(any_class: Any, method: str, parameters: set[str]):
        if not hasattr(any_class, method):
            return False
        method = getattr(any_class, method)
        return (
            callable(method) and
            set(inspect.signature(method).parameters.keys()) == parameters
        )

    @classmethod
    def _has_all_valid_properties(cls, any_class: type, protocol: type) -> bool:
        properties = cls._get_all_protocol_properties(protocol)
        return hasattr(any_class, "__dataclass_fields__") and all(
            p in any_class.__dataclass_fields__ or (
                hasattr(any_class, p) and
                not callable(getattr(any_class, p))
            )
            for p in properties
        )

    @classmethod
    def _has_all_instance_attributes(cls, any_object: Any, protocol: type) -> bool:
        attributes = cls._get_all_protocol_properties(protocol)
        for attribute in attributes:
            if not hasattr(any_object, attribute) or callable(getattr(any_object, attribute)):
                return False
        return True
