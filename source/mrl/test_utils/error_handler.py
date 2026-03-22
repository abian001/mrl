from typing import Callable
import sys


def describe_exception(error: BaseException) -> str:
    lines: list[str] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        description = _describe_single_exception(current)
        if len(lines) == 0:
            lines.append(description)
        else:
            lines.append(f"Caused by: {description}")
        next_error = current.__cause__
        if next_error is None and not current.__suppress_context__:
            next_error = current.__context__
        current = next_error
    return "\n".join(lines)


def _describe_single_exception(error: BaseException) -> str:
    if isinstance(error, KeyError):
        key = error.args[0] if len(error.args) > 0 else "<unknown>"
        return f"KeyError: missing key {key!r}."

    message = str(error).strip()
    if len(message) == 0:
        return type(error).__name__
    return f"{type(error).__name__}: {message}"


def try_main(main_function: Callable[[], None]):
    try:
        main_function()
    except Exception as error:  # pylint: disable=broad-exception-caught
        print(
            "An error was observed during the execution of the program:\n"
            f"{describe_exception(error)}"
        )
        sys.exit(1)
