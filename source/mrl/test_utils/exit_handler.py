from typing import Callable
import sys

def try_main(main_function: Callable[[], None]):
    try:
        main_function()
    except Exception as error:  # pylint: disable=broad-exception-caught
        print(f"An error was observed during the execution of the program:\n{error}")
        sys.exit(1)
