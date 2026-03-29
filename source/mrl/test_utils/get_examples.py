import os
from importlib.resources import files, as_file
from pathlib import Path
import shutil
import argparse
from mrl.test_utils.error_handler import try_main


def main():
    try_main(_run)


def _run():
    args = _get_arguments()
    args.destination.mkdir(parents = True, exist_ok = True)

    example_directory = files("mrl").joinpath("examples")
    for item in example_directory.iterdir():
        item_name = item.name[1:] if item.name.startswith('_') else item.name
        path = args.destination / item_name
        if os.path.exists(path) and not args.overwrite:
            print(
                f"File {path} already exist. Skipping."
                "If you would like to force overwrite use "
                "the --overwrite option"
            )
        else:
            with as_file(item) as item_path:
                if item_path.is_file():
                    shutil.copy(item_path, path)


def _get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'destination',
        type = Path,
        default = Path("examples"),
        nargs = "?"
    )
    parser.add_argument(
        '--overwrite',
        action = 'store_true'
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
