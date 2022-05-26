"""Main entrypoint to the dataoverview package."""
import argparse
from pathlib import Path
from typing import Sequence

import dataoverview


def _parse_args(command_line_arguments: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        command_line_arguments (Sequence[str]): Arguments to parse

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "path",
        help="Path to a file or directory to explore",
        type=Path,
    )
    parser.add_argument(
        "-s",
        "--summarize",
        help="Display a summary of a dataset",
        action="store_true",
    )
    parser.add_argument(
        "-c",
        "--connect",
        help="Display a summary of the inter-dataframe relationships within a dataset",
        action="store_true",
    )
    return parser.parse_args(command_line_arguments)


def main(args: Sequence[str]) -> int:
    """Run either summarize or connect (or both). Normally used by python -m dataoverview ...

    Args:
        args (Sequence[str]): Command line arguments

    Returns:
        int: 0 if successful
    """
    args = _parse_args(args)

    if args.summarize:
        for line in dataoverview.explore.summarize_from_path(args.path):
            print(line)  # noqa: WPS421

    if args.connect:
        print(dataoverview.explore.connect_from_path(args.path))  # noqa: WPS421

    return 0
