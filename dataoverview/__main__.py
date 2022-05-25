import argparse
from pathlib import Path
import sys
from typing import Iterable, Sequence

import pandas as pd


import dataoverview.explore


def _parse_args(command_line_arguments: Sequence[str]) -> argparse.Namespace:

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


def _iter_dataframes(path: Path) -> Iterable[pd.DataFrame]:
    for filename in path.iterdir():
        if filename.is_file():
            try:
                df = pd.read_csv(filename)
            except (IndexError, pd.errors.EmptyDataError):
                continue
            if not len(df) or not len(df.columns):
                continue
            yield df, filename


def main(args: Sequence[str]) -> int:
    """Main function. Used by python -m dataoverview ...

    Args:
        args (Sequence[str]): Command line arguments

    Returns:
        int: 0 if successful
    """
    args = _parse_args(args)

    if args.summarize:
        if args.path.is_file():
            summary = dataoverview.explore.summarize(pd.read_csv(args.path).convert_dtypes())
            print(summary)
        elif args.path.is_dir():
            count = 0
            for dataframe, filename in _iter_dataframes(args.path):
                print(filename.relative_to(args.path))
                summary = dataoverview.explore.summarize(dataframe.convert_dtypes())
                print(summary)
                count += 1
            if count == 0:
                print(f"Unable to extract a pd.DataFrame from {args.path}")
                return 1
        else:
            print(f"{args.path} does not exist.")
            return 1

    if args.connect:
        dataframe_iterator, filename_iterator = zip(*_iter_dataframes(args.path))

        filename_iterator = (filename.relative_to(args.path) for filename in filename_iterator)
        dataframe_iterator = (dataframe.convert_dtypes() for dataframe in dataframe_iterator)

        print(dataoverview.explore.connect(dataframe_iterator, filename_iterator))

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
