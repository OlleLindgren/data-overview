"""Logic for working with paths."""
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


def iter_dataframes(path: Path) -> Iterable[Tuple[pd.DataFrame, Path]]:
    """Iterate over all files in a directory, and try to convert each to a pd.DataFrame.

    Args:
        path (Path): Path to iterate over

    Raises:
        ValueError: If the provided path is not a directory

    Yields:
        Tuple[pd.DataFrame, Path]: A pd.DataFrame, and the file it was parsed from
    """
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory.")
    for filename in path.glob("*.csv"):
        try:
            dataframe = pd.read_csv(filename)
        except (IndexError, pd.errors.EmptyDataError):
            continue
        if min(dataframe.shape) == 0:
            continue
        yield dataframe, filename
