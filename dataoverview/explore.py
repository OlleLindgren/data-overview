from __future__ import annotations

import itertools
import statistics
import string
from typing import Any, Iterable, Tuple

import numpy as np
import pandas as pd

pd.options.mode.use_inf_as_na = True
from termcolor import colored


def _is_numeric_datatype(datatype) -> bool:
    return pd.api.types.is_numeric_dtype(datatype)


def _get_na_rate(column: pd.Series) -> float:
    return np.count_nonzero(column.isna())


def _get_na_color(na_rate: float) -> str:

    if na_rate == 0:
        return "green"

    if na_rate < 0.2:
        return "yellow"

    return "red"


def _has_duplicates(column: pd.Series) -> bool:
    return len(set(column.values)) < len(column)


def _get_na_msg(na_rate: float) -> str:
    na_col = _get_na_color(na_rate)
    return colored(f"[{na_rate*100:.0f}% N/A]", na_col)


def _get_dupl_msg(column: pd.Series) -> str:

    if _has_duplicates(column):
        return colored("[D]", "yellow")

    return colored("[ ]", "blue")


def _get_unique_color(n_unique: int) -> str:

    if n_unique <= 1:
        return "red"
    if n_unique < 10:
        return "blue"
    if n_unique < 25:
        return "yellow"

    return "red"


def _get_unique_msg(column: pd.Series) -> str:
    n_unique = len(set(column.values))
    return colored(f" [{n_unique} cls]", _get_unique_color(n_unique))


def _get_example(column: pd.Series, na_rate: float) -> Any:

    if na_rate == 0:
        return statistics.mode(column.values)
    if na_rate < 1:
        return statistics.mode(column[~column.isna()].values)

    return column.values[0]


def _get_standardized_range_msg(col_max, col_min, col_stddev) -> str:

    if col_stddev > 0:
        half_standardized_range = (col_max - col_min) / col_stddev / 2
    else:
        half_standardized_range = 0

    if 0.5 < half_standardized_range < 4:
        stdev_rng_col = "green"
    elif 0.2 < half_standardized_range < 5:
        stdev_rng_col = "yellow"
    else:
        stdev_rng_col = "red"

    return colored(f"{half_standardized_range:.1f}", stdev_rng_col)


def _get_range_msg(column: pd.Series, na_rate: float) -> str:

    if na_rate > 0:
        col_stddev = np.std(column[np.isfinite(column)])
        mean = np.mean(column[np.isfinite(column)])
    else:
        col_stddev = np.std(column)
        mean = np.mean(column)

    std_range_msg = _get_standardized_range_msg(column.max(), column.min(), col_stddev)

    return f"[{mean:.2e}±{std_range_msg}*{col_stddev:.2e}]"


def _get_max_msg(column: pd.Series) -> str:

    col_max = column.max()

    if np.isposinf(col_max):
        return colored("+∞", "red")

    return f"{col_max:.2e}"


def _get_min_msg(column: pd.Series) -> str:

    col_min = column.min()

    if np.isneginf(col_min):
        return colored("-∞", "red")

    return f"{col_min:.2e}"


def _summarize_column(column: pd.Series) -> Tuple:
    na_rate = _get_na_rate(column)

    if _is_numeric_datatype(column.dtype):
        return {
            "name": column.name,
            "dtype": column.dtype,
            "na": _get_na_msg(na_rate),
            "duplicated": _get_dupl_msg(column),
            "unique": _get_unique_msg(column),
            "range": f"[{_get_min_msg(column)}:{_get_max_msg(column)}]",
            "stddev": _get_range_msg(column, na_rate),
            "example": " " * 6 + str(_get_example(column, na_rate)),
        }

    return {
        "name": column.name,
        "dtype": column.dtype,
        "na": _get_na_msg(na_rate),
        "duplicated": _get_dupl_msg(column),
        "unique": _get_unique_msg(column),
        "range": "",
        "stddev": "",
        "example": str(_get_example(column, na_rate)),
    }


def summarize(df: pd.DataFrame) -> str:
    """Generate a readable summary of the columns of a pd.DataFrame.

    Args:
        df (pd.DataFrame): Dataframe to generate a summary for

    Returns:
        str: Generated summary
    """
    column_summaries = [_summarize_column(df[column]) for column in df.columns]

    summary_items = list(column_summaries[0].keys())

    def printable_len(cell_value: Any) -> int:
        """How many characters wide a value will be when printed to the terminal.

        Args:
            cell_value (Any): value to find width of

        Returns:
            int: The width the value will be when printed to the terminal
        """
        return sum(char in string.printable for char in str(cell_value))

    column_lengths = {
        column: max(printable_len(summary[column]) for summary in column_summaries)
        for column in summary_items
    }

    rows = []
    for summary in column_summaries:
        row_values = []
        for column in summary_items:
            cell_value = str(summary[column])
            row_values.append(
                cell_value + " " * (column_lengths[column] - printable_len(cell_value))
            )
        rows.append("".join(row_values))

    return "\n".join(rows)


def connect(
    dataframes: Iterable[pd.DataFrame], names: Iterable[str] = None
) -> pd.DataFrame:
    """Infer foreign key-like relationships between different dataframes.

    Args:
        datasets (Iterable[pd.DataFrame]): _description_
        names (Iterable[str]): _description_

    Returns:
        pd.DataFrame: _description_
    """
    if names is None:
        names = map(str, range(1_000_000_000))

    colummn_mapping = {
        name: dataframe.dtypes.to_dict() for dataframe, name in zip(dataframes, names)
    }

    columns = set(
        itertools.chain.from_iterable(
            mapping.keys() for mapping in colummn_mapping.values()
        )
    )

    connections = pd.DataFrame(columns=columns, index=colummn_mapping.keys())

    for name, mapping in colummn_mapping.items():
        for column in columns:
            dtype = mapping.get(column, None)
            dtype = str(dtype)
            if dtype == "None":
                dtype = ""
            if dtype == "string":
                dtype = "str"
            connections.loc[name, column] = dtype

    return connections
