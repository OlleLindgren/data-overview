from __future__ import annotations

import itertools
import statistics
from typing import Any, Iterable, Tuple

import numpy as np
import pandas as pd
from termcolor import colored

pd.options.mode.use_inf_as_na = True


def _is_numeric_datatype(datatype) -> bool:
    return pd.api.types.is_float_dtype(datatype)


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

    if not pd.api.types.is_numeric_dtype(column.dtype):
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

    if pd.api.types.is_float_dtype(column.dtype):
        range_format = "[{min:8.7g} : {max:8.7g}]"
        stddev_range_format = "[{mean:10.5g} ± {std_range:6.5g} * {stddev:10.5g}]"
    elif pd.api.types.is_integer_dtype(column.dtype):
        range_format = "[{min:8d} : {max:8d}]"
        stddev_range_format = "[{mean:10.5g} ± {std_range:6.5g} * {stddev:10.5g}]"
    elif pd.api.types.is_bool_dtype(column.dtype):
        range_format = "[{min} : {max}]"
        stddev_range_format = ""
    else:
        raise NotImplementedError(f"Column datatype {column.dtype} is not implemented")

    return {
        "name": column.name,
        "dtype": column.dtype,
        "na": _get_na_msg(na_rate),
        "duplicated": _get_dupl_msg(column),
        "unique": _get_unique_msg(column),
        "range": range_format.format(min=column.min(), max=column.max()),
        "stddev": stddev_range_format.format(
            mean=column.mean(),
            std_range=(column.max() - column.min()) / column.std(),
            stddev=column.std(),
        )
        if not pd.api.types.is_bool_dtype(column.dtype)
        else "",
        "example": str(_get_example(column, na_rate)),
    }


def summarize(dataframe: pd.DataFrame) -> str:
    """Generate a readable summary of the columns of a pd.DataFrame.

    Args:
        dataframe (pd.DataFrame): Dataframe to generate a summary for

    Returns:
        str: Generated summary
    """
    column_summaries = [
        _summarize_column(dataframe[column]) for column in dataframe.columns
    ]

    summary_items = list(column_summaries[0].keys())

    column_widths = {
        column: 1 + max(len(str(summary[column])) for summary in column_summaries)
        for column in summary_items
    }

    rows = []
    for summary in column_summaries:
        rows.append(
            "".join(
                str(summary[column]).ljust(column_widths[column])
                for column in summary_items
            )
        )

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
