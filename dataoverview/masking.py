"""Quickly visualizing a dataset based on applying masks to it and looking for segments."""

from collections import Counter, namedtuple
from typing import Callable

import numpy as np
import pandas as pd
from termcolor import colored

MIN_GROUP = 0.001
COL_SPACING = 3

MaskEntry = namedtuple("MaskEntry", ["symbol", "color"])

_default_entries = {
    "POSINF": MaskEntry("+", "red"),
    "NEGINF": MaskEntry("-", "blue"),
    "NAN": MaskEntry("N", "yellow"),
    "DEFAULT": MaskEntry("0", "white"),
    "PASS": MaskEntry("*", "blue"),
    "FAIL": MaskEntry("X", "red"),
}

MaskEntries = namedtuple("Entries", _default_entries)(**_default_entries)


def _mask_format(mask_string: str) -> str:
    color = "white"
    last_color = "white"
    result = ""
    accum_result = ""
    for symbol in mask_string:
        # Define symbol and last_color
        for entry in MaskEntries:
            if symbol == entry.symbol:
                color = entry.color

        # If new color, print last
        if color != last_color and len(accum_result) > 0:
            result += colored(accum_result, last_color)
            accum_result = ""

        # Update last_color and accum_result with current data
        last_color = color
        accum_result += symbol

    result += colored(accum_result, last_color)

    return result


def _item_format(item: object, length: int) -> str:
    result = str(item)
    return result + " " * (length - len(result))


def _df_format(masking_result: pd.DataFrame, index_spacing: int) -> str:
    max_count_len = max(map(len, masking_result["count"].astype(str)))
    min_count = min(masking_result["count"])
    return "\n".join(
        (
            _item_format(ix, index_spacing)
            + " " * COL_SPACING
            + _mask_format(mask_string)
            + " " * COL_SPACING
            + _item_format(count, max_count_len)
            + " " * COL_SPACING
            + "+" * int(5 * (np.log(1 + count) - np.log(1 + min_count)))
            for ix, mask_string, count in zip(
                masking_result.index, masking_result["mask"], masking_result["count"]
            )
        )
    )


def _df_head_format(dataframe: pd.DataFrame, index_spacing: int) -> str:
    # Generate a header for a dataframe
    col_marker = "*"
    line_marker = "'"
    line_space_marker = line_marker + " "

    alternate_color = "blue"
    header = "\n".join(
        (
            (
                colored(_item_format(col_name, index_spacing), alternate_color)
                if i % 2 == 0
                else _item_format(col_name, index_spacing)
            )
            + " " * COL_SPACING
            + colored(
                line_space_marker * (i // 2) + line_marker * (i - 2 * (i // 2)),
                alternate_color,
            )
            + (colored(col_marker, alternate_color) if i % 2 == 0 else col_marker)
            + " " * (dataframe.shape[1] - i - 1)
            for i, col_name in enumerate(dataframe.columns)
        )
    )
    divider_line = colored(
        line_space_marker * (dataframe.shape[1] // 2)
        + line_marker * (dataframe.shape[1] - 2 * (dataframe.shape[1] // 2)),
        alternate_color,
    )
    pre_spacing = " " * index_spacing + " " * COL_SPACING
    divider = pre_spacing + divider_line
    return header + "\n" + divider


def __nan_mask(number: float) -> str:
    return MaskEntries.NAN.symbol if np.isnan(number) else MaskEntries.DEFAULT.symbol


def _inf_mask(number: float) -> str:
    return (
        (MaskEntries.POSINF.symbol if number > 0 else MaskEntries.NEGINF.symbol)
        if np.isinf(number)
        else (MaskEntries.DEFAULT.symbol)
    )


def _agg_mask(row: pd.Series) -> str:
    return "".join(row)


class _Counter:
    """I have classes. Make it go away. TODO."""

    def __init__(self) -> None:
        self.current_group: int = 0
        self.last = None
        self.dict: Counter = Counter()

    def group(self, obj) -> int:
        if obj != self.last:
            self.current_group += 1
        self.last = obj
        self.dict[self.current_group] += 1
        return self.current_group

    def count(self, group: int) -> int:
        return self.dict[group]


def mask(dataframe: pd.DataFrame, masking_function: Callable[[object], str]) -> str:
    # Mask a dataframe with an arbitrary elementwise function that accepts any
    # element in the dataframe, and returns a 1 length string
    strmask = pd.DataFrame(index=dataframe.index)
    strmask["mask"] = dataframe.applymap(masking_function).apply(_agg_mask, axis=1)
    counter = _Counter()

    strmask["group"] = strmask["mask"].apply(counter.group)
    strmask["count"] = strmask["group"].apply(counter.count)
    strmask = strmask[strmask["group"] != strmask["group"].shift(1)]

    # TODO merge consecutive small groups into one:
    # ##########*************
    # #####***#*******##*****
    # #####******************
    # ###############********
    # -> (always missing | always complete | can be both) -> (## | ** | #*) ->
    # ######*###******#******
    # #####*#*#********#*****
    strmask = strmask[strmask["count"] > MIN_GROUP * dataframe.shape[0]]

    max_col_name_len = max(map(len, dataframe.columns))
    max_ix_name_len = max(map(len, strmask.index.astype(str)))
    index_spacing = max(max_ix_name_len, max_col_name_len)

    return _df_head_format(dataframe, index_spacing) + "\n" + _df_format(strmask, index_spacing)


def na(dataframe: pd.DataFrame) -> str:
    return mask(dataframe, __nan_mask)


def inf(dataframe: pd.DataFrame) -> str:
    return mask(dataframe, _inf_mask)


def filter(dataframe: pd.DataFrame, filtering_lambda) -> str:
    return mask(
        dataframe,
        lambda value: MaskEntries.PASS.symbol
        if filtering_lambda(value)
        else MaskEntries.FAIL.symbol,
    )
