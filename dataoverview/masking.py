from typing import Callable
import pandas as pd
import numpy as np
from collections import Counter, namedtuple
from termcolor import colored

MIN_GROUP = .001
COL_SPACING = 3

MaskEntry = namedtuple('MaskEntry', ['symbol', 'color'])

__default_entries = {
    "POSINF": MaskEntry('+', 'red'), 
    "NEGINF": MaskEntry('-', 'blue'), 
    "NAN": MaskEntry('N', 'yellow'), 
    "DEFAULT": MaskEntry('0', 'white'), 
    "PASS": MaskEntry('*', 'blue'), 
    "FAIL": MaskEntry('X', 'red')}

MaskEntries = namedtuple('Entries', __default_entries)(**__default_entries)

def __mask_format(mask_string: str) -> str:
    color = 'white'
    last_color = 'white'
    result = ''
    accum_result = ''
    for symbol in mask_string:
        # Define symbol and last_color
        for entry in MaskEntries:
            if symbol == entry.symbol:
                color = entry.color
        
        # If new color, print last
        if color != last_color and len(accum_result) > 0:
            result += colored(accum_result, last_color)
            accum_result = ''
        
        # Update last_color and accum_result with current data
        last_color = color
        accum_result += symbol

    result += colored(accum_result, last_color)

    return result

def __item_format(item: object, length: int) -> str:
    result = str(item)
    return result + ' '*(length-len(result))

def __df_format(masking_result: pd.DataFrame) -> str:
    max_ix_len = max(map(len, masking_result.index.astype(str)))
    max_count_len = max(map(len, masking_result['count'].astype(str)))
    min_count = min(masking_result['count'])
    return '\n'.join((
        __item_format(ix, max_ix_len) + ' '*COL_SPACING+
        __mask_format(mask_string) + ' '*COL_SPACING+
        __item_format(count, max_count_len) + ' '*COL_SPACING+
        '+'*int(5*(np.log(1+count)-np.log(1+min_count)))
        for ix, mask_string, count in zip(masking_result.index, masking_result['mask'], masking_result['count'])
    ))

def __default_float_mask(number: float) -> str:
    if np.isposinf(number):
        return MaskEntries.POSINF.symbol
    if np.isneginf(number):
        return MaskEntries.NEGINF.symbol
    if np.isnan(number):
        return MaskEntries.NAN.symbol
    return MaskEntries.DEFAULT.symbol

def __nan_mask(number: float) -> str:
    return MaskEntries.NAN.symbol if np.isnan(number) else MaskEntries.DEFAULT.symbol

def __inf_mask(number: float) -> str:
    return (MaskEntries.POSINF.symbol if number > 0 else MaskEntries.NEGINF.symbol) if np.isinf(number) else (MaskEntries.DEFAULT.symbol)

def __filter_factory(filtering_lambda: Callable[[object], bool]) -> Callable[[object], str]:
    return lambda _x: MaskEntries.PASS if filtering_lambda(_x) else MaskEntries.FAIL

def __agg_mask(row: pd.Series) -> str:
    return ''.join(row)

class __Counter:

    def __init__(self) -> None:
        self.current_group: int = 0
        self.last = None
        self.dict: Counter = Counter()

    def group(self, obj) -> int:
        if (obj!=self.last):
            self.current_group += 1
        self.last = obj
        self.dict[self.current_group] += 1
        return self.current_group

    def count(self, group: int) -> int:
        return self.dict[group]

def mask(df: pd.DataFrame, masking_function: Callable[[object], str]) -> str:
    # Mask a dataframe with an arbitrary elementwise function that accepts any element in the dataframe, and returns a 1 length string
    strmask = pd.DataFrame(index=df.index)
    strmask['mask'] = df.applymap(masking_function).apply(__agg_mask, axis=1)
    counter = __Counter()

    strmask['group'] = strmask['mask'].apply(counter.group)
    strmask['count'] = strmask['group'].apply(counter.count)
    strmask = strmask[strmask['group']!=strmask['group'].shift(1)]
    strmask = strmask[strmask['count'] > MIN_GROUP*df.shape[0]]

    return __df_format(strmask)

def na(df: pd.DataFrame) -> str:
    return mask(df, __nan_mask)

def inf(df: pd.DataFrame) -> str:
    return mask(df, __inf_mask)

def filter(df: pd.DataFrame, filtering_lambda) -> str:
    return mask(df, __filter_factory(filtering_lambda))
