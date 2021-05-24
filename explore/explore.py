from typing import Iterable, Tuple
import pandas as pd 
import numpy as np 
from termcolor import colored

def _col_summary(col: pd.Series) -> Tuple:
    # Summary of column contents. Not meant to be run independency, but rather as a backend to summarize

    is_numeric = np.issubdtype(col.dtype, np.number)

    col_na = col.apply(lambda x: x in (None, 'nan', 'None'))

    has_na = np.any(col_na)

    if has_na:
        n_unique = len(np.unique(col[~col_na]))
    else:
        n_unique = len(np.unique(col))
    is_dupl = n_unique != len(col)

    if has_na:
        na_rate = sum(col_na) / len(col)
        if na_rate < .2:
            na_col = 'yellow'
        else:
            na_col = 'red'
        na_msg = colored(f"[{na_rate*100:.0f}% N/A]", na_col)
    else:
        na_msg = colored(f"[0% N/A]", 'green')

    if is_dupl:
        is_dupl_msg = colored('[D]', 'yellow')
    else:
        is_dupl_msg = colored('[U]', 'blue')
    if n_unique == 1:
        unique_col = 'red'
    elif n_unique < 10:
        unique_col = 'blue'
    elif n_unique < 25:
        unique_col = 'yellow'
    else:
        unique_col = 'red'
    unique_msg = colored(f' [{n_unique} cls]', unique_col)
    
    if is_numeric:

        example = col[~col_na].values[0]

        col_min = col.min()
        col_max = col.max()
        has_inf = np.isinf(col_max)
        has_neginf = np.isinf(col_min)

        if has_inf:
            max_msg = colored('+∞', 'red')
        else:
            max_msg = f'{col_max:.2e}'
        if has_neginf:
            min_msg = colored('-∞', 'red')
        else:
            min_msg = f'{col_min:.2e}'
        if has_inf or has_neginf or has_na:
            stdev = np.std(col[np.isfinite(col)])
            mean = np.mean(col[np.isfinite(col)])
            mean_msg = colored(f'{mean:.2e}', 'yellow')
        else:
            stdev = np.std(col)
            mean = np.mean(col)
            mean_msg = f'{mean:.2e}'
        
        stdev_range = (col_max - col_min) / stdev if stdev > 0 else 0
        if 2 < stdev_range < 5:
            stdev_rng_col = 'green'
        elif 1 < stdev_range < 6:
            stdev_rng_col = 'yellow'
        else:
            stdev_rng_col = 'red'
        stdev_rng_msg = colored(f'{stdev_range:.1f}', stdev_rng_col)

        name = col.name
        dtype = col.dtype
        range_msg = f'[{min_msg}:{max_msg}]'
        stdev_msg = f'[{mean_msg}±{stdev_rng_msg}*{stdev:.2e}]'

        return name, dtype, na_msg, is_dupl_msg, unique_msg, range_msg, stdev_msg, example
        # return f'{col.name} | [{col.dtype}]{na_msg} [{min_msg}:{max_msg}] [{mean_msg}±{stdev_rng_msg}*{stdev:.2f}] | {example}'

    else:
        example = col[~col_na].values[0]

        name = col.name
        dtype = col.dtype
        range_msg = is_dupl_msg
        stdev_msg = unique_msg

        return name, dtype, na_msg, is_dupl_msg, unique_msg, '', '', example
        # return f'{col.name} | [{len(col)}*{col.dtype}]{na_msg}{is_dupl_msg}{unique_msg} | {example}'

def summarize(df: pd.DataFrame) -> str:
    # Summary of dataframe contents

    cols = ['name', 'dtype', 'na', 'dupl', 'unique', 'range', 'stdev', 'example']
    data = []
    for col in df.columns:
        data.append(_col_summary(df[col]))
    result = pd.DataFrame(data=data, columns=cols)

    # Pretty print report
    asstr = ''
    printable = r'abcdefghijklmnopqrstuvwxyzåäö'
    printable += printable.upper()
    printable += r'/ _.![]' + '0123456789' + r'±∞'
    printable_len = lambda x: sum([_x in printable for _x in str(x)])
    col_lengths = [result[col].apply(printable_len).max() for col in result.columns]
    for ix in result.index:
        for col, col_len in zip(result.columns, col_lengths):
            cell_value = str(result.loc[ix, col])
            asstr += cell_value + ' '*(col_len - printable_len(cell_value)) + ' '
        asstr += '\n'
    return asstr

def connect(datasets: Iterable[pd.DataFrame], names: Iterable[str]) -> pd.DataFrame:
    assert len(datasets) >= 2

    all_cols = []
    dtypes = []
    for df in datasets:
        all_cols.extend(df.columns)
        dtypes.extend(df.dtypes.values)
    result = pd.DataFrame(index=all_cols)
    result['dtype'] = dtypes
    for col in all_cols:
        for df, name in zip(datasets, names):
            result.loc[col, name] = '+' if col in df.columns else ''
    return result
