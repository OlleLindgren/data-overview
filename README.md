# data-overview

Repository with a few small helper functions for quickly overviewing data

## Dependencies

```
python>=3.6
pandas
numpy
termcolor
```

## Install

`pip install git+https://github.com/OlleLindgren/data-overview@v0.2`

## Usage

```python
import dataoverview

# Get some data source. data_walk() -> Iterable[pd.DataFrame]
from some_data_source import data_walk

df = next(data_walk())

# df summary
print(dataoverview.explore.summarize(df))

dfs = [df for df in data_walk()]
names = ['df '+str(i) for i in range(len(dfs))]

# See which columns multiple dataframes have in common they have in common
print(dataoverview.explore.connect(dfs, names))

# String representation of parts in dataframe that are nan
# Very useful when joining data from different sources
print(dataoverview.masking.na(df))

# String representation of parts in dataframe that pass some filter
print(dataoverview.masking.filter(df, lambda x: x > 0))
```
