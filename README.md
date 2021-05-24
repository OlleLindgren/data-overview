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
import explore as dex

df = pd.read_csv(dir)

# df summary
print(dex.summarize(df))

dfs = [pd.read_csv(dir) for dir in dirs]

# multiple df connect
print(dex.connect(dfs, dirs))
```
