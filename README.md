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
`pip install git+https://github.com/OlleLindgren/data-overview@v0.1`

## Usage

```
import explore as dex

df = pd.read_csv(dir)

# df summary
print(df_summary(df))

dfs = [pd.read_csv(dir) for dir in dirs]

# multiple df connect
print(connect(dfs))
```