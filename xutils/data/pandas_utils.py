import os
import pandas as pd
import numpy as np


def read(*paths) -> pd.DataFrame:
    path = paths[0] if len(paths) == 0 else os.path.join(*paths)
    return pd.read_csv(path)


def write(df: pd.DataFrame, path):
    df.to_csv(path, index=False)


def cast_to_int(df: pd.DataFrame, *columns):
    cast_to(df, 'int', *columns)


def cast_to_float(df: pd.DataFrame, *columns):
    cast_to(df, 'float', *columns)


def cast_to_float32(df: pd.DataFrame, *columns):
    cast_to(df, 'float32', *columns)


def cast_to_date(df: pd.DataFrame, *columns):
    cast_to(df, 'date', *columns)


def cast_to(df: pd.DataFrame, cast_type, *columns):
    cast_type = cast_type.lower()
    for column in columns:
        df_column = df[column]
        if cast_type == "date":
            df[column] = pd.to_datetime(df_column)
        else:
            df[column] = df_column.astype(cast_type)
                    

def normalize_max(df: pd.DataFrame, *columns):
    for column in columns:
        df[column] = df[column] / max(df[column])


def fill_with_greater_than(df: pd.DataFrame, find_column, find_value, value_column, cutoff_value, *fields):
    query_data = df.query(f"{find_column} == '{find_value}' and {value_column} == '{cutoff_value}'")
    for field in fields:
        df.loc[(df[find_column] == find_value) &
               (df[value_column] > cutoff_value), field] = query_data[value_column].values[0]


def fill_null(df: pd.DataFrame, *columns, default_value=0,
              fill_median=False, fill_mean=False, fill_back=False, fill_forward=False):
    for column in columns:
        df_column = df[column]
        fill_method = None

        if fill_median:
            value = df_column.median()
            if value is not None:
                default_value = value
        elif fill_mean:
            value = df_column.mean()
            if value is not None:
                default_value = value
        elif fill_forward:
            fill_method = 'ffill'
        elif fill_back:
            fill_method = 'bfill'

        df_column.fillna(default_value, inplace=True, method=fill_method)

        # column_values = df[column]
        # df.loc[column_values.isnull(), column] = default_value


def find_null(df: pd.DataFrame, null_column, return_column):
    return df.loc[df[null_column].isnull(), return_column].unique()


def find_null_columns(df: pd.DataFrame):
    return df.columns[df.isnull().any()]


def find_null_counts(df: pd.DataFrame):
    return df[find_null_columns(df)].isnull().sum()


def find_null_rows(df: pd.DataFrame):
    null_columns = find_null_columns(df)
    return df[df.isnull().any(axis=1)][null_columns]


def find_columns_like(df: pd.DataFrame, like):
    return [c for c in df.columns if like in c]


def create(*columns):
    return pd.DataFrame(columns=columns)


def split(df: pd.DataFrame, percent):
    split_position = int(percent * len(df))
    return df[:split_position], df[split_position:]


def split_column(df: pd.DataFrame, column, percent=0.8):
    split_position = int(percent * len(df))
    return df[column][:split_position], df[column][split_position:]


def describe_data(df: pd.DataFrame):
    df.info()
    print('Shape of data frame:')
    print(df.shape)
    for col in df.columns:
        print("Unique number of values in ")
        print(col)
        print(df.loc[:,col].nunique())
    print("number of null values present in each column")
    print(df.isnull().sum())
    print(df.head(5))


def root_mean_squared_error(value, predicted):
    return ((value - predicted) ** 2).mean() ** .5


def moving_average(df: pd.DataFrame, column, new_column=None, time_frame=3, shift=1):
    return df[column if new_column is None else new_column].rolling(time_frame).mean().shift(shift)


def split_dataframe(df: pd.DataFrame, chunk_size = 10000):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks


def shape(df: pd.DataFrame):
    return len(df.index), len(df.columns)


def get_column_index(df: pd.DataFrame, column):
    return df.columns.get_loc(column)


def print_all(df: pd.DataFrame):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)


def drop_na(df: pd.DataFrame):
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)


def sort(df: pd.DataFrame, column):
    df.sort_values(column, inplace=True)
    df.reset_index(drop=True, inplace=True)


def reduce_memory_usage(df: pd.DataFrame, verbose=False):
    start_memory = None
    if verbose:
        start_memory = df.memory_usage().sum() / 1024**2
        print(f"Memory usage of dataframe is {start_memory} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    pass
        else:
            df[col] = df[col].astype('category')

    if verbose:
        end_memory = df.memory_usage().sum() / 1024**2
        print(f"Memory usage of dataframe after reduction {end_memory} MB")
        print(f"Reduced by {100 * (start_memory - end_memory) / start_memory} % ")
    return df


def get_correlation(df, d1, d2, spearman=False, kendall=False, verbose=False):
    result = df[d1].corr(df[d2], method='spearman' if spearman else 'kendall' if kendall else 'pearson')
    if verbose:
        print('Result:', result,
              '.  There is a ',
              'Strong' if abs(result) > 0.5 else 'Weak',
              'Positive' if result > 0 else 'Negative',
              'Correlation')
    return result


def lower_case_columns(df):
    df.columns = df.columns.str.lower()
