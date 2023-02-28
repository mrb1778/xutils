import os
from typing import Callable

import pandas as pd
import numpy as np

import xutils.core.file_utils as fu
import xutils.core.python_utils as pyu
import xutils.data.numpy_utils as nu


def read(*paths) -> pd.DataFrame:
    return pd.read_csv(os.path.join(*paths))


def write(df: pd.DataFrame, *paths):
    path = os.path.join(*paths)
    fu.create_parent_dirs(path)
    df.to_csv(path, index=False)
    return df


def cast_to_int(df: pd.DataFrame, *columns):
    cast_to(df, 'int', *columns)
    return df


def cast_to_float(df: pd.DataFrame, *columns):
    cast_to(df, 'float', *columns)
    return df


def cast_to_float32(df: pd.DataFrame, *columns):
    cast_to(df, 'float32', *columns)
    return df


def cast_to_date(df: pd.DataFrame, *columns):
    cast_to(df, 'date', *columns)
    return df


def cast_to(df: pd.DataFrame, cast_type, *columns):
    cast_type = cast_type.lower()
    for column in columns:
        df_column = df[column]
        if cast_type == "date":
            df[column] = pd.to_datetime(df_column)
        else:
            df[column] = df_column.astype(cast_type)
    return df


def normalize_max(df: pd.DataFrame, *columns):
    for column in columns:
        df[column] = df[column] / max(df[column])
    return df


def fill_with_greater_than(df: pd.DataFrame, find_column, find_value, value_column, cutoff_value, *fields):
    query_data = df.query(f"{find_column} == '{find_value}' and {value_column} == '{cutoff_value}'")
    for field in fields:
        df.loc[(df[find_column] == find_value) &
               (df[value_column] > cutoff_value), field] = query_data[value_column].values[0]
    return df


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
    return df


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
    print("Unique values:")
    pyu.print_dict({col: df.loc[:, col].nunique() for col in df.columns})
    print("Null Count:")
    print(df.isnull().sum())
    print("First 5:")
    print(df.head(5))
    return df


def root_mean_squared_error(value, predicted):
    return ((value - predicted) ** 2).mean() ** .5


def moving_average(df: pd.DataFrame, column, new_column=None, time_frame=3, shift=1):
    return df[column if new_column is None else new_column].rolling(time_frame).mean().shift(shift)


def split_dataframe(df: pd.DataFrame, chunk_size=10000):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size:(i + 1) * chunk_size])
    return chunks


def shape(df: pd.DataFrame):
    return len(df.index), len(df.columns)


def get_column_index(df: pd.DataFrame, column):
    return df.columns.get_loc(column)


def print_all(df: pd.DataFrame):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, "display.width", None):
        print(df)
    return df


def drop_na(df: pd.DataFrame, verbose=False):
    if verbose:
        print(df.isna().sum())
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def sort(df: pd.DataFrame, column, set_index=False, asc=True):
    if set_index:
        df.set_index(column, drop=False, inplace=True)
        df.sort_index(inplace=True, ascending=asc)
    else:
        df.sort_values(column, inplace=True, ascending=asc)
        df.reset_index(drop=True, inplace=True)
    return df


def reduce_memory_usage(df: pd.DataFrame, verbose=False):
    start_memory = None
    if verbose:
        start_memory = df.memory_usage().sum() / 1024 ** 2
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
        end_memory = df.memory_usage().sum() / 1024 ** 2
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
    return df


def add_calc_column(df, column_name, calc_fn, cleanup=False):
    df[column_name] = calc_fn(df)
    if cleanup:
        drop_na(df)
    return df


def concat_unique(dfs):
    df = pd.concat(dfs, axis=1)
    return df.loc[:, ~df.columns.duplicated()]


def read_enrich_write(source_data_path: str,
                      save_path: str,
                      enrich_fn: Callable[[pd.DataFrame], pd.DataFrame],
                      post_loader_fn: Callable[[pd.DataFrame], pd.DataFrame] = None,
                      update_if_older: bool = True,
                      force_update: bool = False):
    def enrich_fn_wrapper(path):
        df = pd.read_csv(source_data_path)
        if post_loader_fn is not None:
            post_loader_fn(df)

        enrich_result = enrich_fn(df)
        if enrich_result is not None:
            df = enrich_result

        df.to_csv(path, index=False)
        return path

    return fu.create_file_if(save_path,
                             enrich_fn_wrapper,
                             update=force_update or (update_if_older and
                                                     (not fu.exists(save_path) or
                                                      fu.modified_after(source_data_path, save_path))))


def rows_cols(df, row_start, row_end=None, cols=None):
    return df.iloc[row_start:row_end, [df.columns.get_loc(column) for column in cols]]


def cols(df, *cols):
    return df.loc[:, cols]


def quartiles(df, column, points):
    return df[column].quantile(points if points else [0.25, 0.5, 0.75]).values


def get_columns_by_type(df, types):
    return df.select_dtypes(include=types)


def get_min_max(df):
    return nu.get_min_max(df)


def scale_min_max(df, min_max):
    df_scaled = df.copy()

    # apply normalization techniques
    for column, index in enumerate(df_scaled.columns):
        col_min = min_max[index][0]
        col_max = min_max[index][1]
        df_scaled[column] = (df_scaled[column] - col_min) / (col_max - col_min)

    return df_scaled
