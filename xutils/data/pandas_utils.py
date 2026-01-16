import os
from typing import Callable, List, Iterable, Union, Dict, Any, Optional, Hashable

import pandas as pd
import numpy as np
from pandas import Series

import xutils.core.file_utils as fu
import xutils.core.python_utils as pyu
import xutils.data.numpy_utils as nu


def from_data(data) -> pd.DataFrame:
    return pd.DataFrame(data)


def add_row(df: pd.DataFrame, row: Dict) -> pd.DataFrame:
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def load_if(load_fn: Callable[[], pd.DataFrame],
            save_path: str = None,
            if_older_than: Optional[int] = None,
            force_update: bool = False) -> pd.DataFrame:
    if save_path is None or force_update or fu.older_than(path=save_path, days=if_older_than):
        df = load_fn()
        if save_path is not None:
            write(df=df, xpath=save_path)
        return df
    else:
        return read(save_path)


def read(xpath: str, tail: Optional[int] = None, empty_if_none: bool = False, parse_dates=None) -> pd.DataFrame:
    if (not os.path.isfile(xpath) or os.path.getsize(xpath) == 0) and empty_if_none:
        return pd.DataFrame()
    else:
        df = pd.read_csv(xpath, parse_dates=parse_dates)
        return df.tail(tail) if tail is not None else df


def get_value(df: Union[str, pd.DataFrame],
              column: str,
              row: int = -1,
              where: str = None,
              value: str = None) -> Any:
    if isinstance(df, str):
        df = read(df)
    return df.iloc[row][column] if where is None \
        else df.loc[df[where] == value, column].values[0]


def get_value_where(df: Union[str, pd.DataFrame], column: str, other: str, value: str) -> Any:
    if isinstance(df, str):
        df = read(df)
    return df.loc[df[other] == value, column].values[0]


def get_values(df: Union[str, pd.DataFrame], *columns: str, row: int = -1) -> Dict[str, Any]:
    if isinstance(df, str):
        df = read(df)
    return {column: df.iloc[row][column] for column in columns}


def write(df: Union[Any, pd.DataFrame], xpath: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        df = from_data(df)
    fu.create_parent_dirs(xpath)
    df.to_csv(xpath, index=False)
    return df


def append_write(xpath: str, data: Dict) -> pd.DataFrame:
    df = read(xpath, empty_if_none=True)
    df_data = pd.DataFrame([data])
    df = pd.concat([df, df_data], ignore_index=True)
    return write(df, xpath)


def cast_to_int(df: pd.DataFrame, *columns):
    cast_to(df, 'int', *columns)
    return df


def cast_to_timestamp(df: pd.DataFrame, *columns):
    cast_to(df, 'datetime64[ns]', *columns)
    return df


def cast_to_timestamp_timezone(df: pd.DataFrame, timezone, *columns):
    cast_to(df, f'datetime64[ns, {timezone}]', *columns)
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


def find_null(df: pd.DataFrame,
              null_column,
              return_column):
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
        print(df.to_markdown(tablefmt='psql'))
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


def lower_case_columns(df: pd.DataFrame):
    df.columns = [str(col).lower() for col in df.columns]
    return df


def mean(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    return df[columns].mean(axis=1)


def add_mean(df: Union[str, pd.DataFrame],
             columns: List[str],
             column_name: str = "Average",
             out_path: str = None) -> pd.DataFrame:
    if isinstance(df, str):
        if out_path is None:
            out_path = df
        df = read(df)
    df[column_name] = mean(df=df, columns=columns)
    if out_path is not None:
        write(df, out_path)
    return df


def upper_threshold(df: pd.DataFrame,
                    column: str,
                    threshold: float = 0.8,
                    default_value: Any = "") -> pd.Series:
    return df[column].where(df[column].abs() >= threshold, default_value)


def add_upper_threshold(df: Union[str, pd.DataFrame],
                        column: str,
                        column_name: str = "Threshold",
                        threshold: float = 0.8,
                        default_value: Any = "",
                        out_path: str = None) -> pd.DataFrame:
    if isinstance(df, str):
        if out_path is None:
            out_path = df
        df = read(df)
    df[column_name] = upper_threshold(df=df, column=column, threshold=threshold, default_value=default_value)
    if out_path is not None:
        write(df, out_path)
    return df


def add_columns(df: pd.DataFrame, columns: Dict[str, pd.Series]) -> pd.DataFrame:
    for name, value in columns.items():
        df[name] = value

    return df


def add_calc_column(df: pd.DataFrame, column_name: str, calc_fn: Callable, cleanup: bool = False) -> pd.DataFrame:
    df[column_name] = calc_fn(df)
    if cleanup:
        drop_na(df)
    return df


def concat_unique(dfs: Iterable[pd.DataFrame]):
    df = pd.concat(dfs, axis=1)
    return df.loc[:, ~df.columns.duplicated()]


def concat_dicts(dfs: Iterable[Dict]):
    return pd.DataFrame.from_records(dfs)


def merge_all(dfs: Iterable[pd.DataFrame], on: str = None, how='outer'):
    merged = None
    for df in dfs:
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=on, how=how)

    return merged


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


def rows_cols(df: pd.DataFrame, row_start: int, row_end: int = None, cols: List[int] = None):
    return df.iloc[row_start:row_end, [df.columns.get_loc(column) for column in cols]]


def cols(df: pd.DataFrame, *cols: int):
    return df.loc[:, cols]


def quartiles(df: pd.DataFrame, column: str, points: List[float]):
    return df[column].quantile(points if points else [0.25, 0.5, 0.75]).values


def get_columns_by_type(df: pd.DataFrame, types):
    return df.select_dtypes(include=types)


def get_min_max(df: pd.DataFrame):
    return nu.get_min_max(df)


def scale_min_max(df: pd.DataFrame, min_max):
    df_scaled = df.copy()

    # apply normalization techniques
    for column, index in enumerate(df_scaled.columns):
        col_min = min_max[index][0]
        col_max = min_max[index][1]
        df_scaled[column] = (df_scaled[column] - col_min) / (col_max - col_min)

    return df_scaled


def invert_to_dict(df: pd.DataFrame) -> list[dict[Hashable, Any]]:
    return df.to_dict(orient="records")


class DataChain:

    def __init__(self, df: pd.DataFrame = None):
        super().__init__()
        self.df: pd.DataFrame = df
        self.dirty: bool = False

    #     self.last_data_cache = None
    #     self.data_invalid = False
    #
    #
    def load(self,
             load_fn: Callable[[], pd.DataFrame],
             save_path: str = None,
             update_if_older_than: Optional[int] = None,
             force_update: bool = False) -> pd.DataFrame:
        if save_path is None or force_update or fu.older_than(path=save_path, days=update_if_older_than):
            self.df = load_fn()
            if save_path is not None:
                write(df=self.df, xpath=save_path)
            self.dirty = True
        else:
            self.df = read(save_path)
            self.dirty = False

        return self.df

    def add_column(self,
                   column_name: str,
                   column_fn: Callable = None,
                   column_data=None,
                   save_path: str = None,
                   force_update: bool = False):
        pass

    def transform_df(self, cache_path: str):
        pass
