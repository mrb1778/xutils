import pandas as pd


def cast_to_int(df, *fields):
    cast_to(df, 'int', *fields)


def cast_to_float(df, *fields):
    cast_to(df, 'float', *fields)


def cast_to_date(df, *fields):
    cast_to(df, 'date', *fields)


def cast_to(df, cast_type, *fields):
    cast_type = cast_type.lower()
    for field in fields:
        if cast_type == "date":
            df[field] = pd.to_datetime(df[field])
        else:
            df[field] = df[field].astype(cast_type)
                    

def normalize_max(df, *fields):
    for field in fields:
        df[field] = df[field] / max(df[field])


def fill_with_greater_than(df, find_column, find_value, value_column, cutoff_value, *fields):
    query_data = df.query(f"{find_column} == '{find_value}' and {value_column} == '{cutoff_value}'")
    for field in fields:
        df.loc[(df[find_column] == find_value) &
               (df[value_column] > cutoff_value), field] = query_data[value_column].values[0]


def fill_null_with_median(df, *columns):
    for column in columns:
        column_values = df[column]
        df.loc[column_values.isnull(), column] = column_values.median()


def fill_null(df, default_value=0, *columns):
    for column in columns:
        column_values = df[column]
        df.loc[column_values.isnull(), column] = default_value


def find_null(df, null_column, return_column):
    return df.loc[df[null_column].isnull(), return_column].unique()


def find_null_columns(df):
    return df.columns[df.isnull().any()]


def find_null_counts(df):
    return df[find_null_columns(df)].isnull().sum()


def find_null_rows(df):
    null_columns = find_null_columns(df)
    return df[df.isnull().any(axis=1)][null_columns]


def create(*columns):
    return pd.DataFrame(columns=columns)


def split(df, percent):
    split_position = int(percent * len(df))
    return df[:split_position], df[split_position:]


def split_column(df, column, percent=0.8):
    split_position = int(percent * len(df))
    return df[column][:split_position], df[column][split_position:]


def describe_data(df):
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


def moving_average(df, column, new_column=None, time_frame=3, shift=1):
    return df[column if new_column is None else new_column].rolling(time_frame).mean().shift(shift)


def split_dataframe(df, chunk_size = 10000):
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks


def shape(df):
    return len(df.index), len(df.columns)


def get_column_index(df, column):
    return df.columns.get_loc(column)


def print(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)


def drop_na(df):
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)


def sort(df, column):
    df.sort_values(column, inplace=True)
    df.reset_index(drop=True, inplace=True)