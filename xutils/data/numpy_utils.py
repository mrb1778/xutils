import numpy as np


def split(x, split_percent=0.8):
    split_position = int(split_percent * (len(x)))
    return x[0:split_position], x[split_position:]


def convert_time_series_to_look_back(x, look_back):
    out_x, out_y = [], []
    for i in range(len(x) - look_back):
        a = x[i:(i + look_back), ]
        out_x.append(a)
        out_y.append(x[i + look_back,])
    return np.array(out_x), np.array(out_y)


def head(data, count=10):
    return data[:count]


def tail(data, count=10):
    return data[-count]


def trim_multiple(df, size):
    no_of_rows_drop = df.shape[0] % size
    if no_of_rows_drop > 0:
        return df[:-no_of_rows_drop]
    else:
        return df


def build_series(mat, y_col_index, time_steps):
    dim_0 = mat.shape[0] - time_steps
    dim_1 = mat.shape[1]

    x = np.zeros((dim_0, time_steps, dim_1))
    y = np.zeros((dim_0,))

    for i in range(dim_0):
        x[i] = mat[i:time_steps + i]
        y[i] = mat[time_steps + i, y_col_index]

    return x, y


def normalize(x_train, x_test, epsilon=1e-7):
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + epsilon)
    x_test = (x_test - mean) / (std + epsilon)
    return x_train, x_test


def pad_image(x, pad=4, mode="reflect"):
    return np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode=mode)


def ensure_has_channels(x_train, x_test, num_channels=1):
    x_train_shape = x_train.shape
    if len(x_train_shape) == 3:
        x_train = x_train.reshape(x_train_shape[0], x_train_shape[1], x_train_shape[1], num_channels)
        x_test_shape = x_test.shape
        x_test = x_test.reshape(x_test_shape[0], x_test_shape[1], x_test_shape[1], num_channels)

    return x_train, x_test