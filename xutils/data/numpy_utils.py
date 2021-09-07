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


def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T


def count_unique(d):
    return dict(zip(*np.unique(d, return_counts=True)))


def count_nan(d):
    return np.count_nonzero(np.isnan(d))


def print_all(d):
    with np.printoptions(threshold=np.inf):
        print(d)
