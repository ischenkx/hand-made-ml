import numpy as np


def as_matrix(t):
    if t.ndim == 0:
        return t.reshape((1, 1))
    if t.ndim == 1:
        return t.reshape((1, t.shape[0]))
    return t


def prepare(data, dtype=None):
    if type(data) is not np.ndarray:
        data = np.array(data, dtype)
    elif dtype is not None:
        data = data.astype(dtype)
    return as_matrix(data)