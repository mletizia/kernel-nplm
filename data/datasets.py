import numpy as np


def build_pooled_sample(X_ref, X_dat, dtype=np.float64):
    X_ref = np.asarray(X_ref, dtype=dtype)
    X_dat = np.asarray(X_dat, dtype=dtype)

    if X_ref.ndim == 1:
        X_ref = X_ref.reshape(-1, 1)
    if X_dat.ndim == 1:
        X_dat = X_dat.reshape(-1, 1)

    if X_ref.ndim != 2 or X_dat.ndim != 2:
        raise ValueError("X_ref and X_dat must be 1D or 2D arrays")

    n_ref, d_ref = X_ref.shape
    n_dat, d_dat = X_dat.shape

    if d_ref != d_dat:
        raise ValueError(f"Feature mismatch: {d_ref} vs {d_dat}")

    X = np.empty((n_ref + n_dat, d_ref), dtype=dtype)
    y = np.empty(n_ref + n_dat, dtype=dtype)

    X[:n_ref] = X_ref
    X[n_ref:] = X_dat

    y[:n_ref] = 0.0
    y[n_ref:] = 1.0

    return X, y