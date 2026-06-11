"""Dataset assembly utilities for NPLM inputs."""

import numpy as np


#########################################################################################################
# Pooled samples

def build_pooled_sample(X_ref, X_dat, dtype=np.float64):
    """
    Build a pooled sample and binary labels from reference and data arrays.

    :param X_ref: Reference sample with shape ``(N_R,)`` or ``(N_R, d)``.
    :param X_dat: Data sample with shape ``(N_D,)`` or ``(N_D, d)``.
    :param dtype: NumPy dtype used for the returned arrays.
    :returns: Tuple ``(X, y)`` with shapes ``(N_R + N_D, d)`` and ``(N_R + N_D,)``.
    """
    x_ref = np.asarray(X_ref, dtype=dtype)
    x_dat = np.asarray(X_dat, dtype=dtype)

    if x_ref.ndim == 1:
        x_ref = x_ref.reshape(-1, 1)
    if x_dat.ndim == 1:
        x_dat = x_dat.reshape(-1, 1)

    if x_ref.ndim != 2 or x_dat.ndim != 2:
        raise ValueError("X_ref and X_dat must be 1D or 2D arrays")

    if x_ref.shape[0] == 0 or x_dat.shape[0] == 0:
        raise ValueError("X_ref and X_dat must be non-empty")

    n_ref, d_ref = x_ref.shape
    n_dat, d_dat = x_dat.shape

    if d_ref != d_dat:
        raise ValueError(f"Feature mismatch: {d_ref} vs {d_dat}")

    x_pooled = np.empty((n_ref + n_dat, d_ref), dtype=dtype)
    y = np.empty(n_ref + n_dat, dtype=dtype)

    x_pooled[:n_ref] = x_ref
    x_pooled[n_ref:] = x_dat

    y[:n_ref] = 0.0
    y[n_ref:] = 1.0

    return x_pooled, y
