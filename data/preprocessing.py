"""Feature preprocessing helpers."""

import numpy as np


#########################################################################################################
# Standardization

def standardize_dataset(x, mean=None, std=None, eps=1e-12, dtype=np.float64):
    """
    Standardize features to zero mean and unit variance.

    :param x: Input data with shape ``(n_samples,)`` or ``(n_samples, n_features)``.
    :param mean: Optional mean. If ``None``, it is computed from ``x``.
    :param std: Optional standard deviation. If ``None``, it is computed from ``x``.
    :param eps: Threshold below which standard deviations are replaced by one.
    :param dtype: NumPy dtype used for computation and output.
    :returns: Tuple ``(x_std, mean, std)`` with standardized data and fitted moments.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")

    x = np.asarray(x, dtype=dtype)
    if x.size == 0:
        raise ValueError("x must be non-empty")

    if mean is None:
        mean = np.mean(x, axis=0, dtype=dtype)
    else:
        mean = np.asarray(mean, dtype=dtype)

    if std is None:
        std = np.std(x, axis=0, dtype=dtype)
    else:
        std = np.asarray(std, dtype=dtype)

    std = np.where(std < eps, 1.0, std)

    x_std = x.copy()
    x_std -= mean
    x_std /= std

    return x_std, mean, std
