import numpy as np

def standardize_dataset(x, mean=None, std=None, eps=1e-12, dtype=np.float64):
    """
    Standardize features to zero mean and unit variance.

    Parameters
    ----------
    x : array-like, shape (n_samples, n_features) or (n_samples,)
        Input data.
    mean : array-like, optional
        Mean to use for standardization. If None, computed from x.
    std : array-like, optional
        Standard deviation to use for standardization. If None, computed from x.
    eps : float, optional
        Threshold below which std is treated as zero.
    dtype : data-type, optional
        Working/output dtype. Default is np.float64.

    Returns
    -------
    x_std : ndarray
        Standardized data.
    mean : ndarray
        Mean used for standardization.
    std : ndarray
        Standard deviation used for standardization, with tiny values replaced by 1.
    """
    x = np.asarray(x, dtype=dtype)

    if mean is None:
        mean = np.mean(x, axis=0, dtype=dtype)
    else:
        mean = np.asarray(mean, dtype=dtype)

    if std is None:
        std = np.std(x, axis=0, dtype=dtype)
    else:
        std = np.asarray(std, dtype=dtype)

    std = np.where(std < eps, 1.0, std)

    # One copy, then in-place ops
    x_std = x.copy()
    x_std -= mean
    x_std /= std

    return x_std, mean, std