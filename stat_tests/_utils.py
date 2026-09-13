"""Numerical helpers shared by statistical tests and saved-result comparison."""

import numpy as np


_ALTERNATIVE_QUANTILE_LEVELS = np.array([0.16, 0.50, 0.84], dtype=np.float64)


def _empirical_pvalues(null_statistics, statistic_values):
    """Corrected right-tail p-values for a scalar or 1D array of statistics.

    Ties count as extreme. Finite negative statistics are allowed; invalid
    values raise rather than contributing silently to the denominator.
    """
    null = np.asarray(null_statistics, dtype=np.float64)
    values = np.asarray(statistic_values, dtype=np.float64)
    if null.ndim != 1 or null.size == 0 or not np.all(np.isfinite(null)):
        raise ValueError("null_statistics must be a nonempty, finite 1D array")
    if values.ndim > 1 or not np.all(np.isfinite(values)):
        raise ValueError("statistic_values must be finite and scalar or 1D")
    counts = np.sum(null[:, None] >= np.atleast_1d(values)[None, :], axis=0)
    p_values = (1.0 + counts) / (len(null) + 1.0)
    return float(p_values[0]) if values.ndim == 0 else p_values
