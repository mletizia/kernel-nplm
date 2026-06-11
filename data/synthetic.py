"""Synthetic one-dimensional samples for generator-based NPLM examples."""

import numpy as np


#########################################################################################################
# Validation helpers

def _validate_count(value, name):
    """Validate a non-negative event count.

    :param value: Input count-like value.
    :param name: Name used in validation errors.
    :returns: Non-negative integer count.
    """
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_positive(value, name):
    """Validate a positive finite scalar.

    :param value: Input scalar.
    :param name: Name used in validation errors.
    :returns: Positive finite float.
    """
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive finite")
    return value


def _validate_nonnegative(value, name):
    """Validate a non-negative finite scalar.

    :param value: Input scalar.
    :param name: Name used in validation errors.
    :returns: Non-negative finite float.
    """
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative finite")
    return value


#########################################################################################################
# Reference and signal generators

def sample_ref_exp(N, rate=8.0, xmax=1.0, rng=None):
    """
    Sample from an exponential distribution truncated at ``xmax``.

    :param N: Number of samples to draw.
    :param rate: Exponential rate parameter.
    :param xmax: Upper truncation point.
    :param rng: Optional NumPy random generator.
    :returns: One-dimensional array with shape ``(N,)``.
    """
    n = _validate_count(N, "N")
    rate = _validate_positive(rate, "rate")
    xmax = _validate_positive(xmax, "xmax")
    rng = np.random.default_rng() if rng is None else rng
    normalization = 1.0 - np.exp(-rate * xmax)
    uniform = rng.random(n)
    return -(1.0 / rate) * np.log(1.0 - uniform * normalization)


def sample_signal_gauss(N, mu=0.8, sigma=0.02, rng=None):
    """
    Sample signal events from a Gaussian distribution.

    :param N: Number of samples to draw.
    :param mu: Gaussian mean.
    :param sigma: Gaussian standard deviation.
    :param rng: Optional NumPy random generator.
    :returns: One-dimensional array with shape ``(N,)``.
    """
    n = _validate_count(N, "N")
    sigma = _validate_positive(sigma, "sigma")
    rng = np.random.default_rng() if rng is None else rng
    return rng.normal(loc=mu, scale=sigma, size=n)


#########################################################################################################
# Pseudo-data generators

def make_data_sample_poisson(NR=2000, NS=10, rng=None):
    """
    Generate a pseudo-dataset with Poisson-fluctuated background and signal counts.

    :param NR: Expected background count.
    :param NS: Expected signal count.
    :param rng: Optional NumPy random generator.
    :returns: Tuple ``(x, xb, xs, N_B, N_S)`` containing the shuffled data,
        background component, signal component, and realized counts.
    """
    expected_background = _validate_nonnegative(NR, "NR")
    expected_signal = _validate_nonnegative(NS, "NS")
    rng = np.random.default_rng() if rng is None else rng
    n_background = int(rng.poisson(expected_background))
    n_signal = int(rng.poisson(expected_signal))

    xb = sample_ref_exp(n_background, rng=rng)
    xs = sample_signal_gauss(n_signal, rng=rng)

    x = np.concatenate([xb, xs])
    rng.shuffle(x)
    return x, xb, xs, n_background, n_signal
