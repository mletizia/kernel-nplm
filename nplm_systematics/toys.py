"""Analytic one-dimensional toys for nuisance-ratio validation."""

import numpy as np


#########################################################################################################
# Truncated exponential toy

def sample_truncated_exponential(n, rate=8.0, xmax=1.0, rng=None):
    """Sample a truncated exponential distribution on ``[0, xmax]``.

    :param n: Number of samples.
    :param rate: Positive exponential rate.
    :param xmax: Positive upper truncation point.
    :param rng: Optional NumPy random generator.
    :returns: Sample array with shape ``(n,)``.
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")
    rate = float(rate)
    xmax = float(xmax)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be positive and finite")
    if not np.isfinite(xmax) or xmax <= 0:
        raise ValueError("xmax must be positive and finite")

    rng = np.random.default_rng() if rng is None else rng
    z = 1.0 - np.exp(-rate * xmax)
    u = rng.random(n)
    return -(1.0 / rate) * np.log(1.0 - u * z)


def nuisance_rate(nu, rate0=8.0, rate_log_step=0.08):
    """Map a standardized nuisance value to a positive exponential rate.

    :param nu: Scalar nuisance value.
    :param rate0: Central exponential rate.
    :param rate_log_step: Log-rate shift per nuisance unit.
    :returns: Positive nuisance-shifted rate.
    """
    rate0 = float(rate0)
    rate_log_step = float(rate_log_step)
    if not np.isfinite(rate0) or rate0 <= 0:
        raise ValueError("rate0 must be positive and finite")
    if not np.isfinite(rate_log_step):
        raise ValueError("rate_log_step must be finite")
    return float(rate0 * np.exp(rate_log_step * float(nu)))


def sample_exponential_nuisance(
    n,
    nu,
    rate0=8.0,
    rate_log_step=0.08,
    xmax=1.0,
    rng=None,
):
    """Sample the truncated exponential toy at nuisance value ``nu``.

    :param n: Number of samples.
    :param nu: Scalar nuisance value.
    :param rate0: Central exponential rate.
    :param rate_log_step: Log-rate shift per nuisance unit.
    :param xmax: Positive upper truncation point.
    :param rng: Optional NumPy random generator.
    :returns: Sample array with shape ``(n,)``.
    """
    return sample_truncated_exponential(
        n,
        rate=nuisance_rate(nu, rate0=rate0, rate_log_step=rate_log_step),
        xmax=xmax,
        rng=rng,
    )


#########################################################################################################
# Analytic density and nuisance response

def truncated_exponential_logpdf(x, rate, xmax=1.0):
    """Evaluate the normalized truncated-exponential log-density.

    :param x: Evaluation points.
    :param rate: Positive exponential rate.
    :param xmax: Positive upper truncation point.
    :returns: Log-density values with the same shape as ``x``.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    rate = float(rate)
    xmax = float(xmax)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be positive and finite")
    if not np.isfinite(xmax) or xmax <= 0:
        raise ValueError("xmax must be positive and finite")

    z = 1.0 - np.exp(-rate * xmax)
    log_pdf = np.log(rate) - rate * x_arr - np.log(z)
    outside = (x_arr < 0.0) | (x_arr > xmax)
    if np.any(outside):
        log_pdf = np.asarray(log_pdf).copy()
        log_pdf[outside] = -np.inf
    return log_pdf


def truncated_exponential_log_ratio(
    x,
    nu,
    rate0=8.0,
    rate_log_step=0.08,
    xmax=1.0,
):
    """Evaluate analytic ``log p(x | nu) / p(x | 0)`` for the toy.

    :param x: Evaluation points.
    :param nu: Scalar nuisance value.
    :param rate0: Central exponential rate.
    :param rate_log_step: Log-rate shift per nuisance unit.
    :param xmax: Positive upper truncation point.
    :returns: Log-ratio values with the same shape as ``x``.
    """
    rate_nu = nuisance_rate(nu, rate0=rate0, rate_log_step=rate_log_step)
    return (
        truncated_exponential_logpdf(x, rate=rate_nu, xmax=xmax)
        - truncated_exponential_logpdf(x, rate=rate0, xmax=xmax)
    )


def truncated_exponential_log_ratio_derivative(
    x,
    rate0=8.0,
    rate_log_step=0.08,
    xmax=1.0,
):
    """Evaluate the derivative of the analytic log-ratio at ``nu = 0``.

    :param x: Evaluation points.
    :param rate0: Central exponential rate.
    :param rate_log_step: Log-rate shift per nuisance unit.
    :param xmax: Positive upper truncation point.
    :returns: Response values with the same shape as ``x``.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    rate0 = float(rate0)
    rate_log_step = float(rate_log_step)
    xmax = float(xmax)
    z0 = 1.0 - np.exp(-rate0 * xmax)
    d_rate = rate_log_step * rate0
    d_log_norm = d_rate * xmax * np.exp(-rate0 * xmax) / z0
    return rate_log_step - d_rate * x_arr - d_log_norm


#########################################################################################################
# Paper exponential scale toy

def sample_paper_exponential_nuisance(n, nu_scale=0.0, rng=None):
    """Sample the untruncated exponential scale toy used in the paper.

    The central-value distribution is ``exp(-x)`` on ``[0, inf)``. The scale
    nuisance maps ``x`` to an exponential distribution with scale
    ``exp(nu_scale)``.

    :param n: Number of samples.
    :param nu_scale: Scale nuisance value.
    :param rng: Optional NumPy random generator.
    :returns: Sample array with shape ``(n,)``.
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")

    nu_scale = float(nu_scale)
    if not np.isfinite(nu_scale):
        raise ValueError("nu_scale must be finite")

    rng = np.random.default_rng() if rng is None else rng
    return rng.exponential(scale=np.exp(nu_scale), size=n)


def paper_exponential_log_ratio(x, nu_scale, nu_norm=0.0):
    """Evaluate the paper's analytic scale-nuisance log-ratio.

    For the scale nuisance, the analytic expression is
    ``log r(x; nu_scale) = x * (1 - exp(-nu_scale)) - nu_scale``. The optional
    normalization nuisance contributes additively as ``nu_norm``.

    :param x: Evaluation points.
    :param nu_scale: Scale nuisance value.
    :param nu_norm: Optional normalization nuisance value.
    :returns: Log-ratio values with the same shape as ``x``.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    nu_scale = float(nu_scale)
    nu_norm = float(nu_norm)
    if not np.isfinite(nu_scale):
        raise ValueError("nu_scale must be finite")
    if not np.isfinite(nu_norm):
        raise ValueError("nu_norm must be finite")

    return nu_norm + x_arr * (1.0 - np.exp(-nu_scale)) - nu_scale


def paper_exponential_delta1(x):
    """Evaluate the first Taylor coefficient of the paper log-ratio.

    :param x: Evaluation points.
    :returns: First-order coefficient values with the same shape as ``x``.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    return x_arr - 1.0


def paper_exponential_delta2(x):
    """Evaluate the second Taylor coefficient of the paper log-ratio.

    The expansion convention is
    ``log r(x; nu) = nu * delta1(x) + 0.5 * nu**2 * delta2(x) + ...``.

    :param x: Evaluation points.
    :returns: Second-order coefficient values with the same shape as ``x``.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    return -x_arr


def paper_exponential_bin_log_ratio(bin_edges, nu_scale, nu_norm=0.0):
    """Evaluate ``log N_b(nu) / N_b(0)`` for exponential bins.

    :param bin_edges: Monotonic bin edges; the last edge may be ``np.inf``.
    :param nu_scale: Scale nuisance value.
    :param nu_norm: Optional normalization nuisance value.
    :returns: Bin-integrated log-ratio array with shape ``(n_bins,)``.
    """
    edges = np.asarray(bin_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.shape[0] < 2:
        raise ValueError("bin_edges must be a one-dimensional array with at least two edges")
    if np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be strictly increasing")
    if edges[0] < 0:
        raise ValueError("bin_edges must be non-negative")

    nu_scale = float(nu_scale)
    nu_norm = float(nu_norm)
    if not np.isfinite(nu_scale):
        raise ValueError("nu_scale must be finite")
    if not np.isfinite(nu_norm):
        raise ValueError("nu_norm must be finite")

    inv_scale = np.exp(-nu_scale)
    lo = edges[:-1]
    hi = edges[1:]
    central = np.exp(-lo) - np.exp(-hi)
    varied = np.exp(-lo * inv_scale) - np.exp(-hi * inv_scale)

    return nu_norm + np.log(varied) - np.log(central)
