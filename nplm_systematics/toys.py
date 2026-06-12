"""Analytic one-dimensional toys for nuisance-ratio validation."""

from __future__ import annotations

from typing import Optional

import numpy as np


def sample_truncated_exponential(
    n: int,
    *,
    rate: float = 8.0,
    xmax: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample a truncated exponential distribution on ``[0, xmax]``."""
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


def nuisance_rate(nu: float, *, rate0: float = 8.0, rate_log_step: float = 0.08) -> float:
    """Map a standardized nuisance value to a positive exponential rate."""
    rate0 = float(rate0)
    rate_log_step = float(rate_log_step)
    if not np.isfinite(rate0) or rate0 <= 0:
        raise ValueError("rate0 must be positive and finite")
    if not np.isfinite(rate_log_step):
        raise ValueError("rate_log_step must be finite")
    return float(rate0 * np.exp(rate_log_step * float(nu)))


def sample_exponential_nuisance(
    n: int,
    nu: float,
    *,
    rate0: float = 8.0,
    rate_log_step: float = 0.08,
    xmax: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample the truncated exponential toy at nuisance value ``nu``."""
    return sample_truncated_exponential(
        n,
        rate=nuisance_rate(nu, rate0=rate0, rate_log_step=rate_log_step),
        xmax=xmax,
        rng=rng,
    )


def truncated_exponential_logpdf(x: np.ndarray, *, rate: float, xmax: float = 1.0) -> np.ndarray:
    """Evaluate the normalized truncated-exponential log-density."""
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
    x: np.ndarray,
    nu: float,
    *,
    rate0: float = 8.0,
    rate_log_step: float = 0.08,
    xmax: float = 1.0,
) -> np.ndarray:
    """Analytic ``log p(x | nu) / p(x | 0)`` for the toy."""
    rate_nu = nuisance_rate(nu, rate0=rate0, rate_log_step=rate_log_step)
    return (
        truncated_exponential_logpdf(x, rate=rate_nu, xmax=xmax)
        - truncated_exponential_logpdf(x, rate=rate0, xmax=xmax)
    )


def truncated_exponential_log_ratio_derivative(
    x: np.ndarray,
    *,
    rate0: float = 8.0,
    rate_log_step: float = 0.08,
    xmax: float = 1.0,
) -> np.ndarray:
    """Analytic derivative of ``log p(x | nu) / p(x | 0)`` at ``nu = 0``."""
    x_arr = np.asarray(x, dtype=np.float64)
    rate0 = float(rate0)
    rate_log_step = float(rate_log_step)
    xmax = float(xmax)
    z0 = 1.0 - np.exp(-rate0 * xmax)
    d_rate = rate_log_step * rate0
    d_log_norm = d_rate * xmax * np.exp(-rate0 * xmax) / z0
    return rate_log_step - d_rate * x_arr - d_log_norm
