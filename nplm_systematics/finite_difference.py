"""Finite-difference nuisance morphing with Falkon ratio fits."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from .falkon_ratio import FalkonLogRatioEstimator, FalkonRatioConfig
from .morphing import LinearLogRMorphingCache


def _as_2d_float64(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty")
    return np.ascontiguousarray(arr)


@dataclass
class FiniteDifferenceSummary:
    """Diagnostics for a pair of nuisance ratio fits."""

    epsilon: float
    target_norm_ratio_plus: float
    target_norm_ratio_minus: float
    train_time_plus: float
    train_time_minus: float
    config: Mapping[str, Any]


class FiniteDifferenceMorpher:
    """Learn linear nuisance response ``delta(x)`` from ``+/- epsilon`` samples.

    Two Falkon classifiers are trained:

    ``g_+(x) ~= log p(x | +epsilon) / p(x | 0)``

    ``g_-(x) ~= log p(x | -epsilon) / p(x | 0)``

    and the cached response is

    ``delta(x) = (g_+(x) - g_-(x)) / (2 epsilon)``.
    """

    def __init__(
        self,
        epsilon: float,
        config: Optional[Mapping[str, Any]] = None,
        *,
        target_norm_ratio_plus: float = 1.0,
        target_norm_ratio_minus: float = 1.0,
    ):
        epsilon = float(epsilon)
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")

        self.epsilon = epsilon
        self.config = FalkonRatioConfig.from_mapping(config)
        self.target_norm_ratio_plus = float(target_norm_ratio_plus)
        self.target_norm_ratio_minus = float(target_norm_ratio_minus)
        self.plus_estimator: Optional[FalkonLogRatioEstimator] = None
        self.minus_estimator: Optional[FalkonLogRatioEstimator] = None
        self.summary_: Optional[FiniteDifferenceSummary] = None

    def fit(
        self,
        X_central: np.ndarray,
        X_plus: np.ndarray,
        X_minus: np.ndarray,
    ) -> "FiniteDifferenceMorpher":
        """Fit the ``+epsilon`` and ``-epsilon`` Falkon ratio models."""
        x0 = _as_2d_float64(X_central, "X_central")
        xp = _as_2d_float64(X_plus, "X_plus")
        xm = _as_2d_float64(X_minus, "X_minus")
        if xp.shape[1] != x0.shape[1] or xm.shape[1] != x0.shape[1]:
            raise ValueError("Central and varied samples must have matching feature dimension")

        cfg_dict = self.config.to_dict()

        self.plus_estimator = FalkonLogRatioEstimator(cfg_dict)
        t0 = time.time()
        self.plus_estimator.fit(
            x0,
            xp,
            target_norm_ratio=self.target_norm_ratio_plus,
        )
        train_time_plus = time.time() - t0

        self.minus_estimator = FalkonLogRatioEstimator(cfg_dict)
        t0 = time.time()
        self.minus_estimator.fit(
            x0,
            xm,
            target_norm_ratio=self.target_norm_ratio_minus,
        )
        train_time_minus = time.time() - t0

        self.summary_ = FiniteDifferenceSummary(
            epsilon=self.epsilon,
            target_norm_ratio_plus=self.target_norm_ratio_plus,
            target_norm_ratio_minus=self.target_norm_ratio_minus,
            train_time_plus=float(train_time_plus),
            train_time_minus=float(train_time_minus),
            config=cfg_dict,
        )
        return self

    def predict_components(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``g_plus`` and ``g_minus`` predictions on ``X``."""
        if self.plus_estimator is None or self.minus_estimator is None:
            raise RuntimeError("Fit the FiniteDifferenceMorpher before prediction")
        x = _as_2d_float64(X, "X")
        g_plus = self.plus_estimator.predict_log_ratio(x)
        g_minus = self.minus_estimator.predict_log_ratio(x)
        return g_plus, g_minus

    def predict_delta(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the finite-difference response ``delta(x)``."""
        g_plus, g_minus = self.predict_components(X)
        return (g_plus - g_minus) / (2.0 * self.epsilon)

    def predict_log_r(self, X: np.ndarray, nu: float) -> np.ndarray:
        """Evaluate the linearized nuisance log-ratio ``nu * delta(x)``."""
        return float(nu) * self.predict_delta(X)

    def build_cache(
        self,
        X_ref: np.ndarray,
        X_data: Optional[np.ndarray] = None,
        *,
        clip: float = 30.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> LinearLogRMorphingCache:
        """Build a prediction cache for later profiling on fixed events."""
        delta_ref = self.predict_delta(X_ref)
        delta_data = None if X_data is None else self.predict_delta(X_data)

        cache_metadata = dict(metadata or {})
        cache_metadata.update(
            {
                "method": "finite_difference_falkon",
                "epsilon": self.epsilon,
                "target_norm_ratio_plus": self.target_norm_ratio_plus,
                "target_norm_ratio_minus": self.target_norm_ratio_minus,
                "falkon_ratio_config": self.config.to_dict(),
            }
        )
        if self.summary_ is not None:
            cache_metadata["train_time_plus"] = self.summary_.train_time_plus
            cache_metadata["train_time_minus"] = self.summary_.train_time_minus

        return LinearLogRMorphingCache(
            delta_ref=delta_ref,
            delta_data=delta_data,
            clip=clip,
            metadata=cache_metadata,
        )
