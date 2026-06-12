"""Finite-difference nuisance morphing with Falkon ratio fits."""

import time

import numpy as np

from .falkon_ratio import FalkonLogRatioEstimator, FalkonRatioConfig
from .morphing import LinearLogRMorphingCache, QuadraticLogRMorphingCache


#########################################################################################################
# Validation helpers

def _as_2d_float64(x, name):
    """Convert an input sample to a contiguous two-dimensional float array.

    :param x: Input array with shape ``(n_samples,)`` or ``(n_samples, n_features)``.
    :param name: Name used in validation errors.
    :returns: Contiguous array with shape ``(n_samples, n_features)``.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty")
    return np.ascontiguousarray(arr)


#########################################################################################################
# Result containers

class FiniteDifferenceSummary:
    """Diagnostics for a pair of nuisance-ratio fits."""

    def __init__(
        self,
        epsilon,
        target_norm_ratio_plus,
        target_norm_ratio_minus,
        train_time_plus,
        train_time_minus,
        config,
    ):
        """Initialize the fit summary.

        :param epsilon: Nuisance displacement used for the finite difference.
        :param target_norm_ratio_plus: Expected-rate ratio for the plus sample.
        :param target_norm_ratio_minus: Expected-rate ratio for the minus sample.
        :param train_time_plus: Plus-model training time in seconds.
        :param train_time_minus: Minus-model training time in seconds.
        :param config: Falkon ratio configuration dictionary.
        """
        self.epsilon = epsilon
        self.target_norm_ratio_plus = target_norm_ratio_plus
        self.target_norm_ratio_minus = target_norm_ratio_minus
        self.train_time_plus = train_time_plus
        self.train_time_minus = train_time_minus
        self.config = config


#########################################################################################################
# Finite-difference morpher

class FiniteDifferenceMorpher:
    """Learn linear nuisance response ``delta(x)`` from ``+/- epsilon`` samples."""

    def __init__(
        self,
        epsilon,
        config=None,
        target_norm_ratio_plus=1.0,
        target_norm_ratio_minus=1.0,
    ):
        """Initialize the finite-difference morpher.

        :param epsilon: Positive nuisance displacement.
        :param config: Optional Falkon ratio configuration dictionary.
        :param target_norm_ratio_plus: Expected-rate ratio for ``+epsilon``.
        :param target_norm_ratio_minus: Expected-rate ratio for ``-epsilon``.
        """
        epsilon = float(epsilon)
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")

        self.epsilon = epsilon
        self.config = FalkonRatioConfig.from_mapping(config)
        self.target_norm_ratio_plus = float(target_norm_ratio_plus)
        self.target_norm_ratio_minus = float(target_norm_ratio_minus)
        self.plus_estimator = None
        self.minus_estimator = None
        self.summary_ = None

    def fit(self, X_central, X_plus, X_minus):
        """Fit the ``+epsilon`` and ``-epsilon`` Falkon ratio models.

        :param X_central: Central sample with shape ``(n_central, n_features)``.
        :param X_plus: Plus-varied sample with shape ``(n_plus, n_features)``.
        :param X_minus: Minus-varied sample with shape ``(n_minus, n_features)``.
        :returns: ``self``.
        """
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

    def predict_components(self, X):
        """Return ``g_plus`` and ``g_minus`` predictions on ``X``.

        :param X: Evaluation sample with shape ``(n_samples, n_features)``.
        :returns: Pair of arrays, each with shape ``(n_samples,)``.
        """
        if self.plus_estimator is None or self.minus_estimator is None:
            raise RuntimeError("Fit the FiniteDifferenceMorpher before prediction")
        x = _as_2d_float64(X, "X")
        g_plus = self.plus_estimator.predict_log_ratio(x)
        g_minus = self.minus_estimator.predict_log_ratio(x)
        return g_plus, g_minus

    def predict_delta(self, X):
        """Evaluate the finite-difference response ``delta(x)``.

        :param X: Evaluation sample with shape ``(n_samples, n_features)``.
        :returns: Response array with shape ``(n_samples,)``.
        """
        g_plus, g_minus = self.predict_components(X)
        return (g_plus - g_minus) / (2.0 * self.epsilon)

    def predict_log_r(self, X, nu):
        """Evaluate the linearized nuisance log-ratio ``nu * delta(x)``.

        :param X: Evaluation sample with shape ``(n_samples, n_features)``.
        :param nu: Scalar nuisance value.
        :returns: Log-ratio array with shape ``(n_samples,)``.
        """
        return float(nu) * self.predict_delta(X)

    def build_cache(self, X_ref, X_data=None, clip=30.0, metadata=None):
        """Build a prediction cache for later profiling on fixed events.

        :param X_ref: Reference events with shape ``(n_ref, n_features)``.
        :param X_data: Optional data events with shape ``(n_data, n_features)``.
        :param clip: Absolute clipping threshold for cached log-ratios.
        :param metadata: Optional metadata dictionary saved with the cache.
        :returns: ``LinearLogRMorphingCache`` instance.
        """
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


#########################################################################################################
# Quadratic finite-difference morpher

class QuadraticFiniteDifferenceSummary:
    """Diagnostics for quadratic finite-difference nuisance-ratio fits."""

    def __init__(self, epsilon_linear, epsilon_quadratic, linear_summary, quadratic_summary):
        """Initialize the quadratic fit summary.

        :param epsilon_linear: Nuisance displacement used for the linear term.
        :param epsilon_quadratic: Nuisance displacement used for the quadratic term.
        :param linear_summary: Summary from the linear finite-difference fit.
        :param quadratic_summary: Summary from the quadratic finite-difference fit.
        """
        self.epsilon_linear = epsilon_linear
        self.epsilon_quadratic = epsilon_quadratic
        self.linear_summary = linear_summary
        self.quadratic_summary = quadratic_summary


class QuadraticFiniteDifferenceMorpher:
    """Learn first- and second-order nuisance responses from Falkon ratio fits."""

    def __init__(self, epsilon_linear, epsilon_quadratic=None, config=None):
        """Initialize the quadratic finite-difference morpher.

        :param epsilon_linear: Positive nuisance displacement for ``delta1``.
        :param epsilon_quadratic: Positive nuisance displacement for ``delta2``.
        :param config: Optional Falkon ratio configuration dictionary.
        """
        epsilon_linear = float(epsilon_linear)
        if not np.isfinite(epsilon_linear) or epsilon_linear <= 0:
            raise ValueError("epsilon_linear must be positive and finite")

        if epsilon_quadratic is None:
            epsilon_quadratic = epsilon_linear
        epsilon_quadratic = float(epsilon_quadratic)
        if not np.isfinite(epsilon_quadratic) or epsilon_quadratic <= 0:
            raise ValueError("epsilon_quadratic must be positive and finite")

        self.epsilon_linear = epsilon_linear
        self.epsilon_quadratic = epsilon_quadratic
        self.config = FalkonRatioConfig.from_mapping(config)
        self.linear_morpher = None
        self.quadratic_morpher = None
        self.summary_ = None

    def fit(
        self,
        X_central,
        X_linear_plus,
        X_linear_minus,
        X_quadratic_plus=None,
        X_quadratic_minus=None,
    ):
        """Fit Falkon models for first- and second-order nuisance responses.

        :param X_central: Central sample with shape ``(n_central, n_features)``.
        :param X_linear_plus: Sample at ``+epsilon_linear``.
        :param X_linear_minus: Sample at ``-epsilon_linear``.
        :param X_quadratic_plus: Optional sample at ``+epsilon_quadratic``.
        :param X_quadratic_minus: Optional sample at ``-epsilon_quadratic``.
        :returns: ``self``.
        """
        if X_quadratic_plus is None:
            X_quadratic_plus = X_linear_plus
        if X_quadratic_minus is None:
            X_quadratic_minus = X_linear_minus

        cfg_dict = self.config.to_dict()

        self.linear_morpher = FiniteDifferenceMorpher(self.epsilon_linear, cfg_dict)
        self.linear_morpher.fit(X_central, X_linear_plus, X_linear_minus)

        same_epsilon = np.isclose(self.epsilon_linear, self.epsilon_quadratic)
        same_plus = X_quadratic_plus is X_linear_plus
        same_minus = X_quadratic_minus is X_linear_minus
        if same_epsilon and same_plus and same_minus:
            self.quadratic_morpher = self.linear_morpher
        else:
            self.quadratic_morpher = FiniteDifferenceMorpher(self.epsilon_quadratic, cfg_dict)
            self.quadratic_morpher.fit(X_central, X_quadratic_plus, X_quadratic_minus)

        self.summary_ = QuadraticFiniteDifferenceSummary(
            epsilon_linear=self.epsilon_linear,
            epsilon_quadratic=self.epsilon_quadratic,
            linear_summary=self.linear_morpher.summary_,
            quadratic_summary=self.quadratic_morpher.summary_,
        )
        return self

    def predict_delta1(self, X):
        """Evaluate the first-order nuisance response.

        :param X: Evaluation sample with shape ``(n_samples, n_features)``.
        :returns: First-order response array with shape ``(n_samples,)``.
        """
        if self.linear_morpher is None:
            raise RuntimeError("Fit the QuadraticFiniteDifferenceMorpher before prediction")
        return self.linear_morpher.predict_delta(X)

    def predict_delta2(self, X):
        """Evaluate the second-order nuisance response.

        :param X: Evaluation sample with shape ``(n_samples, n_features)``.
        :returns: Second-order response array with shape ``(n_samples,)``.
        """
        if self.quadratic_morpher is None:
            raise RuntimeError("Fit the QuadraticFiniteDifferenceMorpher before prediction")
        g_plus, g_minus = self.quadratic_morpher.predict_components(X)
        return (g_plus + g_minus) / (self.epsilon_quadratic**2)

    def predict_log_r(self, X, nu):
        """Evaluate the quadratic nuisance log-ratio.

        :param X: Evaluation sample with shape ``(n_samples, n_features)``.
        :param nu: Scalar nuisance value.
        :returns: Log-ratio array with shape ``(n_samples,)``.
        """
        nu = float(nu)
        return nu * self.predict_delta1(X) + 0.5 * nu**2 * self.predict_delta2(X)

    def build_cache(self, X_ref, X_data=None, clip=30.0, metadata=None):
        """Build a quadratic prediction cache for fixed events.

        :param X_ref: Reference events with shape ``(n_ref, n_features)``.
        :param X_data: Optional data events with shape ``(n_data, n_features)``.
        :param clip: Absolute clipping threshold for cached log-ratios.
        :param metadata: Optional metadata dictionary saved with the cache.
        :returns: ``QuadraticLogRMorphingCache`` instance.
        """
        delta1_ref = self.predict_delta1(X_ref)
        delta2_ref = self.predict_delta2(X_ref)
        delta1_data = None
        delta2_data = None
        if X_data is not None:
            delta1_data = self.predict_delta1(X_data)
            delta2_data = self.predict_delta2(X_data)

        cache_metadata = dict(metadata or {})
        cache_metadata.update(
            {
                "method": "quadratic_finite_difference_falkon",
                "epsilon_linear": self.epsilon_linear,
                "epsilon_quadratic": self.epsilon_quadratic,
                "falkon_ratio_config": self.config.to_dict(),
            }
        )
        if self.summary_ is not None:
            cache_metadata["linear_train_time_plus"] = self.summary_.linear_summary.train_time_plus
            cache_metadata["linear_train_time_minus"] = self.summary_.linear_summary.train_time_minus
            cache_metadata["quadratic_train_time_plus"] = (
                self.summary_.quadratic_summary.train_time_plus
            )
            cache_metadata["quadratic_train_time_minus"] = (
                self.summary_.quadratic_summary.train_time_minus
            )

        return QuadraticLogRMorphingCache(
            delta1_ref=delta1_ref,
            delta2_ref=delta2_ref,
            delta1_data=delta1_data,
            delta2_data=delta2_data,
            clip=clip,
            metadata=cache_metadata,
        )
