"""Falkon density-ratio estimation for nuisance morphing."""

import numpy as np
import torch
from falkon import LogisticFalkon
from falkon.gsc_losses import WeightedCrossEntropyLoss
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from scipy.spatial.distance import pdist


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


def _as_list(value, name):
    """Return ``value`` as a list.

    :param value: Scalar or sequence value.
    :param name: Name used in validation errors.
    :returns: List representation of ``value``.
    """
    if isinstance(value, (list, tuple)):
        out = list(value)
    else:
        out = [value]
    if len(out) == 0:
        raise ValueError(f"{name} must be non-empty")
    return out


def _resolve_m(M, n_total):
    """Resolve the number of Nyström centers.

    :param M: Positive integer or ``"sqrt"``.
    :param n_total: Total training sample size.
    :returns: Positive number of centers.
    """
    if isinstance(M, str):
        if M != "sqrt":
            raise ValueError("M must be an integer or the string 'sqrt'")
        return int(np.sqrt(n_total))
    M_int = int(M)
    if M_int <= 0:
        raise ValueError("M must be positive")
    return M_int


#########################################################################################################
# Falkon ratio configuration

class FalkonRatioConfig:
    """Configuration for a Falkon log-density-ratio fit.

    The fitted classifier score estimates
    ``log target_norm_ratio * p_varied(x) / p_central(x)``.

    :param sigma: Gaussian-kernel width. If ``None``, use the median heuristic.
    :param M: Number of Nyström centers, or ``"sqrt"``.
    :param penalty_list: Falkon penalty sequence.
    :param iter_list: Falkon iteration sequence.
    :param cg_tol: Conjugate-gradient tolerance.
    :param keops: Falkon KeOps option.
    :param cpu: Whether to force CPU execution.
    :param seed: Optional Falkon seed.
    :param verbose: Verbosity level.
    """

    DEFAULTS = {
        "sigma": None,
        "M": "sqrt",
        "penalty_list": [1e-6],
        "iter_list": [1000],
        "cg_tol": float(np.sqrt(1e-7)),
        "keops": "no",
        "cpu": True,
        "seed": None,
        "verbose": 0,
    }

    ALIASES = {
        "lambda": "penalty_list",
        "iter": "iter_list",
    }

    def __init__(
        self,
        sigma=None,
        M="sqrt",
        penalty_list=None,
        iter_list=None,
        cg_tol=None,
        keops="no",
        cpu=True,
        seed=None,
        verbose=0,
    ):
        """Initialize the ratio-fit configuration."""
        self.sigma = sigma
        self.M = M
        self.penalty_list = _as_list(
            self.DEFAULTS["penalty_list"] if penalty_list is None else penalty_list,
            "penalty_list",
        )
        self.iter_list = _as_list(
            self.DEFAULTS["iter_list"] if iter_list is None else iter_list,
            "iter_list",
        )
        self.cg_tol = self.DEFAULTS["cg_tol"] if cg_tol is None else cg_tol
        self.keops = keops
        self.cpu = cpu
        self.seed = seed
        self.verbose = verbose

    @classmethod
    def from_mapping(cls, config):
        """Build a config from repo-style or native key names.

        :param config: Configuration dictionary, or ``None`` for defaults.
        :returns: Resolved ``FalkonRatioConfig`` instance.
        """
        if config is None:
            return cls()

        cfg = dict(config)
        for old_key, new_key in cls.ALIASES.items():
            if old_key in cfg:
                if new_key in cfg:
                    raise ValueError(f"Config cannot contain both {old_key!r} and {new_key!r}")
                cfg[new_key] = cfg.pop(old_key)

        allowed = set(cls.DEFAULTS)
        unknown = sorted(set(cfg) - allowed)
        if unknown:
            raise ValueError(f"Unknown FalkonRatioConfig keys: {unknown}")

        resolved = dict(cls.DEFAULTS)
        resolved.update(cfg)
        return cls(**resolved)

    def to_dict(self):
        """Return a JSON-friendly configuration dictionary.

        :returns: Dictionary with resolved configuration values.
        """
        return {
            "sigma": self.sigma,
            "M": self.M,
            "penalty_list": list(self.penalty_list),
            "iter_list": list(self.iter_list),
            "cg_tol": self.cg_tol,
            "keops": self.keops,
            "cpu": self.cpu,
            "seed": self.seed,
            "verbose": self.verbose,
        }


#########################################################################################################
# Falkon ratio estimator

class FalkonLogRatioEstimator:
    """Estimate ``log p_varied(x) / p_central(x)`` with LogisticFalkon."""

    def __init__(self, config=None):
        """Initialize the ratio estimator.

        :param config: Optional ``FalkonRatioConfig`` dictionary.
        """
        self.config = FalkonRatioConfig.from_mapping(config)
        self.model = None
        self.sigma_ = None
        self.neg_weight_ = None
        self.target_norm_ratio_ = None

    @staticmethod
    def estimate_sigma_median(X, max_points=5000, seed=None):
        """Estimate a Gaussian-kernel width from median pairwise distance.

        :param X: Input sample with shape ``(n_samples,)`` or ``(n_samples, n_features)``.
        :param max_points: Maximum number of points used by the heuristic.
        :param seed: Optional random seed used when subsampling.
        :returns: Positive median-distance estimate.
        """
        x = _as_2d_float64(X, "X")
        if x.shape[0] < 2:
            raise ValueError("Need at least two points to estimate sigma")

        max_points = int(max_points)
        if max_points < 2:
            raise ValueError("max_points must be at least two")

        if x.shape[0] > max_points:
            rng = np.random.default_rng(seed)
            idx = rng.choice(x.shape[0], size=max_points, replace=False)
            x = x[idx]

        sigma = float(np.median(pdist(x)))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Median heuristic returned invalid sigma={sigma}")
        return sigma

    def fit(self, X_central, X_varied, target_norm_ratio=1.0):
        """Fit a log-ratio model between central and nuisance-varied samples.

        :param X_central: Sample from ``p_0`` with shape ``(n_central, n_features)``.
        :param X_varied: Sample from ``p_nu`` with shape ``(n_varied, n_features)``.
        :param target_norm_ratio: Optional expected total-rate ratio.
        :returns: ``self``.
        """
        x0 = _as_2d_float64(X_central, "X_central")
        x1 = _as_2d_float64(X_varied, "X_varied")
        if x0.shape[1] != x1.shape[1]:
            raise ValueError(f"Feature mismatch: {x0.shape[1]} vs {x1.shape[1]}")

        target_norm_ratio = float(target_norm_ratio)
        if not np.isfinite(target_norm_ratio) or target_norm_ratio <= 0:
            raise ValueError("target_norm_ratio must be positive and finite")

        sigma = self.config.sigma
        if sigma is None:
            sigma = self.estimate_sigma_median(
                np.vstack([x0, x1]),
                seed=self.config.seed,
            )
        sigma = float(sigma)
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be positive and finite")

        n0 = x0.shape[0]
        n1 = x1.shape[0]
        m_centers = _resolve_m(self.config.M, n0 + n1)

        # This weight removes the finite sample-count prior and leaves the requested ratio.
        neg_weight = float(n1) / (float(n0) * target_norm_ratio)

        if isinstance(self.config.keops, bool):
            keops = "yes" if self.config.keops else "no"
        else:
            keops = str(self.config.keops)

        kernel = GaussianKernel(sigma)
        options = FalkonOptions(
            cg_tolerance=float(self.config.cg_tol),
            keops_active=keops,
            use_cpu=bool(self.config.cpu),
            debug=False,
        )

        self.model = LogisticFalkon(
            kernel=kernel,
            M=m_centers,
            penalty_list=list(self.config.penalty_list),
            iter_list=list(self.config.iter_list),
            loss=WeightedCrossEntropyLoss(kernel, neg_weight=neg_weight),
            seed=self.config.seed,
            options=options,
        )

        x_train = np.vstack([x0, x1])
        y_train = np.concatenate([
            np.zeros(n0, dtype=np.float64),
            np.ones(n1, dtype=np.float64),
        ]).reshape(-1, 1)

        if int(self.config.verbose) > 0:
            print(
                "[FalkonRatio] fit "
                f"n0={n0}, n1={n1}, sigma={sigma:.6g}, M={m_centers}, "
                f"neg_weight={neg_weight:.6g}"
            )

        self.model.fit(
            torch.from_numpy(np.ascontiguousarray(x_train)),
            torch.from_numpy(y_train),
        )

        self.sigma_ = sigma
        self.neg_weight_ = neg_weight
        self.target_norm_ratio_ = target_norm_ratio
        return self

    def predict_log_ratio(self, X):
        """Predict the learned log-ratio on new points.

        :param X: Evaluation sample with shape ``(n_samples,)`` or ``(n_samples, n_features)``.
        :returns: Predicted log-ratio with shape ``(n_samples,)``.
        """
        if self.model is None:
            raise RuntimeError("Fit the FalkonLogRatioEstimator before prediction")

        x = _as_2d_float64(X, "X")
        pred = self.model.predict(torch.from_numpy(x))
        return pred.detach().cpu().numpy().reshape(-1)

    def fit_predict(self, X_central, X_varied, X_eval, target_norm_ratio=1.0):
        """Fit a ratio model and return predictions on ``X_eval``.

        :param X_central: Sample from ``p_0`` with shape ``(n_central, n_features)``.
        :param X_varied: Sample from ``p_nu`` with shape ``(n_varied, n_features)``.
        :param X_eval: Evaluation sample with shape ``(n_eval, n_features)``.
        :param target_norm_ratio: Optional expected total-rate ratio.
        :returns: Predicted log-ratio with shape ``(n_eval,)``.
        """
        self.fit(X_central, X_varied, target_norm_ratio=target_norm_ratio)
        return self.predict_log_ratio(X_eval)
