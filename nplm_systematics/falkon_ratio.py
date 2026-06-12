"""Falkon density-ratio estimation for nuisance morphing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
from scipy.spatial.distance import pdist


@dataclass
class FalkonRatioConfig:
    """Configuration for a Falkon log-density-ratio fit.

    The fitted classifier score estimates

    ``log target_norm_ratio * p_varied(x) / p_central(x)``.

    With the default ``target_norm_ratio=1`` this is a normalized shape ratio.
    A non-unit ratio can encode expected-yield changes in the varied sample.
    """

    sigma: Optional[float] = None
    M: Union[int, str] = "sqrt"
    penalty_list: Sequence[float] = (1e-6,)
    iter_list: Sequence[int] = (1000,)
    cg_tol: float = float(np.sqrt(1e-7))
    keops: Union[str, bool] = "no"
    cpu: bool = True
    seed: Optional[int] = None
    verbose: int = 0

    @classmethod
    def from_mapping(cls, config: Optional[Mapping[str, Any]]) -> "FalkonRatioConfig":
        """Build a config from repo-style or native key names."""
        if config is None:
            return cls()

        cfg = dict(config)
        if "lambda" in cfg and "penalty_list" not in cfg:
            cfg["penalty_list"] = cfg.pop("lambda")
        if "iter" in cfg and "iter_list" not in cfg:
            cfg["iter_list"] = cfg.pop("iter")

        if not isinstance(cfg.get("penalty_list", ()), (list, tuple)):
            cfg["penalty_list"] = [cfg["penalty_list"]]
        if not isinstance(cfg.get("iter_list", ()), (list, tuple)):
            cfg["iter_list"] = [cfg["iter_list"]]

        allowed = {field.name for field in fields(cls)}
        cfg = {key: value for key, value in cfg.items() if key in allowed}
        return cls(**cfg)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly configuration dictionary."""
        out = asdict(self)
        out["penalty_list"] = list(out["penalty_list"])
        out["iter_list"] = list(out["iter_list"])
        return out


def _as_2d_float64(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty")
    return np.ascontiguousarray(arr)


def _resolve_m(M: Union[int, str], n_total: int) -> int:
    if isinstance(M, str):
        if M != "sqrt":
            raise ValueError("M must be an integer or the string 'sqrt'")
        return int(np.sqrt(n_total))
    M_int = int(M)
    if M_int <= 0:
        raise ValueError("M must be positive")
    return M_int


class FalkonLogRatioEstimator:
    """Estimate ``log p_varied(x) / p_central(x)`` with LogisticFalkon."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        self.config = FalkonRatioConfig.from_mapping(config)
        self.model = None
        self.sigma_: Optional[float] = None
        self.neg_weight_: Optional[float] = None
        self.target_norm_ratio_: Optional[float] = None

    @staticmethod
    def estimate_sigma_median(X: np.ndarray, max_points: int = 5000, seed: Optional[int] = None) -> float:
        """Estimate a Gaussian-kernel width from median pairwise distance."""
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

    def fit(
        self,
        X_central: np.ndarray,
        X_varied: np.ndarray,
        *,
        target_norm_ratio: float = 1.0,
    ) -> "FalkonLogRatioEstimator":
        """Fit a log-ratio model between central and nuisance-varied samples.

        Parameters
        ----------
        X_central:
            Sample from the central reference distribution ``p_0``.
        X_varied:
            Sample from the nuisance-varied distribution ``p_nu``.
        target_norm_ratio:
            Desired total-rate ratio ``N_nu / N_0``. The default learns the
            normalized shape ratio ``p_nu / p_0``.
        """
        x0 = _as_2d_float64(X_central, "X_central")
        x1 = _as_2d_float64(X_varied, "X_varied")
        if x0.shape[1] != x1.shape[1]:
            raise ValueError(f"Feature mismatch: {x0.shape[1]} vs {x1.shape[1]}")

        target_norm_ratio = float(target_norm_ratio)
        if not np.isfinite(target_norm_ratio) or target_norm_ratio <= 0:
            raise ValueError("target_norm_ratio must be positive and finite")

        try:
            import torch
            from falkon import LogisticFalkon
            from falkon.gsc_losses import WeightedCrossEntropyLoss
            from falkon.kernels import GaussianKernel
            from falkon.options import FalkonOptions
        except ImportError as exc:
            raise ImportError(
                "FalkonLogRatioEstimator requires torch and falkon. "
                "Install Falkon to train nuisance-ratio models."
            ) from exc

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

        # LogisticFalkon minimizes a weighted classification loss. This choice
        # removes the finite sample-count prior and leaves the requested ratio.
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

    def predict_log_ratio(self, X: np.ndarray) -> np.ndarray:
        """Predict the learned log-ratio on new points."""
        if self.model is None:
            raise RuntimeError("Fit the FalkonLogRatioEstimator before prediction")

        try:
            import torch
        except ImportError as exc:
            raise ImportError("Prediction requires torch.") from exc

        x = _as_2d_float64(X, "X")
        pred = self.model.predict(torch.from_numpy(x))
        return pred.detach().cpu().numpy().reshape(-1)

    def fit_predict(
        self,
        X_central: np.ndarray,
        X_varied: np.ndarray,
        X_eval: np.ndarray,
        *,
        target_norm_ratio: float = 1.0,
    ) -> np.ndarray:
        """Fit a ratio model and return predictions on ``X_eval``."""
        self.fit(X_central, X_varied, target_norm_ratio=target_norm_ratio)
        return self.predict_log_ratio(X_eval)
