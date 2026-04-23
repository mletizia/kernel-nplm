from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from scipy.spatial.distance import pdist

from falkon import LogisticFalkon
from falkon.gsc_losses import WeightedCrossEntropyLoss
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions

ConfigLike = Dict[str, Any]
ConfigInput = Union[ConfigLike, str, Path]


class LogFalkonNPLM:
    """
    NPLM test via LogisticFalkon classifier.

    Input API:
      - X: (N, d) pooled sample
      - y: (N,) or (N,1) labels in {0, 1}
           0 = reference (count N_R)
           1 = data      (count N_D)
    """

    DEFAULT_CONFIG: ConfigLike = {
        "sigma": None,
        "M": "sqrt",
        "lambda": [1e-6],
        "iter": [1_000_000],
        "cg_tol": np.sqrt(1e-7),
        "keops": "no",
        "N_R": None,
        "N_D": None,
        "NR": None,
        "seed": None,
        "cpu": False,
        "verbose": 1,
    }

    def __init__(self, config: Optional[ConfigInput] = None, output_path: Optional[Union[str, Path]] = None):
        self.output_path = str(output_path) if output_path is not None else None
        if self.output_path is not None:
            os.makedirs(self.output_path, exist_ok=True)

        cfg_dict = self._load_config(config) if config is not None else {}
        self.config = self._resolve_config(cfg_dict)

        self.model: Optional[LogisticFalkon] = None
        self.N_R: Optional[int] = None
        self.N_D: Optional[int] = None
        self.NR: Optional[float] = None
        self.weight: Optional[float] = None

        self._log("\n[NPLM] Initialized with config:")
        if self.config.get("verbose", 0) > 0:
            for k, v in self.config.items():
                print(f"  {k}: {v}")

    def _log(self, msg: str) -> None:
        if int(self.config.get("verbose", 0)) > 0:
            print(msg)

    @staticmethod
    def _load_config(config: ConfigInput) -> ConfigLike:
        if isinstance(config, dict):
            return dict(config)

        path = Path(config)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()

        if suffix == ".json":
            return json.loads(text)
        if suffix in (".yml", ".yaml"):
            try:
                import yaml
            except ImportError as e:
                raise ImportError("YAML config requires `pip install pyyaml`.") from e
            return yaml.safe_load(text)

        raise ValueError("Config must be dict, .json, .yml or .yaml")

    @classmethod
    def _resolve_config(cls, user_cfg: ConfigLike) -> ConfigLike:
        cfg = dict(cls.DEFAULT_CONFIG)
        cfg.update(user_cfg)

        if isinstance(cfg["keops"], bool):
            cfg["keops"] = "yes" if cfg["keops"] else "no"

        if not isinstance(cfg["lambda"], (list, tuple)):
            cfg["lambda"] = [cfg["lambda"]]
        if not isinstance(cfg["iter"], (list, tuple)):
            cfg["iter"] = [cfg["iter"]]

        cfg["verbose"] = int(cfg.get("verbose", 0))

        if cfg.get("NR", None) is None:
            raise ValueError("Config must specify NR (expected data size under H0).")

        if cfg.get("sigma", None) is not None:
            sigma = float(cfg["sigma"])
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError(f"Config sigma must be positive finite; got sigma={sigma}")

        return cfg

    @staticmethod
    def estimate_sigma_median(X: np.ndarray, max_points: int = 5000, seed: Optional[int] = None) -> float:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D array (N, d)")
        X = np.ascontiguousarray(X)

        n = X.shape[0]
        if n < 2:
            raise ValueError("Need at least 2 points to estimate sigma")

        max_points = int(max_points)
        if max_points < 2:
            raise ValueError("max_points must be >= 2")

        if n > max_points:
            rng = np.random.default_rng(None if seed is None else int(seed))
            idx = rng.choice(n, size=max_points, replace=False)
            X = X[idx]

        d = pdist(X)
        sigma = float(np.median(d))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Median heuristic returned invalid sigma={sigma}. Consider standardizing inputs.")
        return sigma

    @staticmethod
    def _sqrt_rule(n: int) -> int:
        return int(np.sqrt(n))

    def _set_seed(self) -> None:
        seed = self.config.get("seed", None)
        if seed is None:
            self._log("[NPLM] No seed specified (non-reproducible run)")
            return
        seed = int(seed)
        self._log(f"[NPLM] Setting global seed = {seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _normalize_labels_01(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        if y.ndim != 1:
            raise ValueError("y must be shape (N,) or (N,1)")
        vals = np.unique(y)
        if not np.all(np.isin(vals, [0, 1])):
            raise ValueError(f"y must contain only 0 and 1; got unique values {vals}")
        return y.astype(np.float64, copy=False)

    def _resolve_sizes_from_labels(self, y01: np.ndarray) -> None:
        N_R = int(np.sum(y01 == 0))
        N_D = int(np.sum(y01 == 1))

        if N_R <= 0 or N_D <= 0:
            raise ValueError(f"Need both classes present: got N_R={N_R}, N_D={N_D}")

        if self.config["N_R"] is not None and int(self.config["N_R"]) != N_R:
            raise ValueError(f"Config N_R={self.config['N_R']} but labels imply N_R={N_R}")
        if self.config["N_D"] is not None and int(self.config["N_D"]) != N_D:
            raise ValueError(f"Config N_D={self.config['N_D']} but labels imply N_D={N_D}")

        self.N_R = N_R
        self.N_D = N_D
        self.NR = float(self.config["NR"])
        self.weight = self.NR / float(self.N_R)

        self._log(f"[NPLM] Label counts: N_R={self.N_R} (y=0), N_D={self.N_D} (y=1)")
        self._log(f"[NPLM] weight = NR/N_R = {self.NR}/{self.N_R} = {self.weight:.6g}")

    def build_model(self) -> None:
        if self.N_R is None or self.N_D is None or self.weight is None:
            raise RuntimeError("Internal state not initialized. Call compute_statistic() first.")

        sigma = self.config.get("sigma", None)
        if sigma is None:
            raise ValueError(
                "Config must specify sigma (float) to train the model. "
                "You can estimate it once with `estimate_sigma_median` and then set config['sigma']."
            )
        sigma = float(sigma)

        kernel = GaussianKernel(sigma)

        M_cfg = self.config["M"]
        if isinstance(M_cfg, str):
            if M_cfg != "sqrt":
                raise ValueError("config['M'] must be an int or the string 'sqrt'")
            M = self._sqrt_rule(self.N_R + self.N_D)
        else:
            M = int(M_cfg)

        self._log(f"[NPLM] Using fixed sigma = {sigma:.6g}")
        self._log(f"[NPLM] Nyström centers M = {M}")

        opts = FalkonOptions(
            cg_tolerance=float(self.config["cg_tol"]),
            keops_active=str(self.config["keops"]),
            use_cpu=bool(self.config["cpu"]),
            debug=False,
        )

        self.model = LogisticFalkon(
            kernel=kernel,
            penalty_list=self.config["lambda"],
            iter_list=self.config["iter"],
            M=M,
            options=opts,
            loss=WeightedCrossEntropyLoss(kernel, neg_weight=float(self.weight)),
            seed=self.config.get("seed", None),
        )

    @staticmethod
    def _compute_t(scores: torch.Tensor, y01: np.ndarray, weight: float) -> float:
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores.reshape(-1)

        mask_ref_t = torch.from_numpy(y01 == 0).to(scores.device)
        mask_data_t = torch.from_numpy(y01 == 1).to(scores.device)

        s_ref = scores[mask_ref_t]
        s_data = scores[mask_data_t]

        diff = float(weight) * torch.sum(1.0 - torch.exp(s_ref))
        t = 2.0 * (diff + torch.sum(s_data))
        return float(t.item())

    def compute_statistic(self, X: np.ndarray, y: np.ndarray, return_details: bool = False):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2D array (N, d)")
        X = np.ascontiguousarray(X)

        y01 = self._normalize_labels_01(y)
        if y01.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        self._log("\n[NPLM] Starting computation of test statistic")
        self._set_seed()
        self._resolve_sizes_from_labels(y01)

        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y01.reshape(-1, 1)).to(dtype=torch.float64)

        self.build_model()

        self._log("[NPLM] Training LogisticFalkon...")
        t0 = time.time()
        self.model.fit(X_t, y_t)
        train_time = time.time() - t0
        self._log(f"[NPLM] Training finished in {train_time:.2f} s")

        scores = self.model.predict(X_t)
        t = self._compute_t(scores, y01, float(self.weight))

        self._log(f"[NPLM] Test statistic t = {t:.6g}")

        if not return_details:
            return t

        scores_flat = scores.reshape(-1)
        mask_ref_t = torch.from_numpy(y01 == 0).to(scores_flat.device)
        s_ref = scores_flat[mask_ref_t]
        Nw = float((float(self.weight) * torch.sum(torch.exp(s_ref))).item())

        return t, {
            "Nw": Nw,
            "train_time": float(train_time),
            "weight": float(self.weight),
            "sigma": float(self.config["sigma"]),
            "seed": self.config.get("seed", None),
            "scores": scores.detach().cpu().numpy(),
            "resolved_config": dict(self.config),
            "N_R": int(self.N_R),
            "N_D": int(self.N_D),
        }
