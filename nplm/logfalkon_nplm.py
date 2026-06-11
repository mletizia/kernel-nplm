"""Train and evaluate the LogisticFalkon implementation of the NPLM statistic."""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import pdist

from falkon import LogisticFalkon
from falkon.gsc_losses import WeightedCrossEntropyLoss
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions


#########################################################################################################
# Public model wrapper

class LogFalkonNPLM:
    """NPLM test statistic computed with a LogisticFalkon classifier.

    :param config: Model configuration dictionary or path to a JSON/YAML file.
    :param output_path: Optional directory created before fitting.
    """

    DEFAULT_CONFIG = {
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

    def __init__(self, config=None, output_path=None):
        """Initialize the NPLM wrapper.

        :param config: Model configuration dictionary or config-file path.
        :param output_path: Optional output directory path.
        """
        self.output_path = str(output_path) if output_path is not None else None
        if self.output_path is not None:
            os.makedirs(self.output_path, exist_ok=True)

        cfg_dict = self._load_config(config) if config is not None else {}
        self.config = self._resolve_config(cfg_dict)

        self.model = None
        self.N_R = None
        self.N_D = None
        self.NR = None
        self.weight = None

        self._log("\n[NPLM] Initialized with config:")
        if self.config.get("verbose", 0) > 0:
            for k, v in self.config.items():
                print(f"  {k}: {v}")

    def _log(self, msg):
        """Print a message when verbose logging is enabled.

        :param msg: Message to print.
        :returns: ``None``.
        """
        if int(self.config.get("verbose", 0)) > 0:
            print(msg)

    @staticmethod
    def _load_config(config):
        """Load a model configuration from a dictionary or JSON/YAML file.

        :param config: Dictionary or path-like object.
        :returns: Configuration dictionary.
        """
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
    def _resolve_config(cls, user_cfg):
        """Merge user configuration with defaults and validate required values.

        :param user_cfg: User-specified configuration dictionary.
        :returns: Resolved configuration dictionary.
        """
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
    def estimate_sigma_median(X, max_points=5000, seed=None):
        """Estimate the Gaussian kernel width with the median pairwise distance.

        :param X: Input sample with shape ``(n_samples, n_features)``.
        :param max_points: Maximum number of points used by the heuristic.
        :param seed: Optional RNG seed used when subsampling.
        :returns: Positive finite median-distance estimate.
        """
        x = np.asarray(X)
        if x.ndim != 2:
            raise ValueError("X must be 2D array (N, d)")
        x = np.ascontiguousarray(x)

        n = x.shape[0]
        if n < 2:
            raise ValueError("Need at least 2 points to estimate sigma")

        max_points = int(max_points)
        if max_points < 2:
            raise ValueError("max_points must be >= 2")

        if n > max_points:
            rng = np.random.default_rng(None if seed is None else int(seed))
            idx = rng.choice(n, size=max_points, replace=False)
            x = x[idx]

        distances = pdist(x)
        sigma = float(np.median(distances))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Median heuristic returned invalid sigma={sigma}. Consider standardizing inputs.")
        return sigma

    @staticmethod
    def _sqrt_rule(n):
        """Return the square-root rule for the number of Nystrom centers.

        :param n: Number of input points.
        :returns: Integer square-root value.
        """
        return int(np.sqrt(n))

    def _set_seed(self):
        """Set NumPy and Torch global RNG seeds from the resolved config.

        :returns: ``None``.
        """
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
    def _normalize_labels_01(y):
        """Validate and flatten binary labels.

        :param y: Labels with shape ``(n_samples,)`` or ``(n_samples, 1)``.
        :returns: Float labels with shape ``(n_samples,)``.
        """
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        if y.ndim != 1:
            raise ValueError("y must be shape (N,) or (N,1)")
        vals = np.unique(y)
        if not np.all(np.isin(vals, [0, 1])):
            raise ValueError(f"y must contain only 0 and 1; got unique values {vals}")
        return y.astype(np.float64, copy=False)

    def _resolve_sizes_from_labels(self, y01):
        """Resolve class counts and loss weight from binary labels.

        :param y01: Flattened labels with shape ``(n_samples,)``.
        :returns: ``None``.
        """
        n_ref = int(np.sum(y01 == 0))
        n_data = int(np.sum(y01 == 1))

        if n_ref <= 0 or n_data <= 0:
            raise ValueError(f"Need both classes present: got N_R={n_ref}, N_D={n_data}")

        if self.config["N_R"] is not None and int(self.config["N_R"]) != n_ref:
            raise ValueError(f"Config N_R={self.config['N_R']} but labels imply N_R={n_ref}")
        if self.config["N_D"] is not None and int(self.config["N_D"]) != n_data:
            raise ValueError(f"Config N_D={self.config['N_D']} but labels imply N_D={n_data}")

        self.N_R = n_ref
        self.N_D = n_data
        self.NR = float(self.config["NR"])
        self.weight = self.NR / float(self.N_R)

        self._log(f"[NPLM] Label counts: N_R={self.N_R} (y=0), N_D={self.N_D} (y=1)")
        self._log(f"[NPLM] weight = NR/N_R = {self.NR}/{self.N_R} = {self.weight:.6g}")

    def build_model(self):
        """Build the LogisticFalkon classifier for the current split.

        :returns: ``None``.
        """
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

        m_cfg = self.config["M"]
        if isinstance(m_cfg, str):
            if m_cfg != "sqrt":
                raise ValueError("config['M'] must be an int or the string 'sqrt'")
            m_centers = self._sqrt_rule(self.N_R + self.N_D)
        else:
            m_centers = int(m_cfg)

        self._log(f"[NPLM] Using fixed sigma = {sigma:.6g}")
        self._log(f"[NPLM] Nyström centers M = {m_centers}")

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
            M=m_centers,
            options=opts,
            loss=WeightedCrossEntropyLoss(kernel, neg_weight=float(self.weight)),
            seed=self.config.get("seed", None),
        )

    @staticmethod
    def _compute_t(scores, y01, weight):
        """Compute the NPLM test statistic from classifier scores.

        :param scores: Model scores with shape ``(n_samples,)`` or ``(n_samples, 1)``.
        :param y01: Flattened labels with shape ``(n_samples,)``.
        :param weight: Reference-event weight ``NR / N_R``.
        :returns: Scalar NPLM statistic.
        """
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores.reshape(-1)

        mask_ref_t = torch.from_numpy(y01 == 0).to(scores.device)
        mask_data_t = torch.from_numpy(y01 == 1).to(scores.device)

        s_ref = scores[mask_ref_t]
        s_data = scores[mask_data_t]

        diff = float(weight) * torch.sum(1.0 - torch.exp(s_ref))
        t = 2.0 * (diff + torch.sum(s_data))
        return float(t.item())

    def compute_statistic(self, X, y, return_details=False):
        """Train the NPLM classifier and return the test statistic.

        :param X: Pooled sample with shape ``(n_samples, n_features)``.
        :param y: Binary labels with shape ``(n_samples,)`` or ``(n_samples, 1)``.
        :param return_details: If true, also return diagnostics and scores.
        :returns: Statistic, or ``(statistic, details)`` when requested.
        """
        x = np.asarray(X, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("X must be 2D array (N, d)")
        x = np.ascontiguousarray(x)

        y01 = self._normalize_labels_01(y)
        if y01.shape[0] != x.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        self._log("\n[NPLM] Starting computation of test statistic")
        self._set_seed()
        self._resolve_sizes_from_labels(y01)

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y01.reshape(-1, 1)).to(dtype=torch.float64)

        self.build_model()

        self._log("[NPLM] Training LogisticFalkon...")
        t0 = time.time()
        self.model.fit(x_t, y_t)
        train_time = time.time() - t0
        self._log(f"[NPLM] Training finished in {train_time:.2f} s")

        scores = self.model.predict(x_t)
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
