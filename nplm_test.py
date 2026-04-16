from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import pdist
from scipy.stats import chi2, norm

from falkon import LogisticFalkon
from falkon.gsc_losses import WeightedCrossEntropyLoss
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions

ConfigLike = Dict[str, Any]
ConfigInput = Union[ConfigLike, str, Path]


def plot_reconstruction_1d(
    *,
    df,
    data_1d,
    weight_data,
    ref_1d,
    weight_ref,
    t_obs,
    ref_scores,
    bins=24,
    x_range=(0.0, 1.5),
    logy=True,
    ratio_ylim=(0.0, 10.0),
    eps=1e-10,
    var_name="x",
    save=False,
    save_path="",
    file_name="",
    figsize=(8, 8),
    show=True,
):
    data_1d = np.asarray(data_1d).reshape(-1)
    ref_1d = np.asarray(ref_1d).reshape(-1)

    if np.isscalar(weight_ref):
        w_ref = np.full(len(ref_1d), float(weight_ref))
    else:
        w_ref = np.asarray(weight_ref)

    if np.isscalar(weight_data):
        w_data = np.full(len(data_1d), float(weight_data))
    else:
        w_data = np.asarray(weight_data)

    ref_scores = np.asarray(ref_scores)
    if ref_scores.ndim == 2:
        ref_scores = ref_scores[:, 0]

    if np.isscalar(bins):
        edges = np.linspace(x_range[0], x_range[1], bins + 1)
    else:
        edges = np.asarray(bins)

    centers = 0.5 * (edges[1:] + edges[:-1])

    hD, _ = np.histogram(data_1d, bins=edges, weights=w_data)
    hR, _ = np.histogram(ref_1d, bins=edges, weights=w_ref)
    hN, _ = np.histogram(ref_1d, bins=edges, weights=np.exp(ref_scores) * w_ref)

    yerr = np.sqrt(np.maximum(hD, 0.0))

    ztxt = ""
    if df is not None:
        Z = norm.ppf(chi2.cdf(float(t_obs), int(df)))
        ztxt = f", Z-score={Z:.2f}"

    fig = plt.figure(figsize=figsize)
    ax_top = fig.add_axes([0.1, 0.43, 0.8, 0.5])
    ax_ratio = fig.add_axes([0.1, 0.1, 0.8, 0.3], sharex=ax_top)

    ax_top.hist(data_1d, bins=edges, weights=w_data, histtype="step", color="black", lw=1.5, label="DATA", zorder=2)
    ax_top.hist(ref_1d, bins=edges, weights=w_ref, color="#a6cee3", edgecolor="#1f78b4", lw=1, label="REFERENCE", zorder=1)
    ax_top.errorbar(centers, hD, yerr=yerr, color="black", ls="", marker="o", ms=5, zorder=3)
    ax_top.scatter(centers, hN, edgecolor="black", color="#b2df8a", lw=1, s=30, label="RECO", zorder=4)

    title = f"t={float(t_obs):.2f}{ztxt}"
    ax_top.legend(title=title, ncol=2)
    ax_top.set_ylabel("events")
    ax_top.set_xlim(edges[0], edges[-1])
    if logy:
        ax_top.set_yscale("log")
    ax_top.tick_params(axis="x", labelbottom=False)

    denom = hR + eps
    ax_ratio.errorbar(centers, hD / denom, yerr=yerr / denom, ls="", marker="o", color="black", label="DATA/REF")
    ax_ratio.plot(centers, hN / denom, color="#b2df8a", lw=3, label="RECO")

    ax_ratio.set_xlabel(var_name)
    ax_ratio.set_ylabel("ratio")
    ax_ratio.set_ylim(*ratio_ylim)
    ax_ratio.set_xlim(edges[0], edges[-1])
    ax_ratio.grid()
    ax_ratio.legend()

    if save:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, file_name), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, (ax_top, ax_ratio)


def plot_reconstruction_marginals(
    *,
    df,
    data,
    weight_data,
    ref,
    weight_ref,
    t_obs,
    ref_scores,
    feature_names=None,
    bins=24,
    x_ranges=None,
    save=False,
    save_path="",
    file_prefix="reco_dim",
    show=True,
    **kwargs,
):
    data = np.asarray(data)
    ref = np.asarray(ref)

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)

    if data.shape[1] != ref.shape[1]:
        raise ValueError("data and ref must have the same dimensionality")

    d = data.shape[1]
    if feature_names is None:
        feature_names = [f"x[{j}]" for j in range(d)]
    if len(feature_names) != d:
        raise ValueError("feature_names must have length d")

    if x_ranges is None:
        x_ranges = [None] * d
    elif len(x_ranges) != d:
        raise ValueError("x_ranges must be None or have length d")

    figs = []
    for j in range(d):
        xr = x_ranges[j]
        if xr is None:
            lo = float(np.min([data[:, j].min(), ref[:, j].min()]))
            hi = float(np.max([data[:, j].max(), ref[:, j].max()]))
            pad = 1e-6 + 0.02 * (hi - lo if hi > lo else 1.0)
            xr = (lo - pad, hi + pad)

        fname = ""
        if save:
            fname = f"{file_prefix}_{j}.pdf"

        fig, _ = plot_reconstruction_1d(
            df=df,
            data_1d=data[:, j],
            weight_data=weight_data,
            ref_1d=ref[:, j],
            weight_ref=weight_ref,
            t_obs=t_obs,
            ref_scores=ref_scores,
            bins=bins,
            x_range=xr,
            var_name=feature_names[j],
            save=save,
            save_path=save_path,
            file_name=fname,
            show=show,
            **kwargs,
        )
        figs.append(fig)

    return figs


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
        "cg_tol": 1e-7,
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

        mask_ref = (y01 == 0)
        mask_data = (y01 == 1)

        mask_ref_t = torch.from_numpy(mask_ref)
        mask_data_t = torch.from_numpy(mask_data)

        s_ref = scores[mask_ref_t]
        s_data = scores[mask_data_t]

        diff = float(weight) * torch.sum(1.0 - torch.exp(s_ref))
        t = 2.0 * (diff + torch.sum(s_data))
        return float(t.item())

    def compute_statistic(self, X: np.ndarray, y: np.ndarray, return_details: bool = False):
        X = np.asarray(X)
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
        y_t = torch.from_numpy(y01.reshape(-1, 1)).to(dtype=X_t.dtype)

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
        mask_ref_t = torch.from_numpy(y01 == 0)
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


def sample_ref_exp(N, rate=8.0, xmax=1.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    Z = 1.0 - np.exp(-rate * xmax)
    u = rng.random(N)
    return -(1.0 / rate) * np.log(1.0 - u * Z)


def sample_signal_gauss(N, mu=0.8, sigma=0.02, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return rng.normal(loc=mu, scale=sigma, size=N)


def make_data_sample_poisson(NR=2000, NS=10, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    N_B = int(rng.poisson(NR))
    N_S = int(rng.poisson(NS))

    xb = sample_ref_exp(N_B, rng=rng)
    xs = sample_signal_gauss(N_S, rng=rng)

    x = np.concatenate([xb, xs])
    rng.shuffle(x)
    return x, xb, xs, N_B, N_S
