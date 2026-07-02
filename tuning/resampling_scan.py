"""Hyperparameter scans built on the NPLM reference-resampling null."""

import time
from collections import namedtuple
from pathlib import Path

import numpy as np

from stat_tests import nplm_resampling_test


#########################################################################################################
# Constants and result containers

_MAX_SEED = 2**32 - 1


NPLMHyperparameterScanResult = namedtuple(
    "NPLMHyperparameterScanResult",
    [
        "raw_results",
        "summary",
        "lambda_values",
        "m_values",
        "sigma",
        "n_ref",
        "n_data",
        "poisson_fluctuate_n_data",
        "n_trials",
        "null_sampling",
        "seed",
        "paired_toys",
        "resample_nystrom",
        "model_config",
    ],
)
NPLMHyperparameterScanResult.__doc__ = "Result of a resampling-based NPLM hyperparameter scan."


#########################################################################################################
# Public tuning API

def nplm_resampling_hyperparameter_scan(
    x_ref,
    model_config,
    *,
    lambda_values,
    m_values,
    n_ref,
    n_data,
    n_trials=10,
    null_sampling="disjoint",
    poisson_fluctuate_n_data=False,
    seed=0,
    paired_toys=True,
    resample_nystrom=True,
    reference_factor_warning_threshold=10.0,
    progress=False,
    dtype=np.float64,
):
    """Scan ``lambda`` and ``M`` using reference-resampled null toys.

    :param x_ref: Reference pool with shape ``(n_ref_pool, n_features)``.
    :param model_config: Base ``LogFalkonNPLM`` configuration with fixed ``sigma``.
    :param lambda_values: Positive Falkon penalty values to scan.
    :param m_values: Positive integer Nystrom-center counts to scan.
    :param n_ref: Reference sample size used in each NPLM fit.
    :param n_data: Pseudo-data sample size used in each null toy.
    :param n_trials: Number of null statistics computed per grid point. Defaults to ``10``.
    :param null_sampling: Resampling mode passed to ``nplm_resampling_test``.
    :param poisson_fluctuate_n_data: If true, draw each toy data count from ``Poisson(n_data)``.
    :param seed: Local RNG seed. Reused for every grid point when ``paired_toys=True``.
    :param paired_toys: If true, use the same resampled toys at every grid point.
    :param resample_nystrom: If true, draw fresh model seeds across null toys.
    :param reference_factor_warning_threshold: Pool-size warning threshold passed to resampling.
    :param progress: If true, print one progress line per grid point.
    :param dtype: Floating dtype used by the resampling test.
    :returns: ``NPLMHyperparameterScanResult`` with raw rows and grouped summaries.
    """
    n_ref = _validate_positive_int(n_ref, name="n_ref")
    n_data = _validate_positive_int(n_data, name="n_data")
    n_trials = _validate_positive_int(n_trials, name="n_trials")
    lambda_values = _validate_positive_float_values(lambda_values, name="lambda_values")
    m_values = _validate_positive_int_values(m_values, name="m_values")
    base_config, sigma = _validate_model_config(model_config, n_data=n_data)

    raw_results = []
    point_summaries = []
    point_seeds = _make_point_seeds(
        n_points=len(lambda_values) * len(m_values),
        seed=seed,
        paired_toys=paired_toys,
    )

    point_idx = 0
    for lam in lambda_values:
        for m_value in m_values:
            point_seed = point_seeds[point_idx]
            point_idx += 1

            if progress:
                print(
                    "[tuning] "
                    f"lambda={lam:g} M={m_value} "
                    f"trials={n_trials} seed={point_seed}"
                )

            start_time = time.perf_counter()
            result = nplm_resampling_test(
                x_ref,
                x_ref,
                _make_point_config(
                    base_config=base_config,
                    sigma=sigma,
                    lam=lam,
                    m_value=m_value,
                ),
                n_ref=n_ref,
                n_data=n_data,
                n_null=n_trials,
                n_alternative=0,
                null_sampling=null_sampling,
                poisson_fluctuate_n_data=poisson_fluctuate_n_data,
                seed=point_seed,
                resample_nystrom=resample_nystrom,
                return_null=True,
                return_alternative=False,
                return_observed=False,
                reference_factor_warning_threshold=reference_factor_warning_threshold,
                dtype=dtype,
            )
            elapsed = time.perf_counter() - start_time

            rows = _make_raw_rows(
                resampling_result=result,
                lam=lam,
                m_value=m_value,
                sigma=sigma,
                point_seed=point_seed,
            )
            raw_results.extend(rows)

            point_summaries.append(
                _summarize_point(
                    rows=rows,
                    elapsed_sec=elapsed,
                    null_sampling=null_sampling,
                )
            )

    return NPLMHyperparameterScanResult(
        raw_results=raw_results,
        summary=summarize_scan_rows(raw_results, point_summaries=point_summaries),
        lambda_values=list(lambda_values),
        m_values=list(m_values),
        sigma=float(sigma),
        n_ref=int(n_ref),
        n_data=int(n_data),
        poisson_fluctuate_n_data=bool(poisson_fluctuate_n_data),
        n_trials=int(n_trials),
        null_sampling=null_sampling,
        seed=None if seed is None else int(seed),
        paired_toys=bool(paired_toys),
        resample_nystrom=bool(resample_nystrom),
        model_config=dict(base_config),
    )


def summarize_scan_rows(raw_results, point_summaries=None):
    """Summarize raw tuning rows by ``(lambda, M)``.

    :param raw_results: Raw rows from ``nplm_resampling_hyperparameter_scan``.
    :param point_summaries: Optional timing summaries from the scan loop.
    :returns: List of summary dictionaries sorted by ``lambda`` and ``M``.
    """
    groups = {}
    for row in raw_results:
        key = (float(row["lambda"]), int(row["M"]))
        groups.setdefault(key, []).append(row)

    timing_by_key = {}
    if point_summaries is not None:
        for row in point_summaries:
            key = (float(row["lambda"]), int(row["M"]))
            timing_by_key[key] = row

    summary = []
    for key in sorted(groups):
        rows = groups[key]
        t_values = np.asarray([row["t_nplm"] for row in rows], dtype=np.float64)
        n_data_values = np.asarray([row["n_data"] for row in rows], dtype=np.float64)
        t_std = _sample_std(t_values)
        n_data_std = _sample_std(n_data_values)
        sqrt_n = np.sqrt(len(rows))
        lam, m_value = key

        row = {
            "lambda": float(lam),
            "M": int(m_value),
            "n_trials": int(len(rows)),
            "t_mean": float(np.mean(t_values)),
            "t_std": float(t_std),
            "t_sem": float(t_std / sqrt_n),
            "n_data_mean": float(np.mean(n_data_values)),
            "n_data_std": float(n_data_std),
        }
        if key in timing_by_key:
            row["elapsed_sec"] = float(timing_by_key[key]["elapsed_sec"])
            row["time_per_trial_sec"] = float(timing_by_key[key]["time_per_trial_sec"])
        summary.append(row)

    return summary


def plot_average_test_statistic(
    scan_result,
    *,
    ax=None,
    include_sem=True,
    logx=False,
    save_path=None,
):
    """Plot average null test statistic versus ``M`` for each ``lambda``.

    :param scan_result: ``NPLMHyperparameterScanResult`` or a summary-row list.
    :param ax: Optional Matplotlib axes.
    :param include_sem: If true, draw standard-error bars around the mean.
    :param logx: If true, use a logarithmic x-axis.
    :param save_path: Optional path where the figure is saved.
    :returns: Pair ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    summary = scan_result.summary if hasattr(scan_result, "summary") else scan_result
    summary = list(summary)
    if len(summary) == 0:
        raise ValueError("scan_result has no summary rows to plot")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    lambda_values = sorted({float(row["lambda"]) for row in summary})
    for lam in lambda_values:
        rows = [row for row in summary if float(row["lambda"]) == lam]
        rows = sorted(rows, key=lambda row: int(row["M"]))

        x = np.asarray([int(row["M"]) for row in rows], dtype=np.float64)
        y = np.asarray([float(row["t_mean"]) for row in rows], dtype=np.float64)
        yerr = (
            np.asarray([float(row["t_sem"]) for row in rows], dtype=np.float64)
            if include_sem
            else None
        )

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=3 if include_sem else 0,
            label=f"lambda={lam:g}",
        )

    ax.set_xlabel("M")
    ax.set_ylabel("Average test statistic")
    ax.set_title("NPLM reference-resampling hyperparameter scan")
    ax.grid(True, alpha=0.3)
    if logx:
        ax.set_xscale("log")
    ax.legend()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def plot_average_test_statistic_heatmap(
    scan_result,
    *,
    ax=None,
    save_path=None,
):
    """Plot average null test statistic as a ``(lambda, M)`` heatmap.

    :param scan_result: ``NPLMHyperparameterScanResult`` or a summary-row list.
    :param ax: Optional Matplotlib axes.
    :param save_path: Optional path where the figure is saved.
    :returns: Pair ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    summary = scan_result.summary if hasattr(scan_result, "summary") else scan_result
    summary = list(summary)
    if len(summary) == 0:
        raise ValueError("scan_result has no summary rows to plot")

    lambda_values = sorted({float(row["lambda"]) for row in summary})
    m_values = sorted({int(row["M"]) for row in summary})
    value_by_key = {
        (float(row["lambda"]), int(row["M"])): float(row["t_mean"])
        for row in summary
    }

    matrix = np.full((len(lambda_values), len(m_values)), np.nan, dtype=np.float64)
    for lambda_idx, lam in enumerate(lambda_values):
        for m_idx, m_value in enumerate(m_values):
            matrix[lambda_idx, m_idx] = value_by_key.get((lam, m_value), np.nan)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    image = ax.imshow(matrix, aspect="auto", origin="lower")
    fig.colorbar(image, ax=ax, label="Average test statistic")

    ax.set_xticks(np.arange(len(m_values)))
    ax.set_xticklabels([str(m_value) for m_value in m_values])
    ax.set_yticks(np.arange(len(lambda_values)))
    ax.set_yticklabels([f"{lam:g}" for lam in lambda_values])
    ax.set_xlabel("M")
    ax.set_ylabel("lambda")
    ax.set_title("NPLM reference-resampling scan")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def plot_training_time(
    scan_result,
    *,
    ax=None,
    logx=False,
    save_path=None,
):
    """Plot average wall time per NPLM fit versus ``M`` for each ``lambda``.

    :param scan_result: ``NPLMHyperparameterScanResult`` or a summary-row list.
    :param ax: Optional Matplotlib axes.
    :param logx: If true, use a logarithmic x-axis.
    :param save_path: Optional path where the figure is saved.
    :returns: Pair ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt

    summary = scan_result.summary if hasattr(scan_result, "summary") else scan_result
    summary = list(summary)
    if len(summary) == 0:
        raise ValueError("scan_result has no summary rows to plot")

    missing = [row for row in summary if "time_per_trial_sec" not in row]
    if missing:
        raise ValueError("summary rows do not contain time_per_trial_sec")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    lambda_values = sorted({float(row["lambda"]) for row in summary})
    for lam in lambda_values:
        rows = [row for row in summary if float(row["lambda"]) == lam]
        rows = sorted(rows, key=lambda row: int(row["M"]))

        x = np.asarray([int(row["M"]) for row in rows], dtype=np.float64)
        y = np.asarray([float(row["time_per_trial_sec"]) for row in rows], dtype=np.float64)

        ax.plot(
            x,
            y,
            marker="o",
            label=f"lambda={lam:g}",
        )

    ax.set_xlabel("M")
    ax.set_ylabel("Average training time per toy [s]")
    ax.set_title("NPLM training time in resampling scan")
    ax.grid(True, alpha=0.3)
    if logx:
        ax.set_xscale("log")
    ax.legend()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def save_scan_plots(
    scan_result,
    output_dir,
    *,
    prefix="nplm_resampling_tuning",
    include_sem=True,
    logx=False,
):
    """Save the standard tuning-inspection plots.

    :param scan_result: ``NPLMHyperparameterScanResult``.
    :param output_dir: Directory where plots are written.
    :param prefix: Filename prefix for saved plots.
    :param include_sem: If true, draw standard-error bars on the line plot.
    :param logx: If true, use logarithmic ``M`` axis on the line plot.
    :returns: Dictionary mapping plot names to saved paths.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "average_statistic": output_dir / f"{prefix}_average_statistic_vs_m.png",
        "average_statistic_heatmap": output_dir / f"{prefix}_average_statistic_heatmap.png",
        "training_time": output_dir / f"{prefix}_training_time_vs_m.png",
    }

    fig, _ = plot_average_test_statistic(
        scan_result,
        include_sem=include_sem,
        logx=logx,
        save_path=paths["average_statistic"],
    )
    plt.close(fig)

    fig, _ = plot_average_test_statistic_heatmap(
        scan_result,
        save_path=paths["average_statistic_heatmap"],
    )
    plt.close(fig)

    fig, _ = plot_training_time(
        scan_result,
        logx=logx,
        save_path=paths["training_time"],
    )
    plt.close(fig)

    return paths


#########################################################################################################
# Validation helpers

def _validate_positive_int(value, *, name):
    """Validate a positive integer input.

    :param value: Input value.
    :param name: Name used in validation errors.
    :returns: Validated positive integer.
    """
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _validate_positive_float_values(values, *, name):
    """Validate a scalar or sequence of positive finite floats.

    :param values: Input scalar or sequence.
    :param name: Name used in validation errors.
    :returns: List of positive finite floats.
    """
    values = _coerce_sequence(values, name=name)
    parsed = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{name} values must be positive finite floats")
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} values must be positive finite floats")
        parsed.append(value)
    return parsed


def _validate_positive_int_values(values, *, name):
    """Validate a scalar or sequence of positive integer values.

    :param values: Input scalar or sequence.
    :param name: Name used in validation errors.
    :returns: List of positive integers.
    """
    values = _coerce_sequence(values, name=name)
    parsed = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{name} values must be positive integers")
        numeric = float(value)
        integer = int(numeric)
        if not np.isfinite(numeric) or numeric < 1 or numeric != integer:
            raise ValueError(f"{name} values must be positive integers")
        parsed.append(integer)
    return parsed


def _coerce_sequence(values, *, name):
    """Coerce scalar or iterable values into a non-empty list.

    :param values: Input scalar or iterable.
    :param name: Name used in validation errors.
    :returns: Non-empty list.
    """
    if values is None:
        raise ValueError(f"{name} must not be None")

    if isinstance(values, str) or np.isscalar(values):
        values = [values]
    else:
        values = list(values)

    if len(values) == 0:
        raise ValueError(f"{name} must be non-empty")
    return values


def _validate_model_config(model_config, *, n_data):
    """Validate and normalize the fixed-sigma base model configuration.

    :param model_config: User model configuration dictionary.
    :param n_data: Pseudo-data sample size, used as default ``NR``.
    :returns: Pair ``(base_config, sigma)``.
    """
    if not isinstance(model_config, dict):
        raise ValueError("model_config must be a dictionary")

    base_config = dict(model_config)
    if base_config.get("sigma", None) is None:
        raise ValueError("model_config must specify fixed sigma")

    sigma = float(base_config["sigma"])
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"model_config sigma must be positive finite; got {sigma}")

    if base_config.get("NR", None) is None:
        base_config["NR"] = float(n_data)
    nr = float(base_config["NR"])
    if not np.isfinite(nr) or nr <= 0:
        raise ValueError(f"model_config NR must be positive finite; got {nr}")

    base_config.pop("lambda", None)
    base_config.pop("M", None)
    return base_config, sigma


#########################################################################################################
# Scan helpers

def _make_point_seeds(*, n_points, seed, paired_toys):
    """Build deterministic point seeds for the scan.

    :param n_points: Number of grid points.
    :param seed: Base seed, or ``None``.
    :param paired_toys: Whether every grid point should share the same toys.
    :returns: List of point seeds.
    """
    if paired_toys:
        if seed is not None:
            point_seed = int(seed)
        else:
            rng = np.random.default_rng(None)
            point_seed = int(rng.integers(0, _MAX_SEED, dtype=np.uint32))
        return [point_seed] * n_points

    rng = np.random.default_rng(seed)
    return [
        int(rng.integers(0, _MAX_SEED, dtype=np.uint32))
        for _ in range(n_points)
    ]


def _make_point_config(*, base_config, sigma, lam, m_value):
    """Build the model configuration for one scan point.

    :param base_config: Base model configuration.
    :param sigma: Fixed kernel width.
    :param lam: Falkon penalty value.
    :param m_value: Number of Nystrom centers.
    :returns: Model configuration dictionary.
    """
    config = dict(base_config)
    config["sigma"] = float(sigma)
    config["lambda"] = [float(lam)]
    config["M"] = int(m_value)
    return config


def _make_raw_rows(*, resampling_result, lam, m_value, sigma, point_seed):
    """Convert a resampling-test result into raw tuning rows.

    :param resampling_result: Result returned by ``nplm_resampling_test``.
    :param lam: Falkon penalty value.
    :param m_value: Number of Nystrom centers.
    :param sigma: Fixed kernel width.
    :param point_seed: Seed used for this grid point.
    :returns: List of raw row dictionaries.
    """
    rows = []
    for trial, statistic in enumerate(resampling_result.null_statistics):
        n_data = int(resampling_result.n_data)
        if hasattr(resampling_result, "null_data_counts"):
            n_data = int(resampling_result.null_data_counts[trial])
        rows.append(
            {
                "lambda": float(lam),
                "M": int(m_value),
                "sigma": float(sigma),
                "trial": int(trial),
                "t_nplm": float(statistic),
                "n_ref": int(resampling_result.n_ref),
                "n_data": n_data,
                "point_seed": None if point_seed is None else int(point_seed),
                "model_seed": int(resampling_result.null_seeds[trial]),
            }
        )
    return rows


def _summarize_point(*, rows, elapsed_sec, null_sampling):
    """Build a timing summary for one scan point.

    :param rows: Raw rows for one scan point.
    :param elapsed_sec: Total wall-clock time for the point.
    :param null_sampling: Resampling mode used for this point.
    :returns: Summary dictionary.
    """
    first = rows[0]
    return {
        "lambda": float(first["lambda"]),
        "M": int(first["M"]),
        "elapsed_sec": float(elapsed_sec),
        "time_per_trial_sec": float(elapsed_sec / len(rows)),
        "null_sampling": null_sampling,
    }


def _sample_std(values):
    """Return sample standard deviation, with zero for singleton groups.

    :param values: Numeric values.
    :returns: Sample standard deviation.
    """
    if values.shape[0] < 2:
        return 0.0
    return np.std(values, ddof=1)
