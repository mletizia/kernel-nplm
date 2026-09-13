"""Reference-resampled two-sample tests based on the NPLM statistic."""

import warnings
from collections import namedtuple
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy

import numpy as np
from scipy.stats import norm

from .permutation import (
    _MAX_SEED,
    _as_2d_sample,
    _build_pooled_sample,
    _validate_compatible_samples,
    _validate_config_sizes,
    _validate_model_config,
)

from ._utils import _ALTERNATIVE_QUANTILE_LEVELS, _empirical_pvalues


#########################################################################################################
# Constants and result containers

_NULL_SAMPLING_MODES = {"disjoint", "independent"}


NPLMResamplingEnsemble = namedtuple(
    "NPLMResamplingEnsemble",
    "statistics reference_counts data_counts metadata seed model_seeds "
    "model_config resample_nystrom dtype",
)
NPLMResamplingEnsemble.__doc__ = "Raw toy statistics and provenance, before calibration."


NPLMResamplingResult = namedtuple(
    "NPLMResamplingResult",
    [
        "p_value",
        "z_score",
        "t_obs",
        "null_statistics",
        "alternative_statistics",
        "alternative_quantile_levels",
        "alternative_t_quantiles",
        "alternative_p_values",
        "alternative_z_scores",
        "n_extreme",
        "n_null",
        "n_alternative",
        "n_ref",
        "n_data",
        "observed_data_count",
        "null_data_counts",
        "alternative_data_counts",
        "null_sampling",
        "single_data_mode",
        "seed",
        "observed_seed",
        "null_seeds",
        "alternative_seeds",
        "resample_nystrom",
        "poisson_fluctuate_n_data",
        "p_value_resolution",
        "reference_null_factor",
        "reference_alt_factor",
        "data_alt_factor",
    ],
)
NPLMResamplingResult.__doc__ = "Result of a reference-calibrated NPLM resampling test."


#########################################################################################################
# Public test API

def nplm_resampling_null(
    null_sampler, model_config, *, n_null=100, seed=0,
    resample_nystrom=True, dtype=np.float64,
):
    """Reconstruct a null using ``sampler(rng) -> (ref, data, metadata)``.

    The sampler controls counts and preprocessing. ``NR`` remains the expected
    null yield; ``N_R`` and ``N_D`` are set from the realized arrays at each fit.
    No p-values are calculated here. Sampling or fitting failures identify the
    toy (one-based) and model seed, and abort the ensemble without retries.
    """
    return _run_resampling_toys(
        null_sampler, model_config, n_toys=n_null, seed=seed,
        resample_nystrom=resample_nystrom, dtype=dtype, label="null",
    )


def nplm_resampling_alternative(
    alternative_sampler, model_config, *, n_alternative=100, seed=0,
    resample_nystrom=True, dtype=np.float64,
):
    """Reconstruct an alternative independently of the null; see the null API.

    Use ``compare_nplm_results`` afterwards with a compatible null ensemble.
    A fresh model seed is drawn per toy by default. With
    ``resample_nystrom=False`` one model seed is repeated across this ensemble;
    this does not fix the actual center coordinates when sampled rows change.
    """
    return _run_resampling_toys(
        alternative_sampler, model_config, n_toys=n_alternative, seed=seed,
        resample_nystrom=resample_nystrom, dtype=dtype, label="alternative",
    )


def nplm_resampling_test(
    x_ref,
    x_data,
    model_config,
    *,
    n_ref,
    n_data,
    n_null=100,
    n_alternative=None,
    null_sampling="disjoint",
    poisson_fluctuate_n_data=False,
    seed=0,
    resample_nystrom=True,
    return_null=False,
    return_alternative=False,
    return_observed=True,
    reference_factor_warning_threshold=10.0,
    dtype=np.float64,
):
    """Run an NPLM test with a reference-resampled empirical null.

    :param x_ref: Reference pool with shape ``(n_ref_pool, n_features)``.
    :param x_data: Data pool with shape ``(n_data_pool, n_features)``.
    :param model_config: Configuration passed to ``LogFalkonNPLM``; ``NR`` and ``sigma`` are required.
    :param n_ref: Reference sample size used in each NPLM fit.
    :param n_data: Data sample size used in each NPLM fit.
    :param n_null: Number of null toys.
    :param n_alternative: Number of alternative toys; defaults to ``n_null`` outside single-data mode.
    :param null_sampling: Null mode, either ``"disjoint"`` or ``"independent"``.
    :param poisson_fluctuate_n_data: If true, draw each resampled toy data count from ``Poisson(n_data)``.
    :param seed: Local RNG seed controlling resampling and per-fit model seeds.
    :param resample_nystrom: If true, draw a fresh model seed for each fit.
    :param return_null: If true, return the empirical null statistics.
    :param return_alternative: If true, return the alternative statistics.
    :param return_observed: If true, return the statistic from the full-data split when available.
    :param reference_factor_warning_threshold: Warning threshold for sample-pool size factors.
    :param dtype: Floating dtype used when pooling sampled pairs.
    :returns: ``NPLMResamplingResult`` with observed and optional toy summaries.
    """
    n_ref = _validate_positive_int(n_ref, name="n_ref")
    n_data = _validate_positive_int(n_data, name="n_data")
    n_null = _validate_positive_int(n_null, name="n_null")
    null_sampling = _validate_null_sampling(null_sampling)
    reference_factor_warning_threshold = _validate_warning_threshold(
        reference_factor_warning_threshold
    )

    base_config = _validate_model_config(model_config)
    _validate_config_sizes(base_config, n_ref=n_ref, n_data=n_data)
    base_config["N_R"] = n_ref
    base_config["N_D"] = n_data

    x_ref_2d = _as_2d_sample(x_ref, name="x_ref", dtype=dtype)
    x_data_2d = _as_2d_sample(x_data, name="x_data", dtype=dtype)
    _validate_compatible_samples(x_ref_2d, x_data_2d)

    n_ref_pool = x_ref_2d.shape[0]
    n_data_pool = x_data_2d.shape[0]
    single_data_mode = n_data == n_data_pool

    n_alternative_requested = (
        None
        if n_alternative is None
        else _validate_nonnegative_int(n_alternative, name="n_alternative")
    )

    if (
        n_alternative_requested is not None
        and single_data_mode
        and n_alternative_requested > 0
    ):
        warnings.warn(
            "n_data == len(x_data), so the full data sample is tested once "
            "and n_alternative is ignored.",
            UserWarning,
        )

    if single_data_mode:
        n_alt_effective = 0
    elif n_alternative_requested is None:
        n_alt_effective = n_null
    else:
        n_alt_effective = n_alternative_requested

    _validate_sampling_feasibility(
        n_ref_pool=n_ref_pool,
        n_data_pool=n_data_pool,
        n_ref=n_ref,
        n_data=n_data,
        null_sampling=null_sampling,
        single_data_mode=single_data_mode,
        n_alternative=n_alt_effective,
    )

    reference_null_factor = n_ref_pool / float(n_ref + n_data)
    reference_alt_factor = n_ref_pool / float(n_ref)
    data_alt_factor = n_data_pool / float(n_data)
    _warn_on_low_sample_factor(
        factor=reference_null_factor,
        threshold=reference_factor_warning_threshold,
        label="x_ref / (n_ref + n_data) for the null",
    )
    if n_alt_effective > 0:
        _warn_on_low_sample_factor(
            factor=reference_alt_factor,
            threshold=reference_factor_warning_threshold,
            label="x_ref / n_ref for the alternative",
        )
        _warn_on_low_sample_factor(
            factor=data_alt_factor,
            threshold=reference_factor_warning_threshold,
            label="x_data / n_data for the alternative",
        )

    rng = np.random.default_rng(seed)
    n_observed = 1 if single_data_mode else 0
    fit_seeds = _draw_fit_seeds(
        rng=rng,
        n_fits=n_observed + n_null + n_alt_effective,
        resample_nystrom=resample_nystrom,
    )

    seed_pos = 0
    observed_seed = int(fit_seeds[seed_pos]) if single_data_mode else None
    seed_pos += n_observed
    null_seeds = fit_seeds[seed_pos : seed_pos + n_null]
    seed_pos += n_null
    alternative_seeds = fit_seeds[seed_pos : seed_pos + n_alt_effective]

    def sample_null(rng):
        count = _draw_resampled_data_count(
            n_data=n_data, rng=rng,
            poisson_fluctuate_n_data=poisson_fluctuate_n_data,
        )
        _validate_resampled_data_count(
            n_ref_pool=n_ref_pool, n_data_pool=n_data_pool, n_ref=n_ref,
            n_data=count, null_sampling=null_sampling, sampling_context="null",
        )
        reference, data = _sample_null_pair(
            x_ref_2d=x_ref_2d, n_ref=n_ref, n_data=count,
            rng=rng, null_sampling=null_sampling,
        )
        return reference, data, {}

    def sample_alternative(rng):
        count = _draw_resampled_data_count(
            n_data=n_data, rng=rng,
            poisson_fluctuate_n_data=poisson_fluctuate_n_data,
        )
        _validate_resampled_data_count(
            n_ref_pool=n_ref_pool, n_data_pool=n_data_pool, n_ref=n_ref,
            n_data=count, null_sampling=null_sampling, sampling_context="alternative",
        )
        ref_idx = rng.choice(n_ref_pool, size=n_ref, replace=False)
        data_idx = rng.choice(n_data_pool, size=count, replace=False)
        return x_ref_2d[ref_idx], x_data_2d[data_idx], {}

    def sample_observed(rng):
        ref_idx = rng.choice(n_ref_pool, size=n_ref, replace=False)
        return x_ref_2d[ref_idx], x_data_2d, {}

    # Reuse the original RNG and preallocated seeds across the three stages.
    toy_options = dict(seed=seed, resample_nystrom=resample_nystrom, dtype=dtype, rng=rng)
    null = _run_resampling_toys(
        sample_null, base_config, n_toys=n_null, model_seeds=null_seeds,
        label="null", **toy_options,
    )
    null_statistics = null.statistics
    null_data_counts = null.data_counts
    t_obs_value = None
    observed_data_count = None
    if single_data_mode:
        observed = _run_resampling_toys(
            sample_observed, base_config, n_toys=1, model_seeds=[observed_seed],
            label="observed", **toy_options,
        )
        t_obs_value = float(observed.statistics[0])
        observed_data_count = int(observed.data_counts[0])

    alternative_statistics = None
    alternative_data_counts = np.empty(0, dtype=np.int64)
    if n_alt_effective > 0:
        alternative = _run_resampling_toys(
            sample_alternative, base_config, n_toys=n_alt_effective,
            model_seeds=alternative_seeds, label="alternative", **toy_options,
        )
        alternative_statistics = alternative.statistics
        alternative_data_counts = alternative.data_counts

    p_value = None
    z_score = None
    n_extreme = None
    if t_obs_value is not None:
        n_extreme = int(np.sum(null_statistics >= t_obs_value))
        p_value = _empirical_pvalues(null_statistics, t_obs_value)
        z_score = float(norm.isf(p_value))

    alt_t_quantiles = None
    alt_p_values = None
    alt_z_scores = None
    alt_quantile_levels = None
    if alternative_statistics is not None:
        alt_quantile_levels = _ALTERNATIVE_QUANTILE_LEVELS.copy()
        alt_t_quantiles = np.quantile(alternative_statistics, alt_quantile_levels)
        alt_p_values = _empirical_pvalues(null_statistics, alt_t_quantiles)
        alt_z_scores = norm.isf(alt_p_values)

    return NPLMResamplingResult(
        p_value=p_value,
        z_score=z_score,
        t_obs=float(t_obs_value) if t_obs_value is not None and return_observed else None,
        null_statistics=null_statistics.copy() if return_null else None,
        alternative_statistics=(
            alternative_statistics.copy()
            if return_alternative and alternative_statistics is not None
            else None
        ),
        alternative_quantile_levels=alt_quantile_levels,
        alternative_t_quantiles=alt_t_quantiles,
        alternative_p_values=alt_p_values,
        alternative_z_scores=alt_z_scores,
        n_extreme=n_extreme,
        n_null=n_null,
        n_alternative=n_alt_effective,
        n_ref=n_ref,
        n_data=n_data,
        observed_data_count=observed_data_count,
        null_data_counts=null_data_counts.copy(),
        alternative_data_counts=alternative_data_counts.copy(),
        null_sampling=null_sampling,
        single_data_mode=single_data_mode,
        seed=None if seed is None else int(seed),
        observed_seed=None if observed_seed is None else int(observed_seed),
        null_seeds=null_seeds.copy(),
        alternative_seeds=alternative_seeds.copy(),
        resample_nystrom=bool(resample_nystrom),
        poisson_fluctuate_n_data=bool(poisson_fluctuate_n_data),
        p_value_resolution=float(1.0 / (n_null + 1.0)),
        reference_null_factor=float(reference_null_factor),
        reference_alt_factor=float(reference_alt_factor),
        data_alt_factor=float(data_alt_factor),
    )


#########################################################################################################
# Shared toy loop

def _run_resampling_toys(
    sampler, model_config, *, n_toys, seed, resample_nystrom, dtype, label,
    rng=None, model_seeds=None,
):
    """Sample, fit and collect toys; NR stays fixed and N_R/N_D follow each draw.

    The combined wrapper supplies its existing RNG and preallocated model seeds
    to preserve its sampling sequence. Separate reconstruction calls create them
    here. Failed toys abort with their number and model seed; no retries occur.
    """
    n_toys = _validate_positive_int(n_toys, name="n_toys")
    if not callable(sampler):
        raise TypeError("sampler must be callable: sampler(rng) -> (ref, data, metadata)")
    config = deepcopy(_validate_model_config(model_config))
    dtype = np.dtype(dtype)
    if rng is None:
        if dtype.kind != "f":
            raise ValueError("dtype must be a floating dtype")
        rng = np.random.default_rng(seed)
    if model_seeds is None:
        model_seeds = _draw_fit_seeds(
            rng=rng, n_fits=n_toys, resample_nystrom=resample_nystrom,
        )
    model_seeds = np.asarray(model_seeds, dtype=np.int64)

    statistics = np.empty(n_toys, dtype=np.float64)
    reference_counts = np.empty(n_toys, dtype=np.int64)
    data_counts = np.empty(n_toys, dtype=np.int64)
    metadata = []
    with _preserve_global_rng():
        from nplm import LogFalkonNPLM

        for idx, model_seed in enumerate(model_seeds):
            try:
                x_ref, x_data, toy_metadata = sampler(rng)
                if np.asarray(x_data).ndim > 0 and len(x_data) == 0:
                    raise ValueError("empty pseudo-dataset (realized data count is zero)")
                x_ref = _as_2d_sample(x_ref, name="x_ref", dtype=dtype)
                x_data = _as_2d_sample(x_data, name="x_data", dtype=dtype)
                _validate_compatible_samples(x_ref, x_data)
                if not isinstance(toy_metadata, Mapping):
                    raise TypeError("sampler metadata must be a mapping")
                metadata.append(deepcopy(dict(toy_metadata)))

                x, y = _build_pooled_sample(x_ref, x_data, dtype=dtype)
                fit_config = deepcopy(config)
                fit_config.update(N_R=len(x_ref), N_D=len(x_data), seed=int(model_seed))
                model = LogFalkonNPLM(fit_config)
                statistics[idx] = float(model.compute_statistic(x, y, return_details=False))
                if not np.isfinite(statistics[idx]):
                    raise ValueError(f"nonfinite NPLM statistic: {statistics[idx]}")
                reference_counts[idx] = len(x_ref)
                data_counts[idx] = len(x_data)
            except Exception as exc:
                raise RuntimeError(
                    f"{label} toy {idx + 1} failed (model seed={model_seed}): {exc}"
                ) from exc
    return NPLMResamplingEnsemble(
        statistics, reference_counts, data_counts, metadata,
        None if seed is None else int(seed), model_seeds.copy(), config,
        bool(resample_nystrom), dtype.name,
    )


@contextmanager
def _preserve_global_rng():
    import torch

    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


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


def _validate_nonnegative_int(value, *, name):
    """Validate a non-negative integer input.

    :param value: Input value.
    :param name: Name used in validation errors.
    :returns: Validated non-negative integer.
    """
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _validate_null_sampling(null_sampling):
    """Validate the null resampling mode.

    :param null_sampling: Requested null sampling mode.
    :returns: Validated mode string.
    """
    if null_sampling not in _NULL_SAMPLING_MODES:
        allowed = ", ".join(sorted(_NULL_SAMPLING_MODES))
        raise ValueError(f"null_sampling must be one of {{{allowed}}}")
    return null_sampling


def _validate_warning_threshold(threshold):
    """Validate the sample-factor warning threshold.

    :param threshold: Warning threshold.
    :returns: Positive finite threshold.
    """
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("reference_factor_warning_threshold must be positive finite")
    return threshold


def _validate_sampling_feasibility(
    *,
    n_ref_pool,
    n_data_pool,
    n_ref,
    n_data,
    null_sampling,
    single_data_mode,
    n_alternative,
):
    """Validate that requested samples can be drawn from the available pools.

    :param n_ref_pool: Number of rows in the reference pool.
    :param n_data_pool: Number of rows in the data pool.
    :param n_ref: Reference rows per fit.
    :param n_data: Data rows per fit.
    :param null_sampling: Null resampling mode.
    :param single_data_mode: Whether the full data pool is tested once.
    :param n_alternative: Number of alternative toys.
    :returns: ``None``.
    """
    if null_sampling == "disjoint" and n_ref + n_data > n_ref_pool:
        raise ValueError(
            "Disjoint null sampling requires len(x_ref) >= n_ref + n_data; "
            f"got len(x_ref)={n_ref_pool}, n_ref={n_ref}, n_data={n_data}"
        )

    if n_ref > n_ref_pool:
        raise ValueError(
            f"Alternative/observed reference sampling requires len(x_ref) >= n_ref; "
            f"got len(x_ref)={n_ref_pool}, n_ref={n_ref}"
        )

    if n_data > n_data_pool:
        raise ValueError(
            f"Alternative/observed data sampling requires len(x_data) >= n_data; "
            f"got len(x_data)={n_data_pool}, n_data={n_data}"
        )

    if n_alternative > 0 and single_data_mode:
        raise RuntimeError("Internal error: alternative toys in single-data mode")


def _validate_resampled_data_count(
    *,
    n_ref_pool,
    n_data_pool,
    n_ref,
    n_data,
    null_sampling,
    sampling_context,
):
    """Validate one realized resampled data count.

    :param n_ref_pool: Number of rows in the reference pool.
    :param n_data_pool: Number of rows in the data pool.
    :param n_ref: Reference rows per fit.
    :param n_data: Realized pseudo-data rows for this fit.
    :param null_sampling: Null resampling mode.
    :param sampling_context: Human-readable context for errors.
    :returns: ``None``.
    """
    if n_data == 0:
        raise ValueError("empty pseudo-dataset (realized data count is zero)")

    if sampling_context == "null":
        if null_sampling == "disjoint" and n_ref + n_data > n_ref_pool:
            raise ValueError(
                "Disjoint null sampling requires len(x_ref) >= n_ref + realized n_data; "
                f"got len(x_ref)={n_ref_pool}, n_ref={n_ref}, realized n_data={n_data}"
            )
        return

    if n_data > n_data_pool:
        raise ValueError(
            f"Alternative sampling requires len(x_data) >= realized n_data; "
            f"got len(x_data)={n_data_pool}, realized n_data={n_data}"
        )


def _warn_on_low_sample_factor(
    *,
    factor,
    threshold,
    label,
):
    """Warn when a sample pool is small relative to requested toy sizes.

    :param factor: Available-pool factor.
    :param threshold: Recommended minimum factor.
    :param label: Human-readable factor label.
    :returns: ``None``.
    """
    if factor < threshold:
        warnings.warn(
            f"Sample pool factor for {label} is {factor:.3g}x. "
            f"A factor of at least {threshold:g}x is recommended for stable resampling.",
            UserWarning,
        )


#########################################################################################################
# Sampling and statistic helpers

def _draw_fit_seeds(
    *,
    rng,
    n_fits,
    resample_nystrom,
):
    """Draw model seeds for all NPLM fits in the resampling test.

    :param rng: NumPy random generator.
    :param n_fits: Number of model fits.
    :param resample_nystrom: If true, draw one seed per fit.
    :returns: Seed array with shape ``(n_fits,)``.
    """
    if n_fits == 0:
        return np.empty(0, dtype=np.int64)

    if resample_nystrom:
        return rng.integers(0, _MAX_SEED, size=n_fits, dtype=np.uint32).astype(
            np.int64,
            copy=False,
        )

    fixed_seed = int(rng.integers(0, _MAX_SEED, dtype=np.uint32))
    return np.full(n_fits, fixed_seed, dtype=np.int64)


def _draw_resampled_data_count(
    *,
    n_data,
    rng,
    poisson_fluctuate_n_data,
):
    """Draw the realized pseudo-data count for one resampled toy.

    :param n_data: Nominal expected pseudo-data count.
    :param rng: NumPy random generator.
    :param poisson_fluctuate_n_data: Whether to draw from ``Poisson(n_data)``.
    :returns: Nonnegative integer data count (zero is not clamped).
    """
    if poisson_fluctuate_n_data:
        return int(rng.poisson(n_data))
    return int(n_data)


def _sample_null_pair(
    *,
    x_ref_2d,
    n_ref,
    n_data,
    rng,
    null_sampling,
):
    """Sample one pseudo-reference and pseudo-data pair under the null.

    :param x_ref_2d: Reference pool with shape ``(n_ref_pool, n_features)``.
    :param n_ref: Number of pseudo-reference rows.
    :param n_data: Number of pseudo-data rows.
    :param rng: NumPy random generator.
    :param null_sampling: Null mode, either ``"disjoint"`` or ``"independent"``.
    :returns: Pair of sampled arrays.
    """
    if null_sampling == "disjoint":
        idx = rng.choice(x_ref_2d.shape[0], size=n_ref + n_data, replace=False)
        return x_ref_2d[idx[:n_ref]], x_ref_2d[idx[n_ref:]]

    idx_ref = rng.choice(x_ref_2d.shape[0], size=n_ref, replace=True)
    idx_data = rng.choice(x_ref_2d.shape[0], size=n_data, replace=True)
    return x_ref_2d[idx_ref], x_ref_2d[idx_data]
