"""Permutation-calibrated two-sample tests based on the NPLM statistic."""

from collections import namedtuple

import numpy as np
import torch
from scipy.stats import norm

from nplm import LogFalkonNPLM


#########################################################################################################
# Constants and result containers

_MAX_SEED = 2**32 - 1


NPLMPermutationResult = namedtuple(
    "NPLMPermutationResult",
    [
        "p_value",
        "z_score",
        "t_obs",
        "null_statistics",
        "n_extreme",
        "n_permutations",
        "seed",
        "observed_seed",
        "permutation_seeds",
        "resample_nystrom",
    ],
)
NPLMPermutationResult.__doc__ = "Result of a right-tail NPLM permutation test."


#########################################################################################################
# Public test API

def nplm_permutation_test(
    x_ref,
    x_data,
    model_config,
    *,
    n_permutations=100,
    seed=0,
    resample_nystrom=True,
    return_null=False,
    return_observed=True,
    dtype=np.float64,
):
    """Run a right-tail permutation test using the NPLM statistic.

    :param x_ref: Reference sample with shape ``(n_ref, n_features)``.
    :param x_data: Data sample with shape ``(n_data, n_features)``.
    :param model_config: Configuration passed to ``LogFalkonNPLM``; ``NR`` and ``sigma`` are required.
    :param n_permutations: Number of permuted splits used to build the empirical null.
    :param seed: Local RNG seed controlling permutations and per-fit model seeds.
    :param resample_nystrom: If true, draw a fresh model seed for each fit.
    :param return_null: If true, return the empirical null excluding the observed split.
    :param return_observed: If true, return the statistic computed on the initial split.
    :param dtype: Floating dtype used when pooling the samples.
    :returns: ``NPLMPermutationResult`` with p-value, Z-score, seeds, and optional arrays.
    """
    n_permutations = _validate_n_permutations(n_permutations)
    base_config = _validate_model_config(model_config)

    x_ref_2d = _as_2d_sample(x_ref, name="x_ref", dtype=dtype)
    x_data_2d = _as_2d_sample(x_data, name="x_data", dtype=dtype)
    _validate_compatible_samples(x_ref_2d, x_data_2d)

    n_ref = x_ref_2d.shape[0]
    n_data = x_data_2d.shape[0]
    _validate_config_sizes(base_config, n_ref=n_ref, n_data=n_data)

    base_config["N_R"] = n_ref
    base_config["N_D"] = n_data

    x_pool, y_fixed = _build_pooled_sample(x_ref_2d, x_data_2d, dtype=dtype)

    rng = np.random.default_rng(seed)
    observed_seed, permutation_seeds = _make_model_seeds(
        rng=rng,
        n_permutations=n_permutations,
        resample_nystrom=resample_nystrom,
    )

    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        t_obs_value = _compute_nplm_statistic(
            x=x_pool,
            y=y_fixed,
            base_config=base_config,
            seed=observed_seed,
        )

        null_statistics = np.empty(n_permutations, dtype=np.float64)
        for idx, model_seed in enumerate(permutation_seeds):
            permutation = rng.permutation(x_pool.shape[0])
            x_perm = np.ascontiguousarray(x_pool[permutation], dtype=dtype)
            null_statistics[idx] = _compute_nplm_statistic(
                x=x_perm,
                y=y_fixed,
                base_config=base_config,
                seed=int(model_seed),
            )
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    n_extreme = int(np.sum(null_statistics >= t_obs_value))
    p_value = float((1.0 + n_extreme) / (n_permutations + 1.0))
    z_score = float(norm.isf(p_value))

    return NPLMPermutationResult(
        p_value=p_value,
        z_score=z_score,
        t_obs=float(t_obs_value) if return_observed else None,
        null_statistics=null_statistics.copy() if return_null else None,
        n_extreme=n_extreme,
        n_permutations=n_permutations,
        seed=None if seed is None else int(seed),
        observed_seed=int(observed_seed),
        permutation_seeds=permutation_seeds.copy(),
        resample_nystrom=bool(resample_nystrom),
    )


#########################################################################################################
# Validation helpers

def _validate_n_permutations(n_permutations):
    """Validate the number of permutations.

    :param n_permutations: Requested number of permutations.
    :returns: Positive integer number of permutations.
    """
    n_permutations = int(n_permutations)
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    return n_permutations


def _validate_model_config(model_config):
    """Validate the NPLM model configuration required by stat tests.

    :param model_config: Configuration dictionary.
    :returns: Shallow copy of the validated configuration.
    """
    if not isinstance(model_config, dict):
        raise ValueError("model_config must be a dictionary")

    if model_config.get("NR", None) is None:
        raise ValueError("model_config must specify required parameter 'NR'")
    nr = float(model_config["NR"])
    if not np.isfinite(nr) or nr <= 0:
        raise ValueError(f"model_config 'NR' must be positive finite; got {nr}")

    if model_config.get("sigma", None) is None:
        raise ValueError("model_config must specify required parameter 'sigma'")
    sigma = float(model_config["sigma"])
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"model_config 'sigma' must be positive finite; got {sigma}")

    return dict(model_config)


def _as_2d_sample(x, *, name, dtype):
    """Convert an input sample to a contiguous two-dimensional array.

    :param x: Input sample with shape ``(n_samples,)`` or ``(n_samples, n_features)``.
    :param name: Name used in validation errors.
    :param dtype: NumPy dtype for the returned sample.
    :returns: Contiguous sample with shape ``(n_samples, n_features)``.
    """
    x = np.asarray(x, dtype=dtype)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array")
    if x.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty")
    return np.ascontiguousarray(x)


def _validate_compatible_samples(x_ref, x_data):
    """Validate that reference and data samples share feature dimensionality.

    :param x_ref: Reference sample with shape ``(n_ref, n_features)``.
    :param x_data: Data sample with shape ``(n_data, n_features)``.
    :returns: ``None``.
    """
    if x_ref.shape[1] != x_data.shape[1]:
        raise ValueError(
            f"Feature mismatch: x_ref has d={x_ref.shape[1]}, "
            f"x_data has d={x_data.shape[1]}"
        )


def _validate_config_sizes(
    model_config,
    *,
    n_ref,
    n_data,
):
    """Validate optional ``N_R`` and ``N_D`` values against sample sizes.

    :param model_config: NPLM model configuration dictionary.
    :param n_ref: Number of reference rows.
    :param n_data: Number of data rows.
    :returns: ``None``.
    """
    if model_config.get("N_R", None) is not None and int(model_config["N_R"]) != n_ref:
        raise ValueError(
            f"model_config N_R={model_config['N_R']} but x_ref has {n_ref} rows"
        )
    if model_config.get("N_D", None) is not None and int(model_config["N_D"]) != n_data:
        raise ValueError(
            f"model_config N_D={model_config['N_D']} but x_data has {n_data} rows"
        )


#########################################################################################################
# Sampling and statistic helpers

def _build_pooled_sample(
    x_ref,
    x_data,
    *,
    dtype,
):
    """Build a pooled sample and binary labels for NPLM fitting.

    :param x_ref: Reference sample with shape ``(n_ref, n_features)``.
    :param x_data: Data sample with shape ``(n_data, n_features)``.
    :param dtype: NumPy dtype for the pooled sample.
    :returns: Pair ``(x_pool, y)`` with shapes ``(n_ref + n_data, n_features)`` and ``(n_ref + n_data,)``.
    """
    n_ref = x_ref.shape[0]
    n_data = x_data.shape[0]
    n_total = n_ref + n_data

    x_pool = np.empty((n_total, x_ref.shape[1]), dtype=dtype)
    x_pool[:n_ref] = x_ref
    x_pool[n_ref:] = x_data

    y = np.empty(n_total, dtype=np.float64)
    y[:n_ref] = 0.0
    y[n_ref:] = 1.0

    return np.ascontiguousarray(x_pool), y


def _make_model_seeds(
    *,
    rng,
    n_permutations,
    resample_nystrom,
):
    """Draw model seeds for the observed and permuted NPLM fits.

    :param rng: NumPy random generator.
    :param n_permutations: Number of permuted fits.
    :param resample_nystrom: If true, draw one seed per fit.
    :returns: Pair ``(observed_seed, permutation_seeds)``.
    """
    if resample_nystrom:
        seeds = rng.integers(0, _MAX_SEED, size=n_permutations + 1, dtype=np.uint32)
        return int(seeds[0]), seeds[1:].astype(np.int64, copy=False)

    fixed_seed = int(rng.integers(0, _MAX_SEED, dtype=np.uint32))
    return fixed_seed, np.full(n_permutations, fixed_seed, dtype=np.int64)


def _compute_nplm_statistic(
    *,
    x,
    y,
    base_config,
    seed,
):
    """Compute one NPLM statistic with a fixed model seed.

    :param x: Pooled sample with shape ``(n_samples, n_features)``.
    :param y: Binary labels with shape ``(n_samples,)``.
    :param base_config: Base NPLM configuration dictionary.
    :param seed: Model seed used for this fit.
    :returns: Scalar NPLM statistic.
    """
    config = dict(base_config)
    config["seed"] = int(seed)

    model = LogFalkonNPLM(config)
    return float(model.compute_statistic(x, y, return_details=False))
