"""Pool samplers returning ``(x_ref, x_data, metadata)`` for one NPLM toy.

Counts are Poisson; row indices are sampled without replacement within a toy.
Pools may be reused across toys. Preprocess pools before constructing a sampler.
"""

import numpy as np


def _pool(values, name):
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError(f"{name} must have shape (n_events, n_features)")
    return values


def _expected_count(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return value


def _reference(pool, n_ref):
    pool = _pool(pool, "reference_pool")
    if isinstance(n_ref, (bool, np.bool_)) or int(n_ref) != n_ref or n_ref < 1:
        raise ValueError("n_ref must be a positive integer")
    n_ref = int(n_ref)
    _capacity(pool, n_ref, "reference_pool")
    return pool, n_ref


def _compatible(reference, other):
    if reference.shape[1] != other.shape[1]:
        raise ValueError("Pools must have the same number of feature columns")


def _capacity(pool, count, name):
    if count > len(pool):
        raise ValueError(
            f"Insufficient {name}: requested {count} rows without replacement, "
            f"but only {len(pool)} are available"
        )


def _draw(pool, count, rng, name):
    _capacity(pool, count, name)
    return pool[rng.choice(len(pool), size=count, replace=False)]


def make_null_sampler(reference_pool, *, n_ref, expected_data):
    """Draw Poisson(expected_data), jointly sample reference/data, then split.

    ``n_ref`` is fixed. Capacity is checked for each realized Poisson count;
    an oversized draw raises rather than being truncated or redrawn.
    """
    pool, n_ref = _reference(reference_pool, n_ref)
    expected_data = _expected_count(expected_data, "expected_data")

    def sample(rng):
        n_data = int(rng.poisson(expected_data))
        joint = _draw(pool, n_ref + n_data, rng, "reference_pool")
        return joint[:n_ref], joint[n_ref:], {
            "sampling": "null", "n_data": n_data,
            "expected_data": expected_data,
        }

    return sample


def make_mixture_sampler(
    reference_pool, component_pool, *, n_ref, expected_background, expected_component
):
    """Combine independent Poisson background and added-component counts.

    Reference and background are disjoint draws from ``reference_pool``.
    ``component_pool`` must contain only the added component, with distinct
    event identities from the reference pool. A zero component count is valid.
    """
    pool, n_ref = _reference(reference_pool, n_ref)
    component = _pool(component_pool, "component_pool")
    _compatible(pool, component)
    expected_background = _expected_count(expected_background, "expected_background")
    expected_component = _expected_count(expected_component, "expected_component")

    def sample(rng):
        n_background = int(rng.poisson(expected_background))
        n_component = int(rng.poisson(expected_component))
        joint = _draw(pool, n_ref + n_background, rng, "reference_pool")
        added = _draw(component, n_component, rng, "component_pool")
        data = np.concatenate((joint[n_ref:], added), axis=0)
        return joint[:n_ref], data, {
            "sampling": "mixture", "n_data": len(data),
            "n_background": n_background, "n_component": n_component,
            "expected_background": expected_background,
            "expected_component": expected_component,
        }

    return sample


def make_alternative_sampler(reference_pool, alternative_pool, *, n_ref, expected_data):
    """Draw all pseudo-data from a separate, full-alternative event pool.

    ``expected_data`` is its expected total yield. This pool already represents
    the alternative, including any background; no component split is inferred.
    The two pools must represent separate event collections.
    """
    pool, n_ref = _reference(reference_pool, n_ref)
    alternative = _pool(alternative_pool, "alternative_pool")
    _compatible(pool, alternative)
    expected_data = _expected_count(expected_data, "expected_data")

    def sample(rng):
        n_data = int(rng.poisson(expected_data))
        reference = _draw(pool, n_ref, rng, "reference_pool")
        data = _draw(alternative, n_data, rng, "alternative_pool")
        return reference, data, {
            "sampling": "full_alternative", "n_data": n_data,
            "expected_data": expected_data,
        }

    return sample
