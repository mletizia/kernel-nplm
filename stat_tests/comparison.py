"""Calibrate saved NPLM ensembles without sampling events or fitting NPLM."""

from collections import namedtuple
from collections.abc import Mapping

import numpy as np
from scipy.stats import chi2, norm

from ._utils import _ALTERNATIVE_QUANTILE_LEVELS, _empirical_pvalues


NPLMChi2Fit = namedtuple(
    "NPLMChi2Fit", "dof ks_statistic p_value accepted reason n_bootstrap threshold"
)
NPLMAlternativeComparison = namedtuple(
    "NPLMAlternativeComparison",
    "n_alternative quantile_levels t_quantiles p_values z_scores chi2_p_values chi2_z_scores",
)
NPLMComparisonResult = namedtuple(
    "NPLMComparisonResult",
    "alternatives n_null p_value_resolution z_score_resolution chi2_fit seed",
)


def compare_nplm_results(null, alternatives, *, fit_chi2=False, seed=0):
    """Calibrate named alternative ensembles against one reconstructed null.

    Returns statistic quantiles at 0.16, 0.50, 0.84 and their right-tail p/Z
    values. The interval describes the spread across toys, not uncertainty on
    the median. Empirical p = (1 + number of null statistics >= t)/(n_null + 1),
    Z = norm.isf(p); no tail clipping or extrapolation is used.

    If requested, fit chi-square degrees of freedom to the null mean once.
    Assess compatibility with 999 parametric-bootstrap KS replicates, refitting
    the mean in every replicate. Only accepted fits (p >= 0.05) supply chi-square
    p/Z values; ``chi2_fit`` retains diagnostics on rejection or invalidity.
    ``seed`` controls only this bootstrap and defaults to zero.

    Null reuse requires compatible model settings, preprocessing, expected null
    yield, reference sampling and randomness policy. The caller must establish
    these conditions; array provenance cannot verify a custom sampler's physics
    or transformations. This function does not mutate either ensemble.
    """
    null_statistics = _statistics(null, "null")
    if not isinstance(alternatives, Mapping):
        raise TypeError("alternatives must map names to NPLMResamplingEnsemble objects")
    alternative_statistics = {
        name: _statistics(ensemble, f"alternative {name!r}")
        for name, ensemble in alternatives.items()
    }
    fit = _fit_chi2(null_statistics, seed) if fit_chi2 else None
    results = {}
    for name, statistics in alternative_statistics.items():
        levels = _ALTERNATIVE_QUANTILE_LEVELS.copy()
        quantiles = np.quantile(statistics, levels)
        p_values = _empirical_pvalues(null_statistics, quantiles)
        chi2_p = chi2.sf(quantiles, df=fit.dof) if fit is not None and fit.accepted else None
        results[name] = NPLMAlternativeComparison(
            len(statistics), levels, quantiles, p_values, norm.isf(p_values),
            chi2_p, None if chi2_p is None else norm.isf(chi2_p),
        )
    resolution = 1.0 / (len(null_statistics) + 1.0)
    return NPLMComparisonResult(
        results, len(null_statistics), resolution, float(norm.isf(resolution)),
        fit, None if seed is None else int(seed),
    )


def _statistics(ensemble, name):
    statistics = np.asarray(ensemble.statistics, dtype=np.float64)
    if statistics.ndim != 1 or statistics.size == 0 or not np.all(np.isfinite(statistics)):
        raise ValueError(f"{name} statistics must be a nonempty, finite 1D array")
    return statistics


def _ks_statistic(values, dof):
    cdf = chi2.cdf(np.sort(values), df=dof)
    ranks = np.arange(1, len(values) + 1) / len(values)
    return float(max(np.max(ranks - cdf), np.max(cdf - (ranks - 1.0 / len(values)))))


def _fit_chi2(statistics, seed):
    n_bootstrap, threshold = 999, 0.05
    dof = float(np.mean(statistics))
    if len(statistics) < 2 or np.any(statistics < 0) or not np.isfinite(dof) or dof <= 0:
        return NPLMChi2Fit(
            dof, None, None, False,
            "Requires at least two nonnegative null statistics and a positive finite mean",
            0, threshold,
        )
    observed = _ks_statistic(statistics, dof)
    rng = np.random.default_rng(seed)
    n_extreme = 0
    for _ in range(n_bootstrap):
        simulated = rng.chisquare(dof, size=len(statistics))
        simulated_dof = float(np.mean(simulated))
        if not np.isfinite(simulated_dof) or simulated_dof <= 0:
            return NPLMChi2Fit(
                dof, observed, None, False, "Invalid bootstrap degrees of freedom",
                n_bootstrap, threshold,
            )
        n_extreme += _ks_statistic(simulated, simulated_dof) >= observed
    p_value = float((1 + n_extreme) / (n_bootstrap + 1))
    accepted = p_value >= threshold
    return NPLMChi2Fit(
        dof, observed, p_value, accepted,
        "Compatible at the 0.05 threshold" if accepted else "Rejected at the 0.05 threshold",
        n_bootstrap, threshold,
    )
