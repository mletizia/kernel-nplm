"""Expose NPLM-based statistical test helpers."""

#########################################################################################################
# Public package API

from .permutation import NPLMPermutationResult, nplm_permutation_test
from .resampling import (
    NPLMResamplingResult, NPLMResamplingEnsemble, nplm_resampling_test,
    nplm_resampling_null, nplm_resampling_alternative,
)
from .comparison import (
    compare_nplm_results, NPLMComparisonResult, NPLMAlternativeComparison, NPLMChi2Fit,
)

__all__ = [
    "NPLMPermutationResult",
    "NPLMResamplingResult",
    "nplm_permutation_test",
    "nplm_resampling_test",
    "NPLMResamplingEnsemble",
    "nplm_resampling_null",
    "nplm_resampling_alternative",
    "compare_nplm_results",
    "NPLMComparisonResult",
    "NPLMAlternativeComparison",
    "NPLMChi2Fit",
]
