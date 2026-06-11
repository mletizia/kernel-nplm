"""Expose NPLM-based statistical test helpers."""

#########################################################################################################
# Public package API

from .permutation import NPLMPermutationResult, nplm_permutation_test
from .resampling import NPLMResamplingResult, nplm_resampling_test

__all__ = [
    "NPLMPermutationResult",
    "NPLMResamplingResult",
    "nplm_permutation_test",
    "nplm_resampling_test",
]
