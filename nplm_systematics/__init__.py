"""Experimental systematics tools for kernel NPLM.

This package is intentionally separate from :mod:`nplm` while the nuisance
modeling strategy is being developed.
"""

#########################################################################################################
# Public exports

from .falkon_ratio import FalkonLogRatioEstimator, FalkonRatioConfig
from .finite_difference import FiniteDifferenceMorpher, QuadraticFiniteDifferenceMorpher
from .morphing import LinearLogRMorphingCache, QuadraticLogRMorphingCache

__all__ = [
    "FalkonLogRatioEstimator",
    "FalkonRatioConfig",
    "FiniteDifferenceMorpher",
    "LinearLogRMorphingCache",
    "QuadraticLogRMorphingCache",
    "QuadraticFiniteDifferenceMorpher",
]
