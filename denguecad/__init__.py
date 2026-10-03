"""
DengueCAD — computer-aided diagnosis helpers based on IEEE TITB 2012.

Uses the NMI non-parametric imputation library, GA wrapper feature selection,
and NMPrediction (Algorithm 1) with ADT / C4.5 / SVM / LOR comparison.
"""

from denguecad.feature_selection import (
    GAWrapperConfig,
    WrapperGAResult,
    select_influential_features,
)
from denguecad.nm_prediction import NMPrediction, NMPredictionConfig, NMPredictionResult

__all__ = [
    "GAWrapperConfig",
    "WrapperGAResult",
    "select_influential_features",
    "NMPrediction",
    "NMPredictionConfig",
    "NMPredictionResult",
]

__version__ = "0.1.4"
