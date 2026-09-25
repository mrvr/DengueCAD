"""
DengueCAD — computer-aided diagnosis helpers based on IEEE TITB 2012.

Uses the NMI non-parametric imputation library and a GA wrapper for
influential feature subset selection (Kohavi & John, 1997).
"""

from denguecad.feature_selection import (
    GAWrapperConfig,
    WrapperGAResult,
    select_influential_features,
)

__all__ = [
    "GAWrapperConfig",
    "WrapperGAResult",
    "select_influential_features",
]

__version__ = "0.1.0"
