"""
Regression testing and failure detection for Phase 4.

Provides:
  - RegressionDetector: compare metrics against baseline
  - WeakSpotAnalyser: identify targets not yet met
  - EvalHistoryTracker: track progress over evaluation runs
"""

from scripts.evaluation.regression.regression_test import (
    RegressionDetector,
    WeakSpotAnalyser,
    EvalHistoryTracker,
    METRIC_DEFINITIONS,
)

__all__ = [
    "RegressionDetector",
    "WeakSpotAnalyser",
    "EvalHistoryTracker",
    "METRIC_DEFINITIONS",
]
