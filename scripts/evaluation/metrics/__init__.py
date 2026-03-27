"""
Automatic Metrics computation modules for Phase 4 evaluation.

Includes:
- WER/CER metrics
- MOS proxy estimation
- Speaker similarity
- Lip-sync metrics
- Translation quality
- Performance benchmarking
"""

from .wer_metrics import WERComputer
from .mos_proxy_metrics import MOSProxyEstimator
from .speaker_similarity import SpeakerSimilarityComputer
from .lipsync_metrics import LipSyncMetricsComputer
from .translation_metrics import TranslationQualityComputer
from .performance_metrics import PerformanceBenchmark

__all__ = [
    "WERComputer",
    "MOSProxyEstimator",
    "SpeakerSimilarityComputer",
    "LipSyncMetricsComputer",
    "TranslationQualityComputer",
    "PerformanceBenchmark",
]
