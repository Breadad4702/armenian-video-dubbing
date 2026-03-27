#!/usr/bin/env python3
"""
Regression Testing & Iteration Loop for Armenian Video Dubbing — Phase 4

Compares current evaluation results against stored baselines to:
  1. Detect metric regressions (degradations from baseline)
  2. Detect improvements (verify that fixes actually helped)
  3. Flag weak spots for iteration (worst-performing categories)
  4. Track progress over time (history of evaluation runs)

Usage:
    # Run regression check after an evaluation
    python -m scripts.evaluation.regression.regression_test \
        --current outputs/evaluation/metrics_2026-03-24.json \
        --baseline outputs/evaluation/baseline_metrics.json \
        --output outputs/evaluation/regression_report.json

    # Update baseline after accepting a new model
    python -m scripts.evaluation.regression.regression_test \
        --current outputs/evaluation/metrics_2026-03-24.json \
        --set-baseline
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


# ============================================================================
# Configuration: metric thresholds and directions
# ============================================================================

# (direction, abs_threshold, rel_threshold)
# direction: "higher_better" or "lower_better"
# abs_threshold: absolute change that counts as regression
# rel_threshold: relative change (fraction) that counts as regression
METRIC_DEFINITIONS = {
    "wer": {"direction": "lower_better", "abs_threshold": 0.005, "rel_threshold": 0.05, "target": 0.08},
    "cer": {"direction": "lower_better", "abs_threshold": 0.005, "rel_threshold": 0.05},
    "mos_mean": {"direction": "higher_better", "abs_threshold": 0.1, "rel_threshold": 0.02, "target": 4.6},
    "mean_similarity": {"direction": "higher_better", "abs_threshold": 0.02, "rel_threshold": 0.02, "target": 0.85},
    "lse_c": {"direction": "lower_better", "abs_threshold": 0.1, "rel_threshold": 0.05, "target": 1.8},
    "lse_d": {"direction": "lower_better", "abs_threshold": 0.1, "rel_threshold": 0.05, "target": 1.8},
    "comet_score": {"direction": "higher_better", "abs_threshold": 0.02, "rel_threshold": 0.02, "target": 0.85},
    "mean_bertscore_f1": {"direction": "higher_better", "abs_threshold": 0.02, "rel_threshold": 0.02},
    "mean_semantic_sim": {"direction": "higher_better", "abs_threshold": 0.02, "rel_threshold": 0.02},
    "total_sec": {"direction": "lower_better", "abs_threshold": 30, "rel_threshold": 0.10},
    "rtf": {"direction": "lower_better", "abs_threshold": 0.2, "rel_threshold": 0.10},
}


# ============================================================================
# Regression Detector
# ============================================================================

class RegressionDetector:
    """Compare current metrics against a baseline and detect regressions."""

    def __init__(self, metric_defs: Optional[Dict] = None):
        self.metric_defs = metric_defs or METRIC_DEFINITIONS

    def compare(
        self,
        current: Dict,
        baseline: Dict,
    ) -> Dict:
        """Compare current evaluation results against baseline.

        Both inputs are flat dicts or nested dicts of metric values.
        Nested dicts are flattened for comparison.

        Args:
            current: Current evaluation metrics.
            baseline: Baseline evaluation metrics.

        Returns:
            Dict with regressions, improvements, unchanged, and summary.
        """
        flat_current = self._flatten(current)
        flat_baseline = self._flatten(baseline)

        regressions = []
        improvements = []
        unchanged = []

        for key in sorted(set(flat_current.keys()) & set(flat_baseline.keys())):
            cur_val = flat_current[key]
            base_val = flat_baseline[key]

            if not isinstance(cur_val, (int, float)) or not isinstance(base_val, (int, float)):
                continue

            # Look up metric definition (strip nested prefixes for matching)
            metric_name = key.split(".")[-1]
            defn = self.metric_defs.get(metric_name)

            if defn is None:
                # Unknown metric — skip regression check
                continue

            abs_change = cur_val - base_val
            rel_change = abs_change / abs(base_val) if base_val != 0 else 0

            direction = defn["direction"]
            abs_thresh = defn["abs_threshold"]
            rel_thresh = defn["rel_threshold"]

            entry = {
                "metric": key,
                "current": round(cur_val, 6),
                "baseline": round(base_val, 6),
                "abs_change": round(abs_change, 6),
                "rel_change": round(rel_change, 4),
                "direction": direction,
            }

            # Determine if this is a regression, improvement, or unchanged
            if direction == "higher_better":
                if abs_change < -abs_thresh and abs(rel_change) > rel_thresh:
                    entry["status"] = "REGRESSION"
                    regressions.append(entry)
                elif abs_change > abs_thresh:
                    entry["status"] = "IMPROVED"
                    improvements.append(entry)
                else:
                    entry["status"] = "unchanged"
                    unchanged.append(entry)
            else:  # lower_better
                if abs_change > abs_thresh and abs(rel_change) > rel_thresh:
                    entry["status"] = "REGRESSION"
                    regressions.append(entry)
                elif abs_change < -abs_thresh:
                    entry["status"] = "IMPROVED"
                    improvements.append(entry)
                else:
                    entry["status"] = "unchanged"
                    unchanged.append(entry)

        has_regressions = len(regressions) > 0

        return {
            "has_regressions": has_regressions,
            "regressions": regressions,
            "improvements": improvements,
            "unchanged": unchanged,
            "n_regressions": len(regressions),
            "n_improvements": len(improvements),
            "n_unchanged": len(unchanged),
            "verdict": "FAIL — regressions detected" if has_regressions else "PASS — no regressions",
        }

    def _flatten(self, d: Dict, prefix: str = "") -> Dict:
        """Flatten nested dict into dot-separated keys."""
        flat = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(self._flatten(v, key))
            elif isinstance(v, (int, float)):
                flat[key] = v
        return flat


# ============================================================================
# Weak Spot Analyser
# ============================================================================

class WeakSpotAnalyser:
    """Identify weak spots in the pipeline for targeted iteration."""

    def __init__(self, metric_defs: Optional[Dict] = None):
        self.metric_defs = metric_defs or METRIC_DEFINITIONS

    def analyse(self, current_metrics: Dict) -> Dict:
        """Identify metrics that fail to meet their targets.

        Args:
            current_metrics: Flat or nested dict of current metric values.

        Returns:
            Dict with weak spots, ranked by severity.
        """
        flat = self._flatten(current_metrics)
        weak_spots = []

        for key, value in flat.items():
            metric_name = key.split(".")[-1]
            defn = self.metric_defs.get(metric_name)
            if defn is None or "target" not in defn:
                continue

            target = defn["target"]
            direction = defn["direction"]

            if direction == "higher_better":
                gap = target - value
                meets_target = value >= target
            else:
                gap = value - target
                meets_target = value <= target

            if not meets_target:
                severity = abs(gap) / abs(target) if target != 0 else abs(gap)
                weak_spots.append({
                    "metric": key,
                    "current": round(value, 6),
                    "target": target,
                    "gap": round(gap, 6),
                    "severity": round(severity, 4),
                    "direction": direction,
                    "suggestion": self._suggest_fix(metric_name, gap),
                })

        # Sort by severity descending
        weak_spots.sort(key=lambda x: x["severity"], reverse=True)

        all_targets_met = len(weak_spots) == 0
        return {
            "all_targets_met": all_targets_met,
            "weak_spots": weak_spots,
            "n_weak_spots": len(weak_spots),
            "priority_fix": weak_spots[0]["metric"] if weak_spots else None,
        }

    def _suggest_fix(self, metric_name: str, gap: float) -> str:
        """Suggest remediation for a weak metric."""
        suggestions = {
            "wer": "Increase ASR training data, adjust LoRA rank, or add language model rescoring",
            "mos_mean": "Improve TTS training data quality, increase Fish-Speech LoRA epochs, or tune prosody",
            "mean_similarity": "Increase voice cloning reference duration, or fine-tune speaker encoder",
            "lse_c": "Improve MuseTalk face alignment, adjust bbox_shift, or increase lip-sync training data",
            "lse_d": "Reduce audio-video temporal offset, check segment alignment in pipeline",
            "comet_score": "Fine-tune SeamlessM4T on Armenian parallel data, or increase beam size",
        }
        return suggestions.get(metric_name, "Review and improve this component")

    def _flatten(self, d: Dict, prefix: str = "") -> Dict:
        flat = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(self._flatten(v, key))
            elif isinstance(v, (int, float)):
                flat[key] = v
        return flat


# ============================================================================
# History Tracker
# ============================================================================

class EvalHistoryTracker:
    """Track evaluation results over time for progress monitoring."""

    def __init__(self, history_path: str = "outputs/evaluation/eval_history.jsonl"):
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, metrics: Dict, label: str = "") -> None:
        """Append an evaluation run to history."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
            "label": label,
            "metrics": metrics,
        }
        with open(self.history_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        logger.info("Appended to eval history: {}", self.history_path)

    def load_history(self) -> List[Dict]:
        """Load all historical evaluation runs."""
        if not self.history_path.exists():
            return []

        entries = []
        with open(self.history_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def get_trend(self, metric_key: str) -> List[Tuple[str, float]]:
        """Get a metric's values over time.

        Args:
            metric_key: Dot-separated path to metric (e.g., "asr.wer").

        Returns:
            List of (timestamp, value) tuples.
        """
        history = self.load_history()
        trend = []

        for entry in history:
            val = self._get_nested(entry.get("metrics", {}), metric_key)
            if val is not None:
                trend.append((entry["timestamp"], val))

        return trend

    def _get_nested(self, d: Dict, key: str):
        """Get value from nested dict using dot-separated key."""
        parts = key.split(".")
        current = d
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current if isinstance(current, (int, float)) else None


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Regression Testing & Iteration Loop")
    parser.add_argument("--current", type=str, required=True, help="Path to current metrics JSON")
    parser.add_argument("--baseline", type=str, help="Path to baseline metrics JSON")
    parser.add_argument("--output", type=str, default="outputs/evaluation/regression_report.json")
    parser.add_argument("--set-baseline", action="store_true", help="Save current as new baseline")
    parser.add_argument("--history", type=str, default="outputs/evaluation/eval_history.jsonl")
    parser.add_argument("--label", type=str, default="", help="Label for this evaluation run")

    args = parser.parse_args()

    # Load current metrics
    with open(args.current) as f:
        current = json.load(f)

    # Track in history
    tracker = EvalHistoryTracker(args.history)
    tracker.append(current, label=args.label)

    # Weak spot analysis (always runs)
    analyser = WeakSpotAnalyser()
    weak_spots = analyser.analyse(current)

    logger.info("\n=== Weak Spot Analysis ===")
    if weak_spots["all_targets_met"]:
        logger.info("All targets met!")
    else:
        for ws in weak_spots["weak_spots"]:
            logger.info("  {} = {:.4f} (target: {}, gap: {:.4f})", ws["metric"], ws["current"], ws["target"], ws["gap"])
            logger.info("    -> {}", ws["suggestion"])

    # Regression check (if baseline exists)
    regression_result = None
    baseline_path = Path(args.baseline) if args.baseline else Path("outputs/evaluation/baseline_metrics.json")

    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)

        detector = RegressionDetector()
        regression_result = detector.compare(current, baseline)

        logger.info("\n=== Regression Check ===")
        logger.info("Verdict: {}", regression_result["verdict"])

        if regression_result["regressions"]:
            for reg in regression_result["regressions"]:
                logger.warning("  REGRESSION: {} = {:.4f} (was {:.4f}, change: {:+.4f})",
                             reg["metric"], reg["current"], reg["baseline"], reg["abs_change"])

        if regression_result["improvements"]:
            for imp in regression_result["improvements"]:
                logger.info("  IMPROVED: {} = {:.4f} (was {:.4f}, change: {:+.4f})",
                          imp["metric"], imp["current"], imp["baseline"], imp["abs_change"])
    else:
        logger.info("No baseline found — skipping regression check")

    # Set as baseline if requested
    if args.set_baseline:
        baseline_out = Path("outputs/evaluation/baseline_metrics.json")
        baseline_out.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_out, "w") as f:
            json.dump(current, f, indent=2, default=str)
        logger.info("Saved as new baseline: {}", baseline_out)

    # Save report
    report = {
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "weak_spots": weak_spots,
        "regression": regression_result,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("\nReport saved to: {}", output_path)

    # Exit with error if regressions detected
    if regression_result and regression_result["has_regressions"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
