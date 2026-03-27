#!/usr/bin/env python3
"""
Phase 4: Full Evaluation Suite Orchestrator

Complete evaluation pipeline:
1. Automatic metrics (WER, MOS, speaker similarity, LSE-C/D, translation, performance)
2. Baseline comparisons
3. Regression detection
4. Failure analysis
5. Report generation

Usage:
    python scripts/evaluation/evaluate_full.py \\
        --checkpoint models/ \\
        --test-set data/splits/ \\
        --mode full \\
        --output outputs/evaluation
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml
from loguru import logger

from scripts.evaluation.metrics import (
    WERComputer,
    MOSProxyEstimator,
    SpeakerSimilarityComputer,
    LipSyncMetricsComputer,
    TranslationQualityComputer,
    PerformanceBenchmark,
)


class FullEvaluationSuite:
    """Orchestrate complete evaluation pipeline."""

    def __init__(self, config_path: str = "scripts/evaluation/eval_config.yaml"):
        """
        Initialize evaluation suite.

        Args:
            config_path: Path to evaluation configuration file
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = self.config.get("device", {}).get("gpu", "cuda")
        self.output_dir = Path(self.config.get("evaluation", {}).get("output_dir", "outputs/evaluation"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {self.output_dir}")

        self.results = {}
        self.timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    def run_complete_evaluation(
        self,
        checkpoint_dir: str,
        test_set_dir: str,
    ) -> Dict:
        """
        Run complete evaluation pipeline.

        Args:
            checkpoint_dir: Path to model checkpoints
            test_set_dir: Path to test set directory

        Returns:
            Dictionary with all evaluation results
        """
        logger.info("=" * 60)
        logger.info("STARTING COMPLETE EVALUATION SUITE")
        logger.info("=" * 60)

        try:
            # 1. Load test sets
            logger.info("\n[Step 1/6] Loading test sets...")
            test_sets = self._load_testsets(test_set_dir)

            # 2. Run automatic metrics
            logger.info("\n[Step 2/6] Computing automatic metrics...")
            auto_metrics = self._run_automatic_metrics(checkpoint_dir, test_sets)

            # 3. Run baseline comparisons
            logger.info("\n[Step 3/6] Evaluating baselines...")
            baseline_metrics = self._run_baselines(test_sets)

            # 4. Check regressions
            logger.info("\n[Step 4/6] Checking for regressions...")
            regressions = self._check_regressions(auto_metrics)

            # 5. Analyze failures
            logger.info("\n[Step 5/6] Analyzing failures...")
            failures = self._analyze_failures(auto_metrics)

            # 6. Generate reports
            logger.info("\n[Step 6/6] Generating reports...")
            reports = self._generate_reports(auto_metrics, baseline_metrics, failures)

            # Combine all results
            final_results = {
                "timestamp": self.timestamp,
                "checkpoint_dir": checkpoint_dir,
                "test_set_dir": test_set_dir,
                "automatic_metrics": auto_metrics,
                "baseline_metrics": baseline_metrics,
                "regressions": regressions,
                "failures": failures,
                "reports": reports,
            }

            # Save complete results
            self._save_results(final_results)

            logger.info("\n" + "=" * 60)
            logger.info("EVALUATION COMPLETE ✅")
            logger.info("=" * 60)

            return final_results

        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return {"error": str(e)}

    def run_quick_evaluation(
        self,
        checkpoint_dir: str,
        test_set_dir: str,
    ) -> Dict:
        """
        Fast evaluation (metrics only, ~10 min).

        Args:
            checkpoint_dir: Path to model checkpoints
            test_set_dir: Path to test set directory

        Returns:
            Dictionary with evaluation results
        """
        logger.info("Running QUICK evaluation (metrics only)")

        try:
            test_sets = self._load_testsets(test_set_dir)
            auto_metrics = self._run_automatic_metrics(checkpoint_dir, test_sets)

            return {
                "timestamp": self.timestamp,
                "checkpoint_dir": checkpoint_dir,
                "automatic_metrics": auto_metrics,
            }

        except Exception as e:
            logger.error(f"Quick evaluation failed: {e}")
            return {"error": str(e)}

    def _load_testsets(self, test_set_dir: str) -> Dict:
        """Load test sets from directory."""
        logger.info(f"Loading test sets from: {test_set_dir}")

        test_sets = {}

        # Try to load predefined test sets
        test_dir = Path(test_set_dir)

        for test_type in ["asr", "tts", "lipsync", "translation"]:
            manifest_path = test_dir / f"test_{test_type}.jsonl"

            if manifest_path.exists():
                logger.info(f"  ✓ {test_type}: {manifest_path}")
                test_sets[test_type] = str(manifest_path)
            else:
                logger.warning(f"  ✗ {test_type}: not found")

        if not test_sets:
            logger.warning("No test sets found!")

        return test_sets

    def _run_automatic_metrics(
        self,
        checkpoint_dir: str,
        test_sets: Dict,
    ) -> Dict:
        """Run all automatic metric computations."""
        metrics = {}

        # ASR metrics
        if "asr" in test_sets:
            logger.info("\n  Computing ASR metrics (WER, CER)...")
            try:
                asr_computer = WERComputer(checkpoint_dir, device=self.device)
                wer_results = asr_computer.compute_wer_on_testset(test_sets["asr"])
                metrics["asr"] = wer_results
                logger.info(f"    WER: {wer_results.get('wer', 0):.4f}")
            except Exception as e:
                logger.error(f"    ASR metrics failed: {e}")
                metrics["asr"] = {"error": str(e)}

        # TTS metrics
        if "tts" in test_sets:
            logger.info("\n  Computing TTS metrics (MOS proxy)...")
            try:
                mos_estimator = MOSProxyEstimator(device=self.device)
                mos_results = mos_estimator.estimate_from_manifest(test_sets["tts"])
                metrics["tts"] = mos_results
                logger.info(f"    MOS estimate: {mos_results.get('mos_mean', 'N/A')}")
            except Exception as e:
                logger.error(f"    TTS metrics failed: {e}")
                metrics["tts"] = {"error": str(e)}
        else:
            metrics["tts"] = {"note": "No TTS test set found"}

        # Speaker similarity metrics
        logger.info("\n  Computing speaker similarity...")
        try:
            speaker_computer = SpeakerSimilarityComputer(device=self.device)
            if "tts" in test_sets:
                spk_results = speaker_computer.compute_from_manifest(test_sets["tts"])
                metrics["speaker_similarity"] = spk_results
                logger.info(f"    Mean similarity: {spk_results.get('mean_similarity', 'N/A')}")
            else:
                metrics["speaker_similarity"] = {"note": "Requires TTS test set with paired audio"}
        except Exception as e:
            logger.error(f"    Speaker similarity failed: {e}")
            metrics["speaker_similarity"] = {"error": str(e)}

        # Lip-sync metrics
        logger.info("\n  Computing lip-sync metrics (LSE-C/D)...")
        try:
            lipsync_computer = LipSyncMetricsComputer(device=self.device)
            if "lipsync" in test_sets:
                ls_results = lipsync_computer.compute_from_manifest(test_sets["lipsync"])
                metrics["lipsync"] = ls_results
                logger.info(f"    LSE-C: {ls_results.get('lse_c', 'N/A')}, LSE-D: {ls_results.get('lse_d', 'N/A')}")
            else:
                metrics["lipsync"] = {"note": "Requires lipsync test set with video pairs"}
        except Exception as e:
            logger.error(f"    Lip-sync metrics failed: {e}")
            metrics["lipsync"] = {"error": str(e)}

        # Translation metrics
        logger.info("\n  Computing translation metrics (COMET, METEOR, BERTScore)...")
        try:
            translation_computer = TranslationQualityComputer(device=self.device)
            if "translation" in test_sets:
                trans_results = translation_computer.compute_from_manifest(test_sets["translation"])
                metrics["translation"] = trans_results
                logger.info(f"    COMET: {trans_results.get('comet_score', 'N/A')}")
            else:
                metrics["translation"] = {"note": "Requires translation test set with references"}
        except Exception as e:
            logger.error(f"    Translation metrics failed: {e}")
            metrics["translation"] = {"error": str(e)}

        # Performance metrics
        logger.info("\n  Computing performance metrics...")
        try:
            benchmark = PerformanceBenchmark(device=self.device)
            perf_results = benchmark.measure_pipeline_latency()
            metrics["performance"] = perf_results
            logger.info(f"    Pipeline latency: {perf_results.get('total_sec', 'N/A')}s")
        except Exception as e:
            logger.error(f"    Performance metrics failed: {e}")
            metrics["performance"] = {"error": str(e)}

        return metrics

    def _run_baselines(self, test_sets: Dict) -> Dict:
        """Run baseline comparisons."""
        baselines = self.config.get("evaluation", {}).get("baselines", [])

        logger.info(f"Evaluating {len(baselines)} baselines: {baselines}")

        baseline_results = {}

        for baseline in baselines:
            logger.info(f"\n  {baseline}...")
            baseline_results[baseline] = {
                "status": "pending",
                "note": f"Baseline evaluation for {baseline} pending"
            }

        return baseline_results

    def _check_regressions(self, auto_metrics: Dict) -> Dict:
        """Check for metric regressions against stored baseline."""
        logger.info("Checking for regressions...")

        from scripts.evaluation.regression.regression_test import (
            RegressionDetector,
            EvalHistoryTracker,
        )

        # Track this run in history
        tracker = EvalHistoryTracker(
            str(self.output_dir / "eval_history.jsonl")
        )
        tracker.append(auto_metrics, label=self.timestamp)

        # Try to load baseline
        baseline_path = self.output_dir / "baseline_metrics.json"
        if not baseline_path.exists():
            logger.info("  No baseline found (first run) — saving current as baseline")
            with open(baseline_path, "w") as f:
                json.dump(auto_metrics, f, indent=2, default=str)
            return {"detected": False, "regressions": [], "note": "First run — saved as baseline"}

        with open(baseline_path) as f:
            baseline = json.load(f)

        detector = RegressionDetector()
        result = detector.compare(auto_metrics, baseline)

        if result["has_regressions"]:
            logger.warning("  REGRESSIONS DETECTED:")
            for reg in result["regressions"]:
                logger.warning("    {} = {:.4f} (was {:.4f})", reg["metric"], reg["current"], reg["baseline"])
        else:
            logger.info("  No regressions detected")

        if result["improvements"]:
            logger.info("  Improvements:")
            for imp in result["improvements"]:
                logger.info("    {} = {:.4f} (was {:.4f})", imp["metric"], imp["current"], imp["baseline"])

        return {
            "detected": result["has_regressions"],
            "regressions": result["regressions"],
            "improvements": result["improvements"],
            "verdict": result["verdict"],
        }

    def _analyze_failures(self, auto_metrics: Dict) -> Dict:
        """Analyze failure cases and identify weak spots."""
        logger.info("Analyzing failures...")

        from scripts.evaluation.regression.regression_test import WeakSpotAnalyser

        # Weak spot analysis against targets
        analyser = WeakSpotAnalyser()
        weak_spots = analyser.analyse(auto_metrics)

        failures = {
            "asr_failures": [],
            "tts_failures": [],
            "speaker_similarity_failures": [],
            "lipsync_failures": [],
            "translation_failures": [],
            "weak_spots": weak_spots.get("weak_spots", []),
            "all_targets_met": weak_spots.get("all_targets_met", False),
            "priority_fix": weak_spots.get("priority_fix"),
        }

        # Extract worst samples from metrics where available
        if "asr" in auto_metrics:
            worst = auto_metrics["asr"].get("worst_samples", [])
            failures["asr_failures"] = worst[:5]

        if not weak_spots["all_targets_met"]:
            logger.warning("  {} weak spots identified:", len(weak_spots["weak_spots"]))
            for ws in weak_spots["weak_spots"][:5]:
                logger.warning("    {} = {:.4f} (target: {}, suggestion: {})",
                             ws["metric"], ws["current"], ws["target"], ws["suggestion"])
        else:
            logger.info("  All quality targets met!")

        n_failures = sum(len(v) for k, v in failures.items() if isinstance(v, list))
        logger.info(f"  {n_failures} specific failures identified")

        return failures

    def _generate_reports(
        self,
        auto_metrics: Dict,
        baseline_metrics: Dict,
        failures: Dict,
    ) -> Dict:
        """Generate evaluation reports."""
        logger.info("Generating reports...")

        reports = {}

        # Summary report
        summary = self._generate_summary_report(auto_metrics)
        reports["summary"] = summary

        # Save summary
        summary_path = self.output_dir / f"evaluation_summary_{self.timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  Saved summary to: {summary_path}")

        # Detailed metrics report
        metrics_path = self.output_dir / f"metrics_{self.timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(auto_metrics, f, indent=2)
        logger.info(f"  Saved detailed metrics to: {metrics_path}")

        return reports

    def _generate_summary_report(self, auto_metrics: Dict) -> Dict:
        """Generate summary report of evaluation."""
        targets = self.config.get("evaluation", {}).get("targets", {})

        summary = {
            "timestamp": self.timestamp,
            "targets_met": {},
            "summary_status": "PENDING",
        }

        # Check ASR target
        if "asr" in auto_metrics:
            wer = auto_metrics["asr"].get("wer")
            target_wer = targets.get("wer", 0.08)

            if wer is not None:
                met = wer < target_wer
                summary["targets_met"]["wer"] = {
                    "value": wer,
                    "target": target_wer,
                    "met": met,
                }

        logger.info(f"  Targets met: {summary['targets_met']}")

        return summary

    def _save_results(self, results: Dict) -> None:
        """Save complete evaluation results to file."""
        results_path = self.output_dir / f"evaluation_results_{self.timestamp}.json"

        with open(results_path, "w") as f:
            # Convert numpy types for JSON serialization
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Saved complete results to: {results_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 4: Complete Evaluation Suite"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoints directory"
    )

    parser.add_argument(
        "--test-set",
        type=str,
        required=True,
        help="Path to test set directory"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick", "full"],
        default="quick",
        help="Evaluation mode (quick=10min, full=1hour)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation",
        help="Output directory for results"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="scripts/evaluation/eval_config.yaml",
        help="Path to evaluation config"
    )

    args = parser.parse_args()

    # Setup logging
    log_path = Path(args.output) / "evaluation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level="INFO")
    logger.add(log_path, level="DEBUG")

    try:
        suite = FullEvaluationSuite(args.config)

        if args.mode == "quick":
            results = suite.run_quick_evaluation(args.checkpoint, args.test_set)
        else:
            results = suite.run_complete_evaluation(args.checkpoint, args.test_set)

        if "error" in results:
            logger.error(f"Evaluation failed: {results['error']}")
            sys.exit(1)

        logger.info("\n✅ Evaluation successful!")
        print(f"\nResults saved to: {suite.output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
