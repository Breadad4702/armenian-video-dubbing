#!/usr/bin/env python3
"""
Phase 2 Step 4: Translation Evaluation & Timing Alignment (SeamlessM4T v2)

Evaluates SeamlessM4T v2 Large translation quality for English→Armenian:
  1. Run translation on test set
  2. Compute COMET, BERTScore, semantic similarity
  3. Measure timing alignment (source/target duration ratio)
  4. Analyse failure categories
  5. Report results

Usage:
    python scripts/training/evaluate_translation.py \
        --test-data data/splits/test.jsonl \
        --output-dir outputs/translation_eval

    python scripts/training/evaluate_translation.py \
        --test-data data/splits/test.jsonl \
        --src-lang eng --tgt-lang hye --dialect eastern
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger
from src.utils.helpers import timer


class TranslationEvaluator:
    """Evaluate SeamlessM4T v2 translation quality and timing alignment."""

    def __init__(
        self,
        model_id: str = "facebook/seamless-m4t-v2-large",
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.device = device
        self.translator = None

    def load_model(self):
        """Load SeamlessM4T via the project's TranslationInference module."""
        if self.translator is not None:
            return

        from src.inference import TranslationInference

        self.translator = TranslationInference(model_id=self.model_id, device=self.device)
        self.translator.load()
        logger.info("Translation model loaded: {}", self.model_id)

    def evaluate_test_set(
        self,
        test_data: List[Dict],
        src_lang: str = "eng",
        tgt_lang: str = "hye",
    ) -> Dict:
        """Run translation on every sample and collect results.

        Args:
            test_data: List of dicts with at least "text" (source).
                       Optional: "reference_text" (Armenian ground truth).
            src_lang: Source language code (SeamlessM4T format).
            tgt_lang: Target language code.

        Returns:
            Dict with per-sample results and aggregate quality metrics.
        """
        self.load_model()

        per_sample = []
        src_lengths = []
        tgt_lengths = []
        timing_ratios = []

        for i, sample in enumerate(test_data):
            src_text = sample.get("text", sample.get("source_text", ""))
            if not src_text.strip():
                continue

            t0 = time.monotonic()
            result = self.translator.translate(src_text, src_lang, tgt_lang)
            elapsed = time.monotonic() - t0

            tgt_text = result.get("tgt_text", "")

            # Character-level length ratio (timing proxy)
            src_len = len(src_text)
            tgt_len = len(tgt_text)
            ratio = tgt_len / max(1, src_len)

            src_lengths.append(src_len)
            tgt_lengths.append(tgt_len)
            timing_ratios.append(ratio)

            entry = {
                "index": i,
                "src_text": src_text,
                "tgt_text": tgt_text,
                "src_len_chars": src_len,
                "tgt_len_chars": tgt_len,
                "length_ratio": round(ratio, 3),
                "inference_sec": round(elapsed, 4),
            }

            # Reference comparison if available
            ref = sample.get("reference_text", sample.get("reference", ""))
            if ref:
                entry["reference_text"] = ref

            per_sample.append(entry)

            if (i + 1) % 50 == 0:
                logger.info("  Translated {}/{} samples", i + 1, len(test_data))

        # Aggregate stats
        ratios = np.array(timing_ratios) if timing_ratios else np.array([1.0])
        aggregate = {
            "n_samples": len(per_sample),
            "mean_length_ratio": float(np.mean(ratios)),
            "std_length_ratio": float(np.std(ratios)),
            "median_length_ratio": float(np.median(ratios)),
            "ratio_within_25pct": float(np.mean(np.abs(ratios - 1.0) < 0.25)),
            "mean_src_len": float(np.mean(src_lengths)) if src_lengths else 0,
            "mean_tgt_len": float(np.mean(tgt_lengths)) if tgt_lengths else 0,
        }

        return {"per_sample": per_sample, "aggregate": aggregate}

    def evaluate_quality_metrics(
        self,
        per_sample: List[Dict],
    ) -> Dict:
        """Compute quality metrics (COMET, BERTScore, semantic sim) on results.

        Requires scripts/evaluation/metrics/translation_metrics.py.
        """
        try:
            from scripts.evaluation.metrics.translation_metrics import (
                TranslationQualityComputer,
            )
        except ImportError:
            logger.warning("TranslationQualityComputer not available; skipping quality metrics")
            return {"note": "quality metrics module not importable"}

        computer = TranslationQualityComputer(device=self.device)

        sources = [s["src_text"] for s in per_sample]
        targets = [s["tgt_text"] for s in per_sample]

        result = computer.batch_translation_evaluation(sources, targets)
        return result

    def analyse_timing_alignment(
        self,
        per_sample: List[Dict],
        max_acceptable_ratio: float = 1.25,
        min_acceptable_ratio: float = 0.80,
    ) -> Dict:
        """Analyse how well target text length matches source for duration matching.

        A ratio near 1.0 means the TTS output will be close in duration to the
        original segment, requiring minimal time-stretching. Ratios outside the
        acceptable window may cause noticeable quality loss when stretched.

        Args:
            per_sample: List of per-sample results from evaluate_test_set.
            max_acceptable_ratio: Upper bound of acceptable length ratio.
            min_acceptable_ratio: Lower bound.

        Returns:
            Dict with timing alignment analysis.
        """
        ratios = [s["length_ratio"] for s in per_sample]
        if not ratios:
            return {"n_samples": 0}

        ratios = np.array(ratios)
        in_range = np.logical_and(ratios >= min_acceptable_ratio, ratios <= max_acceptable_ratio)

        # Find outliers (segments that will need aggressive stretching)
        too_long = [(s["index"], s["length_ratio"], s["src_text"][:60])
                     for s in per_sample if s["length_ratio"] > max_acceptable_ratio]
        too_short = [(s["index"], s["length_ratio"], s["src_text"][:60])
                      for s in per_sample if s["length_ratio"] < min_acceptable_ratio]

        return {
            "n_samples": len(ratios),
            "in_range_rate": float(np.mean(in_range)),
            "mean_ratio": float(np.mean(ratios)),
            "median_ratio": float(np.median(ratios)),
            "std_ratio": float(np.std(ratios)),
            "too_long_count": len(too_long),
            "too_short_count": len(too_short),
            "worst_too_long": too_long[:5],
            "worst_too_short": too_short[:5],
            "acceptable_range": (min_acceptable_ratio, max_acceptable_ratio),
        }

    def detect_failures(self, per_sample: List[Dict]) -> Dict:
        """Identify translation failure categories.

        Categories:
          - empty_output: empty/whitespace target
          - copy_through: target == source (untranslated)
          - extreme_ratio: length ratio > 2x or < 0.3x
        """
        empty = []
        copy_through = []
        extreme = []

        for s in per_sample:
            tgt = s["tgt_text"].strip()
            src = s["src_text"].strip()
            ratio = s["length_ratio"]

            if not tgt:
                empty.append(s["index"])
            elif tgt.lower() == src.lower():
                copy_through.append(s["index"])
            elif ratio > 2.0 or ratio < 0.3:
                extreme.append(s["index"])

        total = len(per_sample) or 1
        return {
            "empty_output": {"count": len(empty), "rate": len(empty) / total, "indices": empty[:10]},
            "copy_through": {"count": len(copy_through), "rate": len(copy_through) / total, "indices": copy_through[:10]},
            "extreme_ratio": {"count": len(extreme), "rate": len(extreme) / total, "indices": extreme[:10]},
            "total_failures": len(empty) + len(copy_through) + len(extreme),
            "failure_rate": (len(empty) + len(copy_through) + len(extreme)) / total,
        }


def load_test_data(path: str) -> List[Dict]:
    """Load JSONL test data."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Evaluate SeamlessM4T v2 translation + timing alignment")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test JSONL")
    parser.add_argument("--output-dir", type=str, default="outputs/translation_eval")
    parser.add_argument("--src-lang", type=str, default="eng")
    parser.add_argument("--tgt-lang", type=str, default="hye")
    parser.add_argument("--dialect", type=str, default="eastern", choices=["eastern", "western"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    setup_logger()

    dialect_map = {"eastern": "hye", "western": "hyw"}
    tgt_lang = dialect_map.get(args.dialect, args.tgt_lang)

    # Load data
    logger.info("Loading test data from {}", args.test_data)
    test_data = load_test_data(args.test_data)
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    logger.info("Loaded {} samples", len(test_data))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = TranslationEvaluator(device=args.device)

    # Step 1: Translate all samples
    logger.info("Step 1/4: Running translations ({} -> {})...", args.src_lang, tgt_lang)
    with timer("Translation"):
        results = evaluator.evaluate_test_set(test_data, args.src_lang, tgt_lang)

    # Step 2: Quality metrics
    logger.info("Step 2/4: Computing quality metrics...")
    quality = evaluator.evaluate_quality_metrics(results["per_sample"])

    # Step 3: Timing alignment analysis
    logger.info("Step 3/4: Analyzing timing alignment...")
    timing = evaluator.analyse_timing_alignment(results["per_sample"])

    # Step 4: Failure detection
    logger.info("Step 4/4: Detecting failures...")
    failures = evaluator.detect_failures(results["per_sample"])

    # Combine & save
    full_report = {
        "config": {
            "model": evaluator.model_id,
            "src_lang": args.src_lang,
            "tgt_lang": tgt_lang,
            "dialect": args.dialect,
            "n_samples": len(test_data),
        },
        "aggregate": results["aggregate"],
        "quality_metrics": quality,
        "timing_alignment": timing,
        "failures": failures,
    }

    # Save
    report_path = output_dir / "translation_eval_report.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, default=str, ensure_ascii=False)

    per_sample_path = output_dir / "translation_per_sample.jsonl"
    with open(per_sample_path, "w") as f:
        for s in results["per_sample"]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Print summary
    agg = results["aggregate"]
    logger.info("\n" + "=" * 60)
    logger.info("TRANSLATION EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info("  Samples:           {}", agg["n_samples"])
    logger.info("  Mean length ratio: {:.3f}", agg["mean_length_ratio"])
    logger.info("  In-range rate:     {:.1%}", timing.get("in_range_rate", 0))
    logger.info("  Failure rate:      {:.1%}", failures["failure_rate"])
    if "mean_comet" in quality:
        logger.info("  COMET:             {:.4f}", quality["mean_comet"])
    if "mean_bertscore_f1" in quality:
        logger.info("  BERTScore F1:      {:.4f}", quality["mean_bertscore_f1"])
    logger.info("  Report saved to:   {}", report_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
