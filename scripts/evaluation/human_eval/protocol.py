"""
Human Evaluation Framework for Armenian Video Dubbing — Phase 4

Provides:
  1. Study design (MOS, CMOS, A/B preference)
  2. Evaluator questionnaire generation (JSONL format for LabelStudio)
  3. Statistical analysis (inter-rater agreement, significance tests)
  4. Report generation

Studies:
  - MOS Naturalness (1–5): "How natural does the dubbed speech sound?"
  - MOS Intelligibility (1–5): "How easy is it to understand the speech?"
  - Speaker Similarity (1–5): "How similar is the dubbed voice to the original?"
  - Lip-Sync Quality (1–5): "How well do the lip movements match the speech?"
  - Overall Quality (1–5): "Rate the overall quality of the dubbed video."
  - A/B Preference: Side-by-side comparison of system vs baseline

Usage:
    # Generate evaluation tasks
    python -m scripts.evaluation.human_eval.protocol --generate \
        --samples outputs/evaluation/samples.jsonl \
        --output outputs/evaluation/human_eval_tasks.json

    # Analyse collected ratings
    python -m scripts.evaluation.human_eval.protocol --analyse \
        --ratings outputs/evaluation/human_ratings.jsonl \
        --output outputs/evaluation/human_eval_report.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


# ============================================================================
# Study Design
# ============================================================================

MOS_SCALES = {
    "naturalness": {
        "question": "How natural does the dubbed Armenian speech sound?",
        "anchors": {1: "Very unnatural", 2: "Unnatural", 3: "Fair", 4: "Natural", 5: "Very natural"},
    },
    "intelligibility": {
        "question": "How easy is it to understand the Armenian speech?",
        "anchors": {1: "Unintelligible", 2: "Difficult", 3: "Fair", 4: "Easy", 5: "Very easy"},
    },
    "speaker_similarity": {
        "question": "How similar is the dubbed voice to the original speaker?",
        "anchors": {1: "Completely different", 2: "Different", 3: "Somewhat similar", 4: "Similar", 5: "Identical"},
    },
    "lipsync_quality": {
        "question": "How well do the lip movements match the dubbed speech?",
        "anchors": {1: "Completely mismatched", 2: "Poor", 3: "Fair", 4: "Good", 5: "Excellent"},
    },
    "overall_quality": {
        "question": "Rate the overall quality of this dubbed video.",
        "anchors": {1: "Very bad", 2: "Bad", 3: "Fair", 4: "Good", 5: "Excellent"},
    },
}

AB_PREFERENCE = {
    "question": "Which dubbed video do you prefer overall?",
    "options": ["A is much better", "A is better", "About the same", "B is better", "B is much better"],
}

EVALUATOR_REQUIREMENTS = {
    "language": "Native Eastern Armenian speaker (or fluent Western Armenian)",
    "age_range": "18-65",
    "min_evaluators": 10,
    "recommended_evaluators": 20,
    "samples_per_evaluator": 30,
    "max_session_minutes": 45,
    "break_after_minutes": 20,
    "compensation_note": "Fair compensation per local standards",
}


# ============================================================================
# Task Generation
# ============================================================================

class HumanEvalTaskGenerator:
    """Generate evaluation tasks for human annotators."""

    def __init__(self, study_type: str = "mos"):
        """
        Args:
            study_type: "mos" for Mean Opinion Score, "ab" for A/B preference,
                        "full" for both.
        """
        self.study_type = study_type

    def generate_mos_tasks(
        self,
        samples: List[Dict],
        randomize: bool = True,
    ) -> List[Dict]:
        """Generate MOS rating tasks from sample list.

        Args:
            samples: List of dicts with "video_path", "audio_path", optionally
                     "original_video_path", "sample_id".
            randomize: Shuffle task order to reduce bias.

        Returns:
            List of task dicts ready for annotation interface.
        """
        tasks = []

        for i, sample in enumerate(samples):
            sample_id = sample.get("sample_id", f"sample_{i:04d}")
            task = {
                "task_id": f"mos_{sample_id}",
                "task_type": "mos",
                "sample_id": sample_id,
                "video_path": sample.get("video_path", ""),
                "audio_path": sample.get("audio_path", ""),
                "original_video_path": sample.get("original_video_path", ""),
                "scales": {},
            }

            for scale_name, scale_def in MOS_SCALES.items():
                task["scales"][scale_name] = {
                    "question": scale_def["question"],
                    "min": 1,
                    "max": 5,
                    "anchors": scale_def["anchors"],
                    "rating": None,  # To be filled by evaluator
                }

            # Free-text comment
            task["comment"] = ""
            tasks.append(task)

        if randomize:
            import random
            random.shuffle(tasks)

        return tasks

    def generate_ab_tasks(
        self,
        pairs: List[Dict],
        randomize: bool = True,
    ) -> List[Dict]:
        """Generate A/B preference tasks.

        Args:
            pairs: List of dicts with "video_a" and "video_b" paths.
            randomize: Shuffle and randomly swap A/B to reduce bias.

        Returns:
            List of A/B comparison task dicts.
        """
        import random

        tasks = []

        for i, pair in enumerate(pairs):
            pair_id = pair.get("pair_id", f"pair_{i:04d}")

            # Randomly swap A/B to eliminate positional bias
            swap = random.random() > 0.5 if randomize else False
            video_a = pair["video_b"] if swap else pair["video_a"]
            video_b = pair["video_a"] if swap else pair["video_b"]

            task = {
                "task_id": f"ab_{pair_id}",
                "task_type": "ab_preference",
                "pair_id": pair_id,
                "video_a": video_a,
                "video_b": video_b,
                "swapped": swap,
                "system_a": pair.get("system_b" if swap else "system_a", "unknown"),
                "system_b": pair.get("system_a" if swap else "system_b", "unknown"),
                "question": AB_PREFERENCE["question"],
                "options": AB_PREFERENCE["options"],
                "preference": None,  # To be filled by evaluator
                "comment": "",
            }
            tasks.append(task)

        if randomize:
            random.shuffle(tasks)

        return tasks

    def generate_labelstudio_config(self) -> str:
        """Generate Label Studio XML config for MOS evaluation."""
        xml = """<View>
  <Header value="Armenian Video Dubbing — Human Evaluation"/>

  <Video name="dubbed_video" value="$video_path"/>

  <Header value="Original video (for reference)"/>
  <Video name="original_video" value="$original_video_path" visibleWhen="region-selected"/>

  <Header value="Rate the dubbed video on the following scales (1-5):"/>

  <Rating name="naturalness" toName="dubbed_video" maxRating="5" icon="star" size="large"/>
  <Text name="naturalness_label" value="Naturalness: How natural does the Armenian speech sound?"/>

  <Rating name="intelligibility" toName="dubbed_video" maxRating="5" icon="star" size="large"/>
  <Text name="intelligibility_label" value="Intelligibility: How easy is it to understand?"/>

  <Rating name="speaker_similarity" toName="dubbed_video" maxRating="5" icon="star" size="large"/>
  <Text name="speaker_similarity_label" value="Speaker Similarity: How similar to the original voice?"/>

  <Rating name="lipsync_quality" toName="dubbed_video" maxRating="5" icon="star" size="large"/>
  <Text name="lipsync_label" value="Lip-Sync: How well do lips match the speech?"/>

  <Rating name="overall_quality" toName="dubbed_video" maxRating="5" icon="star" size="large"/>
  <Text name="overall_label" value="Overall Quality: Rate the overall dubbing quality."/>

  <TextArea name="comment" toName="dubbed_video" placeholder="Optional comments..."
            rows="3" maxSubmissions="1"/>
</View>"""
        return xml


# ============================================================================
# Statistical Analysis
# ============================================================================

class HumanEvalAnalyser:
    """Analyse collected human evaluation ratings."""

    def analyse_mos_ratings(self, ratings: List[Dict]) -> Dict:
        """Compute MOS statistics from collected ratings.

        Args:
            ratings: List of completed task dicts (with ratings filled in).

        Returns:
            Dict with per-scale MOS, confidence intervals, inter-rater agreement.
        """
        # Organize ratings by scale
        scale_ratings: Dict[str, List[float]] = {s: [] for s in MOS_SCALES}
        evaluator_ratings: Dict[str, Dict[str, List[float]]] = {}

        for task in ratings:
            evaluator_id = task.get("evaluator_id", "unknown")
            if evaluator_id not in evaluator_ratings:
                evaluator_ratings[evaluator_id] = {s: [] for s in MOS_SCALES}

            scales = task.get("scales", {})
            for scale_name in MOS_SCALES:
                rating = scales.get(scale_name, {}).get("rating")
                if rating is not None and 1 <= rating <= 5:
                    scale_ratings[scale_name].append(float(rating))
                    evaluator_ratings[evaluator_id][scale_name].append(float(rating))

        results = {}

        for scale_name, vals in scale_ratings.items():
            if not vals:
                results[scale_name] = {"mos": 0, "n_ratings": 0}
                continue

            arr = np.array(vals)
            ci95 = 1.96 * np.std(arr) / np.sqrt(len(arr)) if len(arr) > 1 else 0

            results[scale_name] = {
                "mos": round(float(np.mean(arr)), 3),
                "std": round(float(np.std(arr)), 3),
                "ci_95": round(float(ci95), 3),
                "median": float(np.median(arr)),
                "n_ratings": len(vals),
                "distribution": {
                    str(i): int(np.sum(arr == i)) for i in range(1, 6)
                },
            }

        # Inter-rater agreement (Krippendorff's alpha approximation)
        agreement = self._compute_inter_rater_agreement(evaluator_ratings)
        results["inter_rater_agreement"] = agreement

        # Sample-level aggregation for worst/best identification
        results["n_evaluators"] = len(evaluator_ratings)
        results["n_total_ratings"] = sum(len(v) for v in scale_ratings.values())

        return results

    def analyse_ab_preferences(self, ratings: List[Dict]) -> Dict:
        """Compute A/B preference statistics.

        Args:
            ratings: List of completed A/B task dicts.

        Returns:
            Dict with preference counts, rates, and significance test.
        """
        # Map options to numeric scores: -2 (A much better) to +2 (B much better)
        option_scores = {
            "A is much better": -2,
            "A is better": -1,
            "About the same": 0,
            "B is better": 1,
            "B is much better": 2,
        }

        scores = []
        counts = {opt: 0 for opt in AB_PREFERENCE["options"]}

        for task in ratings:
            pref = task.get("preference")
            if pref in option_scores:
                scores.append(option_scores[pref])
                counts[pref] += 1

        if not scores:
            return {"n_comparisons": 0}

        arr = np.array(scores)

        # Preference rates
        n = len(scores)
        prefer_a = np.sum(arr < 0) / n
        prefer_b = np.sum(arr > 0) / n
        no_pref = np.sum(arr == 0) / n

        # Binomial test for significance (is preference significantly != 50/50?)
        from scipy import stats
        try:
            n_a = int(np.sum(arr < 0))
            n_b = int(np.sum(arr > 0))
            _, p_value = stats.binomtest(n_a, n_a + n_b, 0.5) if (n_a + n_b > 0) else (None, 1.0)
        except Exception:
            p_value = 1.0

        return {
            "n_comparisons": n,
            "prefer_a_rate": round(float(prefer_a), 3),
            "prefer_b_rate": round(float(prefer_b), 3),
            "no_preference_rate": round(float(no_pref), 3),
            "mean_score": round(float(np.mean(arr)), 3),
            "p_value": round(float(p_value), 4),
            "significant_at_005": p_value < 0.05,
            "counts": counts,
        }

    def _compute_inter_rater_agreement(
        self,
        evaluator_ratings: Dict[str, Dict[str, List[float]]],
    ) -> Dict:
        """Compute inter-rater agreement (simplified Krippendorff's alpha).

        Uses pairwise correlation as a practical approximation.
        """
        if len(evaluator_ratings) < 2:
            return {"alpha": 0, "note": "Need >= 2 evaluators"}

        # Compute pairwise Pearson correlation across evaluators
        evaluator_ids = list(evaluator_ratings.keys())
        correlations = []

        for i in range(len(evaluator_ids)):
            for j in range(i + 1, len(evaluator_ids)):
                # Flatten all scales into a single vector per evaluator
                flat_i = []
                flat_j = []
                for scale in MOS_SCALES:
                    ri = evaluator_ratings[evaluator_ids[i]][scale]
                    rj = evaluator_ratings[evaluator_ids[j]][scale]
                    min_len = min(len(ri), len(rj))
                    if min_len > 0:
                        flat_i.extend(ri[:min_len])
                        flat_j.extend(rj[:min_len])

                if len(flat_i) >= 3:
                    corr = float(np.corrcoef(flat_i, flat_j)[0, 1])
                    if np.isfinite(corr):
                        correlations.append(corr)

        if not correlations:
            return {"alpha": 0, "note": "Insufficient overlap"}

        return {
            "mean_pairwise_correlation": round(float(np.mean(correlations)), 3),
            "min_pairwise_correlation": round(float(np.min(correlations)), 3),
            "n_pairs": len(correlations),
            "acceptable": float(np.mean(correlations)) >= 0.6,
        }

    def generate_report(
        self,
        mos_results: Optional[Dict] = None,
        ab_results: Optional[Dict] = None,
    ) -> str:
        """Generate human-readable evaluation report.

        Returns:
            Formatted report string.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("HUMAN EVALUATION REPORT")
        lines.append("=" * 60)

        if mos_results:
            lines.append("\n--- MOS Results ---")
            for scale_name, data in mos_results.items():
                if isinstance(data, dict) and "mos" in data:
                    lines.append(
                        f"  {scale_name:<25} MOS={data['mos']:.2f} "
                        f"(+/-{data['ci_95']:.2f}, n={data['n_ratings']})"
                    )

            agreement = mos_results.get("inter_rater_agreement", {})
            if "mean_pairwise_correlation" in agreement:
                lines.append(
                    f"\n  Inter-rater agreement: r={agreement['mean_pairwise_correlation']:.3f} "
                    f"({'acceptable' if agreement.get('acceptable') else 'LOW'})"
                )

            lines.append(f"  Total evaluators: {mos_results.get('n_evaluators', '?')}")

        if ab_results:
            lines.append("\n--- A/B Preference ---")
            lines.append(f"  Prefer A: {ab_results.get('prefer_a_rate', 0):.1%}")
            lines.append(f"  Prefer B: {ab_results.get('prefer_b_rate', 0):.1%}")
            lines.append(f"  No preference: {ab_results.get('no_preference_rate', 0):.1%}")
            lines.append(f"  p-value: {ab_results.get('p_value', 1):.4f}")
            sig = "YES" if ab_results.get("significant_at_005") else "NO"
            lines.append(f"  Significant (p<0.05): {sig}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Human Evaluation Protocol")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true", help="Generate evaluation tasks")
    group.add_argument("--analyse", action="store_true", help="Analyse collected ratings")
    group.add_argument("--labelstudio-config", action="store_true", help="Print LabelStudio XML config")

    parser.add_argument("--samples", type=str, help="Path to samples JSONL (for --generate)")
    parser.add_argument("--ratings", type=str, help="Path to collected ratings JSONL (for --analyse)")
    parser.add_argument("--output", type=str, default="outputs/evaluation/human_eval.json")
    parser.add_argument("--study-type", type=str, default="full", choices=["mos", "ab", "full"])

    args = parser.parse_args()

    if args.generate:
        if not args.samples:
            logger.error("--samples is required for --generate")
            sys.exit(1)

        samples = []
        with open(args.samples) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        generator = HumanEvalTaskGenerator(study_type=args.study_type)
        tasks = generator.generate_mos_tasks(samples)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)

        logger.info("Generated {} evaluation tasks -> {}", len(tasks), output_path)
        logger.info("Evaluator requirements: {}", EVALUATOR_REQUIREMENTS)

    elif args.analyse:
        if not args.ratings:
            logger.error("--ratings is required for --analyse")
            sys.exit(1)

        ratings = []
        with open(args.ratings) as f:
            for line in f:
                line = line.strip()
                if line:
                    ratings.append(json.loads(line))

        analyser = HumanEvalAnalyser()

        mos_results = analyser.analyse_mos_ratings(ratings)
        ab_results = analyser.analyse_ab_preferences(ratings)

        report_text = analyser.generate_report(mos_results, ab_results)
        print(report_text)

        # Save JSON
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "mos_results": mos_results,
                "ab_results": ab_results,
            }, f, indent=2, default=str)

        logger.info("Report saved to {}", output_path)

    elif args.labelstudio_config:
        gen = HumanEvalTaskGenerator()
        print(gen.generate_labelstudio_config())


if __name__ == "__main__":
    main()
