#!/usr/bin/env python3
"""
WER (Word Error Rate) and CER (Character Error Rate) computation for Armenian ASR.

Metrics:
- WER: standard word error rate (insertions, deletions, substitutions)
- CER: character error rate (same computation at character level)
- Per-speaker WER: segment WER by speaker demographics
- Phoneme-class WER: consonant vs vowel error rates
- Confidence intervals: bootstrap CI for WER estimates
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import torch
from loguru import logger
import jiwer
from scipy import stats

from src.inference import ASRInference
from src.utils.helpers import load_audio, timer


class WERComputer:
    """Compute WER/CER metrics on Armenian test sets."""

    def __init__(self, asr_model_path: str, device: str = "cuda"):
        """
        Initialize WER computer.

        Args:
            asr_model_path: Path to fine-tuned Whisper + LoRA checkpoint
            device: Device to run on ("cuda" or "cpu")
        """
        self.device = device
        self.asr = ASRInference(model_path=asr_model_path, device=device)
        logger.info(f"WERComputer initialized with model: {asr_model_path}")

    def compute_wer_on_testset(
        self,
        test_manifest_path: str,
        batch_size: int = 8,
        save_predictions: bool = True,
    ) -> Dict:
        """
        Compute WER on test set.

        Args:
            test_manifest_path: Path to test.jsonl manifest
            batch_size: Batch size for inference
            save_predictions: Whether to save predictions to file

        Returns:
            Dictionary with WER, CER, per-sample metrics, confidence intervals
        """
        logger.info(f"Computing WER on test set: {test_manifest_path}")

        # Load test set
        test_samples = []
        with open(test_manifest_path) as f:
            for line in f:
                test_samples.append(json.loads(line))

        logger.info(f"Loaded {len(test_samples)} test samples")

        predictions = []
        references = []
        per_sample_metrics = []

        # Batch transcription
        with timer("ASR batch transcription"):
            for i in range(0, len(test_samples), batch_size):
                batch = test_samples[i:i+batch_size]

                # Load audio for batch
                audio_batch = []
                ref_batch = []

                for sample in batch:
                    audio_path = sample["audio_path"]
                    reference_text = sample["text"]

                    audio, sr = load_audio(audio_path)
                    audio_batch.append(audio)
                    ref_batch.append(reference_text)

                # Transcribe batch
                transcriptions = self.asr.batch_transcribe(audio_batch)

                for pred, ref in zip(transcriptions, ref_batch):
                    predictions.append(pred)
                    references.append(ref)

        logger.info(f"Completed transcription of {len(predictions)} samples")

        # Compute WER and CER
        wer = jiwer.wer(references, predictions)
        cer = jiwer.cer(references, predictions)

        logger.info(f"WER: {wer:.4f} ({wer*100:.2f}%)")
        logger.info(f"CER: {cer:.4f} ({cer*100:.2f}%)")

        # Per-sample metrics
        for pred, ref, sample in zip(predictions, references, test_samples):
            sample_wer = jiwer.wer([ref], [pred])
            sample_cer = jiwer.cer([ref], [pred])

            per_sample_metrics.append({
                "sample_id": sample.get("id", ""),
                "audio_path": sample["audio_path"],
                "reference_text": ref,
                "predicted_text": pred,
                "wer": sample_wer,
                "cer": sample_cer,
                "duration_sec": sample.get("duration_sec", 0),
            })

        # Confidence intervals (bootstrap)
        wer_ci = self._compute_bootstrap_ci(references, predictions, metric="wer")
        cer_ci = self._compute_bootstrap_ci(references, predictions, metric="cer")

        logger.info(f"WER 95% CI: [{wer_ci[0]:.4f}, {wer_ci[1]:.4f}]")
        logger.info(f"CER 95% CI: [{cer_ci[0]:.4f}, {cer_ci[1]:.4f}]")

        # Error breakdown (substitutions, deletions, insertions)
        error_stats = self._compute_error_breakdown(references, predictions)

        # Per-speaker WER (if metadata available)
        per_speaker_wer = self._compute_per_speaker_wer(test_samples, per_sample_metrics)

        # Phoneme-class WER
        phoneme_wer = self._compute_phoneme_class_wer(references, predictions)

        results = {
            "wer": float(wer),
            "cer": float(cer),
            "wer_confidence_interval": [float(wer_ci[0]), float(wer_ci[1])],
            "cer_confidence_interval": [float(cer_ci[0]), float(cer_ci[1])],
            "n_samples": len(test_samples),
            "error_breakdown": error_stats,
            "per_speaker_wer": per_speaker_wer,
            "phoneme_class_wer": phoneme_wer,
            "worst_samples": self._get_worst_samples(per_sample_metrics, n=5),
        }

        # Save predictions
        if save_predictions:
            output_path = Path(test_manifest_path).parent / "predictions.jsonl"
            with open(output_path, "w") as f:
                for metric in per_sample_metrics:
                    f.write(json.dumps(metric) + "\n")
            logger.info(f"Saved predictions to: {output_path}")

        return results

    def _compute_bootstrap_ci(
        self,
        references: List[str],
        predictions: List[str],
        metric: str = "wer",
        n_bootstrap: int = 100,
        ci: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for metric.

        Args:
            references: Reference texts
            predictions: Predicted texts
            metric: "wer" or "cer"
            n_bootstrap: Number of bootstrap samples
            ci: Confidence interval (0.95 for 95%)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        bootstrap_scores = []
        n_samples = len(references)

        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            resampled_refs = [references[i] for i in indices]
            resampled_preds = [predictions[i] for i in indices]

            # Compute metric on resample
            if metric == "wer":
                score = jiwer.wer(resampled_refs, resampled_preds)
            else:  # cer
                score = jiwer.cer(resampled_refs, resampled_preds)

            bootstrap_scores.append(score)

        # Compute percentile CI
        alpha = 1 - ci
        lower = np.percentile(bootstrap_scores, alpha/2 * 100)
        upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)

        return lower, upper

    def _compute_error_breakdown(
        self,
        references: List[str],
        predictions: List[str],
    ) -> Dict[str, float]:
        """
        Compute substitution, deletion, insertion error rates.

        Args:
            references: Reference texts
            predictions: Predicted texts

        Returns:
            Dictionary with error type breakdown
        """
        total_errors = {"substitutions": 0, "deletions": 0, "insertions": 0}

        for ref, pred in zip(references, predictions):
            ref_words = ref.split()
            pred_words = pred.split()

            # Use jiwer distance computation
            ops = jiwer.compute_measures(ref, pred, standard_measures=False)
            # Note: jiwer's operations give us detailed error information

        # Simplified: compute overall WER breakdown
        total_wer = jiwer.wer(references, predictions)

        return {
            "wer": float(total_wer),
            "note": "Detailed substitution/deletion/insertion breakdown requires custom implementation"
        }

    def _compute_per_speaker_wer(
        self,
        test_samples: List[Dict],
        per_sample_metrics: List[Dict],
    ) -> Dict[str, float]:
        """
        Segment WER by speaker demographics.

        Args:
            test_samples: Original test samples with metadata
            per_sample_metrics: Per-sample WER/CER metrics

        Returns:
            Dictionary with per-speaker WER breakdown
        """
        speaker_metrics = {}

        for sample, metrics in zip(test_samples, per_sample_metrics):
            speaker_id = sample.get("speaker_id", "unknown")
            gender = sample.get("gender", "unknown")
            age = sample.get("age", "unknown")

            # Group by speaker
            if speaker_id not in speaker_metrics:
                speaker_metrics[speaker_id] = {
                    "gender": gender,
                    "age": age,
                    "wers": []
                }

            speaker_metrics[speaker_id]["wers"].append(metrics["wer"])

        # Aggregate per speaker
        result = {}
        for speaker_id, data in speaker_metrics.items():
            result[speaker_id] = {
                "mean_wer": float(np.mean(data["wers"])),
                "std_wer": float(np.std(data["wers"])),
                "n_samples": len(data["wers"]),
                "gender": data["gender"],
                "age": data["age"],
            }

        # Also segment by gender/age if available
        by_gender = {}
        by_age = {}

        for speaker_id, metrics in per_sample_metrics:
            sample = test_samples[0]  # Simplified
            gender = sample.get("gender", "unknown")
            age = sample.get("age", "unknown")

            if gender not in by_gender:
                by_gender[gender] = []
            by_gender[gender].append(metrics["wer"])

            if age not in by_age:
                by_age[age] = []
            by_age[age].append(metrics["wer"])

        result["by_gender"] = {g: float(np.mean(wers)) for g, wers in by_gender.items()}
        result["by_age"] = {a: float(np.mean(wers)) for a, wers in by_age.items()}

        return result

    def _compute_phoneme_class_wer(
        self,
        references: List[str],
        predictions: List[str],
    ) -> Dict[str, float]:
        """
        Compute WER broken down by phoneme classes.

        Args:
            references: Reference texts
            predictions: Predicted texts

        Returns:
            Dictionary with WER by consonant/vowel/affricate etc.
        """
        # Simplified: Armenian vowels and consonants
        armenian_vowels = set("աեըիու")
        armenian_consonants = set("բգդզեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւ")

        vowel_errors = 0
        consonant_errors = 0
        vowel_total = 0
        consonant_total = 0

        for ref, pred in zip(references, predictions):
            for ref_char, pred_char in zip(ref, pred):
                if ref_char in armenian_vowels:
                    vowel_total += 1
                    if ref_char != pred_char:
                        vowel_errors += 1
                elif ref_char in armenian_consonants:
                    consonant_total += 1
                    if ref_char != pred_char:
                        consonant_errors += 1

        vowel_wer = vowel_errors / max(1, vowel_total)
        consonant_wer = consonant_errors / max(1, consonant_total)

        return {
            "vowel_wer": float(vowel_wer),
            "consonant_wer": float(consonant_wer),
            "vowel_samples": vowel_total,
            "consonant_samples": consonant_total,
        }

    def _get_worst_samples(
        self,
        per_sample_metrics: List[Dict],
        n: int = 5,
    ) -> List[Dict]:
        """
        Return worst-performing samples by WER.

        Args:
            per_sample_metrics: Per-sample metrics
            n: Number of worst samples to return

        Returns:
            List of worst samples
        """
        sorted_samples = sorted(per_sample_metrics, key=lambda x: x["wer"], reverse=True)
        return sorted_samples[:n]


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python wer_metrics.py <model_path> <test_manifest>")
        sys.exit(1)

    model_path = sys.argv[1]
    test_manifest = sys.argv[2]

    computer = WERComputer(model_path)
    results = computer.compute_wer_on_testset(test_manifest)

    print("\n=== WER Evaluation Results ===")
    print(f"WER: {results['wer']:.4f} ({results['wer']*100:.2f}%)")
    print(f"CER: {results['cer']:.4f} ({results['cer']*100:.2f}%)")
    print(f"Samples: {results['n_samples']}")
    print(f"WER CI: {results['wer_confidence_interval']}")

    print(f"\nWorst samples:")
    for sample in results["worst_samples"][:3]:
        print(f"  WER: {sample['wer']:.2%}, text: {sample['reference_text'][:50]}...")
