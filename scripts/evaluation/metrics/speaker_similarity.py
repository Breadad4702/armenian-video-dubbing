#!/usr/bin/env python3
"""
Speaker Similarity Computation for Voice Cloning Evaluation.

Measures how well synthesized speech matches reference speaker using:
- Speaker embeddings (Resemblyzer, WavLM)
- Cosine similarity computation
- Confidence intervals via resampling
- Per-speaker evaluation
- Failure detection
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
import torch
from loguru import logger
from scipy.spatial.distance import cosine

try:
    from resemblyzer import VoiceEncoder
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    logger.warning("Resemblyzer not available (optional)")

from src.utils.helpers import load_audio, timer


class SpeakerSimilarityComputer:
    """Compute speaker similarity for voice cloning evaluation."""

    def __init__(self, device: str = "cuda"):
        """
        Initialize speaker similarity computer.

        Args:
            device: Device to run on ("cuda" or "cpu")
        """
        self.device = device

        # Load speaker embedder
        if RESEMBLYZER_AVAILABLE:
            try:
                self.encoder = VoiceEncoder(device=device)
                logger.info("Loaded Resemblyzer for speaker embeddings")
            except Exception as e:
                logger.warning(f"Failed to load Resemblyzer: {e}")
                self.encoder = None
        else:
            self.encoder = None
            logger.warning("Resemblyzer not available, speaker similarity won't work")

    def compute_speaker_similarity(
        self,
        synthesized_audio: np.ndarray,
        reference_audio: np.ndarray,
        sample_rate: int = 44100,
    ) -> Dict:
        """
        Compute speaker similarity between synthesized and reference audio.

        Args:
            synthesized_audio: Synthesized speech waveform
            reference_audio: Reference speaker audio waveform
            sample_rate: Sample rate of audio

        Returns:
            Dictionary with similarity score and metadata
        """
        if self.encoder is None:
            logger.warning("Speaker embedder not available")
            return {"error": "Embedder not available"}

        with timer("Speaker similarity computation"):
            # Extract embeddings
            synth_embed = self.encoder.embed_utterance(synthesized_audio)
            ref_embed = self.encoder.embed_utterance(reference_audio)

            # Compute cosine similarity
            similarity = 1 - cosine(synth_embed, ref_embed)

            logger.info(f"Speaker similarity: {similarity:.4f}")

            return {
                "similarity": float(similarity),
                "synth_embed_dim": int(synth_embed.shape[0]),
                "reference_embed_dim": int(ref_embed.shape[0]),
                "passes_threshold": bool(similarity > 0.75),
            }

    def batch_similarity_evaluation(
        self,
        synthesized_list: List[np.ndarray],
        reference_list: List[np.ndarray],
        sample_rate: int = 44100,
    ) -> Dict:
        """
        Evaluate speaker similarity for multiple samples.

        Args:
            synthesized_list: List of synthesized audio waveforms
            reference_list: List of reference speaker waveforms
            sample_rate: Sample rate of audio

        Returns:
            Dictionary with batch similarity metrics
        """
        if len(synthesized_list) != len(reference_list):
            raise ValueError("Synthesized and reference lists must have same length")

        similarities = []
        failures = []

        logger.info(f"Computing speaker similarity for {len(synthesized_list)} pairs")

        for i, (synth, ref) in enumerate(zip(synthesized_list, reference_list)):
            try:
                result = self.compute_speaker_similarity(synth, ref, sample_rate)

                if "error" not in result:
                    similarities.append(result["similarity"])

                    if result["similarity"] < 0.75:
                        failures.append({
                            "sample_idx": i,
                            "similarity": result["similarity"],
                        })

            except Exception as e:
                logger.error(f"Failed to compute similarity for sample {i}: {e}")
                failures.append({
                    "sample_idx": i,
                    "error": str(e),
                })

        if not similarities:
            logger.warning("No valid similarities computed")
            return {"error": "No valid similarities"}

        return {
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "n_samples": len(similarities),
            "failures": failures,
            "failure_rate": len(failures) / len(synthesized_list),
        }

    def similarity_with_confidence(
        self,
        synthesized_audio: np.ndarray,
        reference_audio: np.ndarray,
        sample_rate: int = 44100,
        n_chunks: int = 3,
    ) -> Dict:
        """
        Compute speaker similarity with confidence via frame splitting.

        Args:
            synthesized_audio: Synthesized speech waveform
            reference_audio: Reference speaker waveform
            sample_rate: Sample rate
            n_chunks: Number of chunks for confidence estimation

        Returns:
            Dictionary with similarity and confidence interval
        """
        # Split audio into chunks
        synth_len = len(synthesized_audio)
        chunk_size = synth_len // n_chunks

        similarities = []

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < n_chunks - 1 else synth_len

            synth_chunk = synthesized_audio[start_idx:end_idx]

            # Use whole reference audio for each chunk
            result = self.compute_speaker_similarity(synth_chunk, reference_audio, sample_rate)

            if "similarity" in result:
                similarities.append(result["similarity"])

        if not similarities:
            return {"error": "No valid similarities computed"}

        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)

        # Confidence interval (95%)
        ci_lower = mean_sim - 1.96 * std_sim / np.sqrt(len(similarities))
        ci_upper = mean_sim + 1.96 * std_sim / np.sqrt(len(similarities))

        return {
            "similarity": float(mean_sim),
            "confidence_interval": (float(ci_lower), float(ci_upper)),
            "n_chunks": n_chunks,
            "per_chunk_similarities": [float(s) for s in similarities],
        }

    def identify_voice_cloning_failures(
        self,
        similarity_scores: List[float],
        threshold: float = 0.75,
    ) -> Dict:
        """
        Identify speakers with poor voice cloning (low similarity).

        Args:
            similarity_scores: List of speaker similarity scores
            threshold: Minimum acceptable similarity

        Returns:
            Dictionary with failure analysis
        """
        failed_indices = [
            i for i, sim in enumerate(similarity_scores)
            if sim < threshold
        ]

        if not failed_indices:
            logger.info("All voice cloning attempts passed threshold")
            return {
                "failures_detected": False,
                "failed_count": 0,
            }

        logger.warning(f"Voice cloning failures: {len(failed_indices)} / {len(similarity_scores)}")

        failed_scores = [similarity_scores[i] for i in failed_indices]

        return {
            "failures_detected": True,
            "failed_count": len(failed_indices),
            "failure_rate": len(failed_indices) / len(similarity_scores),
            "failed_indices": failed_indices,
            "failed_similarities": failed_scores,
            "mean_failed_similarity": float(np.mean(failed_scores)),
            "min_failed_similarity": float(np.min(failed_scores)),
        }

    def per_speaker_similarity_analysis(
        self,
        speaker_pairs: Dict[str, Tuple[np.ndarray, np.ndarray]],
        sample_rate: int = 44100,
    ) -> Dict:
        """
        Analyze voice cloning performance per speaker.

        Args:
            speaker_pairs: Dict mapping speaker_id to (synth_audio, ref_audio) tuples
            sample_rate: Sample rate

        Returns:
            Dictionary with per-speaker metrics
        """
        results = {}

        logger.info(f"Analyzing {len(speaker_pairs)} speakers")

        for speaker_id, (synth_audio, ref_audio) in speaker_pairs.items():
            try:
                result = self.compute_speaker_similarity(synth_audio, ref_audio, sample_rate)

                results[speaker_id] = {
                    "similarity": result.get("similarity", None),
                    "passes_threshold": result.get("passes_threshold", False),
                }

            except Exception as e:
                logger.error(f"Failed to evaluate speaker {speaker_id}: {e}")
                results[speaker_id] = {
                    "error": str(e),
                }

        # Aggregate statistics
        valid_similarities = [
            r["similarity"] for r in results.values()
            if "similarity" in r and r["similarity"] is not None
        ]

        if not valid_similarities:
            return {"error": "No valid similarities computed"}

        return {
            "per_speaker_results": results,
            "mean_similarity": float(np.mean(valid_similarities)),
            "std_similarity": float(np.std(valid_similarities)),
            "min_similarity": float(np.min(valid_similarities)),
            "max_similarity": float(np.max(valid_similarities)),
            "n_speakers": len(results),
            "n_pass_threshold": sum(1 for r in results.values() if r.get("passes_threshold", False)),
            "threshold_pass_rate": sum(1 for r in results.values() if r.get("passes_threshold", False)) / len(results),
        }

    def compute_from_manifest(self, manifest_path: str) -> Dict:
        """Evaluate speaker similarity on a JSONL manifest of paired audio.

        Each line must have "audio_path" (synthesized) and "reference_audio_path".

        Args:
            manifest_path: Path to JSONL manifest file.

        Returns:
            Dictionary with aggregated speaker similarity statistics.
        """
        import json as _json

        manifest = Path(manifest_path)
        if not manifest.exists():
            return {"error": f"Manifest not found: {manifest_path}"}

        similarities = []
        errors = 0

        with open(manifest) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = _json.loads(line)
                synth_path = sample.get("audio_path", "")
                ref_path = sample.get("reference_audio_path", "")
                if not synth_path or not ref_path:
                    errors += 1
                    continue
                if not Path(synth_path).exists() or not Path(ref_path).exists():
                    errors += 1
                    continue

                try:
                    synth_audio, _ = load_audio(synth_path)
                    ref_audio, _ = load_audio(ref_path)
                    result = self.compute_speaker_similarity(synth_audio, ref_audio)
                    if "similarity" in result:
                        similarities.append(result["similarity"])
                except Exception as e:
                    logger.debug(f"Speaker similarity failed for {synth_path}: {e}")
                    errors += 1

        if not similarities:
            return {"mean_similarity": 0, "n_samples": 0, "errors": errors}

        return {
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "n_samples": len(similarities),
            "pass_rate": float(np.mean([s >= 0.85 for s in similarities])),
            "errors": errors,
        }

    def export_speaker_embeddings(
        self,
        audio_list: List[np.ndarray],
        export_path: str,
        labels: Optional[List[str]] = None,
    ) -> None:
        """
        Export speaker embeddings to file for visualization/analysis.

        Args:
            audio_list: List of audio waveforms
            export_path: Path to save embeddings
            labels: Optional labels for each embedding
        """
        embeddings = []

        logger.info(f"Extracting embeddings for {len(audio_list)} samples")

        for i, audio in enumerate(audio_list):
            try:
                embed = self.encoder.embed_utterance(audio)
                embeddings.append({
                    "embedding": embed.tolist(),
                    "label": labels[i] if labels else f"sample_{i}",
                    "dim": len(embed),
                })
            except Exception as e:
                logger.error(f"Failed to extract embedding for sample {i}: {e}")

        # Save as JSON
        with open(export_path, "w") as f:
            json.dump(embeddings, f, indent=2)

        logger.info(f"Saved {len(embeddings)} embeddings to {export_path}")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python speaker_similarity.py <synthesized.wav> <reference.wav>")
        sys.exit(1)

    synth_path = sys.argv[1]
    ref_path = sys.argv[2]

    synth_audio, sr = load_audio(synth_path)
    ref_audio, _ = load_audio(ref_path)

    computer = SpeakerSimilarityComputer()

    # Compute similarity
    result = computer.compute_speaker_similarity(synth_audio, ref_audio)

    print("\n=== Speaker Similarity Results ===")
    print(f"Similarity: {result.get('similarity', 'N/A'):.4f}")
    print(f"Passes Threshold (0.75): {result.get('passes_threshold', False)}")

    # With confidence
    conf_result = computer.similarity_with_confidence(synth_audio, ref_audio)
    print(f"\nWith Confidence Interval:")
    print(f"Similarity: {conf_result['similarity']:.4f}")
    print(f"95% CI: {conf_result['confidence_interval']}")
