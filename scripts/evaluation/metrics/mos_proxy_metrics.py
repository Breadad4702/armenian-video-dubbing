#!/usr/bin/env python3
"""
MOS (Mean Opinion Score) Proxy Estimation for Armenian TTS.

Fast estimation of MOS without human listeners using:
- Prosody analysis (pitch, energy, duration variance)
- Spectral quality assessment (MFCC, mel-spectrogram)
- Artifact detection (clicks, distortion, background hum)
- Emotion preservation scoring
- Speaker consistency measurement
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import json

import numpy as np
import librosa
import torch
from loguru import logger
from scipy import signal
from scipy.stats import entropy

try:
    from resemblyzer import VoiceEncoder
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    logger.warning("Resemblyzer not available (optional)")

from src.utils.helpers import load_audio, timer


class ProsodyAnalyzer:
    """Extract and analyze prosody features (pitch, energy, duration)."""

    def __init__(self, sample_rate: int = 16000):
        """Initialize prosody analyzer."""
        self.sr = sample_rate

    def extract_prosody_features(self, audio: np.ndarray) -> Dict:
        """
        Extract F0 (fundamental frequency), energy, duration features.

        Args:
            audio: Audio waveform (numpy array)

        Returns:
            Dictionary with prosody statistics
        """
        # Compute F0 using librosa (simplified, uses autocorrelation)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=50,
            fmax=500,
            sr=self.sr,
        )

        # Remove unvoiced frames
        f0_voiced = f0[voiced_flag]

        if len(f0_voiced) == 0:
            logger.warning("No voiced frames detected in audio")
            return {
                "f0_mean": 0.0,
                "f0_std": 0.0,
                "f0_range": (0.0, 0.0),
                "voiced_ratio": 0.0,
            }

        # Energy analysis
        S = librosa.feature.melspectrogram(y=audio, sr=self.sr)
        energy = librosa.feature.melspectrogram(y=audio, sr=self.sr, power=2.0)
        energy_db = librosa.power_to_db(energy, ref=np.max)
        energy_mean = np.mean(energy_db)
        energy_std = np.std(energy_db)

        # Duration (total time)
        duration = len(audio) / self.sr

        # Vibrato detection (F0 oscillation)
        f0_grad = np.gradient(f0_voiced)
        f0_vibrato = np.std(f0_grad)

        return {
            "f0_mean": float(np.mean(f0_voiced)),
            "f0_std": float(np.std(f0_voiced)),
            "f0_min": float(np.min(f0_voiced)),
            "f0_max": float(np.max(f0_voiced)),
            "f0_range": (float(np.min(f0_voiced)), float(np.max(f0_voiced))),
            "f0_vibrato": float(f0_vibrato),
            "energy_mean": float(energy_mean),
            "energy_std": float(energy_std),
            "duration_sec": float(duration),
            "voiced_ratio": float(np.sum(voiced_flag) / len(voiced_flag)),
        }

    def compare_to_natural_speech(
        self,
        synthesized_prosody: Dict,
        reference_prosody: Dict,
    ) -> float:
        """
        Compare synthesized prosody to natural speech.

        Args:
            synthesized_prosody: Prosody features of synthesized audio
            reference_prosody: Prosody features of reference natural audio

        Returns:
            Prosody naturalness score (0-1)
        """
        # Compute feature-wise similarity
        features = ["f0_mean", "f0_std", "energy_mean", "energy_std"]

        distances = []
        for feature in features:
            if feature in synthesized_prosody and feature in reference_prosody:
                synth_val = synthesized_prosody[feature]
                ref_val = reference_prosody[feature]

                # Normalized distance
                if ref_val != 0:
                    Distance = abs(synth_val - ref_val) / abs(ref_val)
                    distances.append(1 - min(Distance, 1.0))  # Similarity

        if not distances:
            return 0.5  # Default if no features available

        # Average similarity across features
        naturalness = np.mean(distances)
        return float(naturalness)


class MOSProxyEstimator:
    """Estimate MOS (Mean Opinion Score) without human listeners."""

    def __init__(self, device: str = "cuda"):
        """
        Initialize MOS estimator.

        Args:
            device: Device to run on ("cuda" or "cpu")
        """
        self.device = device
        self.prosody_analyzer = ProsodyAnalyzer()

        # Load speaker embedder if available
        if RESEMBLYZER_AVAILABLE:
            try:
                self.voice_encoder = VoiceEncoder(device=device)
                logger.info("Loaded Resemblyzer voice encoder")
            except Exception as e:
                logger.warning(f"Failed to load voice encoder: {e}")
                self.voice_encoder = None
        else:
            self.voice_encoder = None

        logger.info("MOSProxyEstimator initialized")

    def estimate_mos_from_audio(
        self,
        synthesized_audio: np.ndarray,
        reference_audio: Optional[np.ndarray] = None,
        sample_rate: int = 44100,
    ) -> Dict:
        """
        Estimate MOS score from synthesized audio.

        Args:
            synthesized_audio: Synthesized audio waveform
            reference_audio: Optional reference natural audio for comparison
            sample_rate: Sample rate of audio

        Returns:
            Dictionary with MOS estimate and component scores
        """
        scores = {}

        # 1. Prosody quality
        with timer("Prosody analysis"):
            synth_prosody = self.prosody_analyzer.extract_prosody_features(
                synthesized_audio
            )
            prosody_score = self._score_prosody_quality(synth_prosody)
            scores["prosody_score"] = prosody_score

            if reference_audio is not None:
                ref_prosody = self.prosody_analyzer.extract_prosody_features(
                    reference_audio
                )
                naturalness = self.prosody_analyzer.compare_to_natural_speech(
                    synth_prosody, ref_prosody
                )
                scores["prosody_naturalness"] = naturalness

        # 2. Spectral quality
        with timer("Spectral analysis"):
            spectral_score = self._analyze_spectral_quality(synthesized_audio)
            scores["spectral_quality"] = spectral_score

        # 3. Artifact detection
        with timer("Artifact detection"):
            artifacts = self._detect_artifacts(synthesized_audio)
            scores["has_artifacts"] = artifacts["detected"]
            scores["artifact_severity"] = artifacts["severity"]

        # 4. Silence/noise ratio
        with timer("Silence detection"):
            silence_score = self._compute_silence_ratio(synthesized_audio)
            scores["silence_ratio"] = silence_score

        # 5. Overall MOS estimate (weighted combination)
        mos_estimate = self._combine_scores(scores)
        scores["mos_estimate"] = mos_estimate

        # Confidence (higher if multiple signals agree)
        scores["confidence"] = self._compute_confidence(scores)

        logger.info(f"MOS estimate: {mos_estimate:.2f}/5.0")

        return scores

    def _score_prosody_quality(self, prosody: Dict) -> float:
        """
        Score prosody quality (0-1).

        Args:
            prosody: Prosody features dictionary

        Returns:
            Prosody quality score (0-1)
        """
        # Check for reasonable pitch range
        f0_min = prosody.get("f0_min", 0)
        f0_max = prosody.get("f0_max", 500)
        f0_range = f0_max - f0_min

        # Reasonable pitch range for speech: 50-400 Hz
        expected_range = 150  # Hz
        range_score = min(f0_range / expected_range, 1.0)

        # Check for vibrato (some vibrato is natural)
        vibrato = prosody.get("f0_vibrato", 0)
        vibrato_score = min(vibrato / 5.0, 1.0)  # Expect 1-5 Hz vibrato

        # Check voice activity ratio
        voiced_ratio = prosody.get("voiced_ratio", 0)
        # For natural speech, typically 0.4-0.7 voiced
        voiced_score = 1.0 - abs(voiced_ratio - 0.5) / 0.5

        # Energy consistency
        energy_std = prosody.get("energy_std", 0)
        # Lower std = more stable (good), but too low = flat (bad)
        energy_score = 1.0 - min(energy_std / 20.0, 1.0)

        # Combine
        quality = np.mean([range_score, vibrato_score, voiced_score, energy_score])
        return float(quality)

    def _analyze_spectral_quality(self, audio: np.ndarray) -> float:
        """
        Analyze spectral quality using MFCC variance and formant stability.

        Args:
            audio: Audio waveform

        Returns:
            Spectral quality score (0-1)
        """
        # Compute MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)[1:]  # Skip energy

        # Spectral variance (high variance = natural, too low = robotic)
        mfcc_std = np.std(mfcc)
        variance_score = min(mfcc_std / 5.0, 1.0)

        # Spectral continuity (high continuity = natural transitions)
        mfcc_frames = mfcc.T
        frame_diffs = np.diff(mfcc_frames, axis=0)
        continuity = np.mean(np.std(frame_diffs, axis=1))
        continuity_score = max(0, 1.0 - continuity / 10.0)

        # Mel spectrogram entropy (high entropy = varied, low = flat)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000)
        mel_spec_norm = mel_spec / (np.max(mel_spec) + 1e-10)
        mel_spec_flat = mel_spec_norm.flatten()
        mel_entropy = entropy(mel_spec_flat + 1e-10)
        entropy_score = min(mel_entropy / 5.0, 1.0)

        # Combine
        quality = np.mean([variance_score, continuity_score, entropy_score])
        return float(quality)

    def _detect_artifacts(self, audio: np.ndarray) -> Dict:
        """
        Detect TTS artifacts (clicking, distortion, background hum).

        Args:
            audio: Audio waveform

        Returns:
            Dictionary with artifact detection results
        """
        artifacts = {
            "detected": False,
            "artifact_types": [],
            "severity": 0.0,
        }

        # Detect clicks/pops (high frequency spikes)
        # Use high-pass filter to isolate clicks
        sos = signal.butter(4, 4000, "hp", fs=16000, output="sos")
        filtered = signal.sosfilt(sos, audio)
        click_detected = np.any(np.abs(filtered) > 3.0 * np.std(filtered))

        if click_detected:
            artifacts["detected"] = True
            artifacts["artifact_types"].append("clicking")

        # Detect background hum (60 Hz or 50 Hz)
        freqs, spec = signal.periodogram(audio, fs=16000)
        hum_bands = [
            (50, 52),  # 50 Hz mains
            (59, 61),  # 60 Hz mains
            (119, 121),  # Harmonic
        ]

        for f_min, f_max in hum_bands:
            band_power = np.mean(spec[(freqs >= f_min) & (freqs <= f_max)])
            total_power = np.mean(spec)
            if band_power > 0.3 * total_power:
                artifacts["detected"] = True
                artifacts["artifact_types"].append(f"hum_{int(f_min+0.5)}hz")

        # Distortion detection (clipping)
        clipping_ratio = np.sum(np.abs(audio) > 0.95) / len(audio)
        if clipping_ratio > 0.01:
            artifacts["detected"] = True
            artifacts["artifact_types"].append("clipping")
            artifacts["severity"] = clipping_ratio

        return artifacts

    def _compute_silence_ratio(self, audio: np.ndarray, threshold_db: float = -40) -> float:
        """
        Compute ratio of speech to silence.

        Args:
            audio: Audio waveform
            threshold_db: Energy threshold for silence (dB)

        Returns:
            Silence ratio (0-1, 0 = all speech, 1 = all silence)
        """
        # Compute energy
        S = librosa.feature.melspectrogram(y=audio, sr=16000)
        energy = librosa.power_to_db(S, ref=np.max)

        # Frames below threshold are silence
        silence_frames = np.mean(energy < threshold_db)
        return float(silence_frames)

    def _combine_scores(self, component_scores: Dict) -> float:
        """
        Combine component scores into overall MOS estimate.

        Args:
            component_scores: Dictionary of component scores

        Returns:
            MOS estimate (0-5 scale)
        """
        # Weights for each component
        weights = {
            "prosody_score": 0.25,
            "spectral_quality": 0.25,
            "silence_ratio": 0.15,  # Normalized to (0, 1)
            "artifact_severity": -0.2,  # Negative impact
        }

        mos = 2.5  # Baseline

        for component, weight in weights.items():
            if component in component_scores:
                value = component_scores[component]

                if component == "silence_ratio":
                    # Lower silence is better
                    value = 1.0 - value

                mos += weight * (value * 2.5)  # Scale to 0-2.5 contribution

        # Clamp to valid MOS range [1, 5]
        mos = np.clip(mos, 1.0, 5.0)

        return float(mos)

    def _compute_confidence(self, scores: Dict) -> float:
        """
        Compute confidence in MOS estimate (0-1).

        Args:
            scores: Component scores dictionary

        Returns:
            Confidence score (0-1)
        """
        # High confidence if no artifacts detected
        confidence = 1.0 if not scores.get("has_artifacts", False) else 0.7

        # Adjust based on spectral quality
        if "spectral_quality" in scores:
            confidence *= 0.5 + 0.5 * scores["spectral_quality"]

        return float(np.clip(confidence, 0.0, 1.0))

    def emotion_preservation_score(
        self,
        original_emotion: str,
        synthesized_audio: np.ndarray,
    ) -> Dict:
        """
        Measure if emotion tag persisted to output audio.

        Args:
            original_emotion: Intended emotion (<neutral>, <happy>, <sad>, etc.)
            synthesized_audio: Synthesized audio waveform

        Returns:
            Dictionary with emotion preservation metrics
        """
        # Extract prosody features
        prosody = self.prosody_analyzer.extract_prosody_features(synthesized_audio)

        # Map emotion to expected prosody characteristics
        emotion_targets = {
            "neutral": {"f0_range": (80, 200), "energy_std": 3},
            "happy": {"f0_range": (150, 300), "energy_std": 5},
            "sad": {"f0_range": (60, 150), "energy_std": 2},
            "angry": {"f0_range": (100, 250), "energy_std": 6},
            "excited": {"f0_range": (180, 350), "energy_std": 7},
            "calm": {"f0_range": (70, 180), "energy_std": 2},
        }

        target = emotion_targets.get(original_emotion.lower(), emotion_targets["neutral"])

        # Check if prosody matches expected range
        f0_min, f0_max = prosody.get("f0_range", (0, 0))
        target_min, target_max = target["f0_range"]

        # Compute match score (how much of expected range is achieved)
        achieved_range = max(0, min(f0_max, target_max) - max(f0_min, target_min))
        expected_range = target_max - target_min
        range_match = achieved_range / max(1, expected_range)

        # Energy match
        energy_std = prosody.get("energy_std", 0)
        target_energy = target["energy_std"]
        energy_match = 1.0 - min(abs(energy_std - target_energy) / 5, 1.0)

        # Overall emotion preservation
        emotion_score = 0.6 * range_match + 0.4 * energy_match
        emotion_score = np.clip(emotion_score, 0.0, 1.0)

        return {
            "emotion": original_emotion,
            "emotion_preservation_score": float(emotion_score),
            "pitch_range_match": float(range_match),
            "energy_match": float(energy_match),
            "expected_f0_range": target["f0_range"],
            "actual_f0_range": (float(f0_min), float(f0_max)),
        }


    def estimate_from_manifest(self, manifest_path: str) -> Dict:
        """Evaluate MOS on a JSONL manifest of audio files.

        Each line in the manifest should have at least "audio_path".
        Optionally "reference_audio_path" for paired evaluation.

        Args:
            manifest_path: Path to JSONL manifest file.

        Returns:
            Dictionary with aggregated MOS statistics.
        """
        import json as _json

        manifest = Path(manifest_path)
        if not manifest.exists():
            return {"error": f"Manifest not found: {manifest_path}"}

        scores = []
        errors = 0

        with open(manifest) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = _json.loads(line)
                audio_path = sample.get("audio_path", "")
                if not audio_path or not Path(audio_path).exists():
                    errors += 1
                    continue

                try:
                    audio, _sr = load_audio(audio_path)
                    ref_audio = None
                    ref_path = sample.get("reference_audio_path", "")
                    if ref_path and Path(ref_path).exists():
                        ref_audio, _ = load_audio(ref_path)

                    result = self.estimate_mos_from_audio(audio, ref_audio)
                    if "mos_estimate" in result:
                        scores.append(result["mos_estimate"])
                except Exception as e:
                    logger.debug(f"MOS estimation failed for {audio_path}: {e}")
                    errors += 1

        if not scores:
            return {"mos_mean": 0, "n_samples": 0, "errors": errors}

        return {
            "mos_mean": float(np.mean(scores)),
            "mos_std": float(np.std(scores)),
            "mos_min": float(np.min(scores)),
            "mos_max": float(np.max(scores)),
            "n_samples": len(scores),
            "errors": errors,
        }


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mos_proxy_metrics.py <audio_file> [reference_audio]")
        sys.exit(1)

    audio_path = sys.argv[1]
    audio, sr = load_audio(audio_path)

    # Optional reference
    reference_audio = None
    if len(sys.argv) > 2:
        reference_audio, _ = load_audio(sys.argv[2])

    estimator = MOSProxyEstimator()
    results = estimator.estimate_mos_from_audio(audio, reference_audio)

    print("\n=== MOS Estimation Results ===")
    print(f"MOS Estimate: {results['mos_estimate']:.2f}/5.0")
    print(f"Confidence: {results['confidence']:.2f}")
    print(f"Prosody Score: {results['prosody_score']:.2f}")
    print(f"Spectral Quality: {results['spectral_quality']:.2f}")
    print(f"Artifacts Detected: {results['has_artifacts']}")
