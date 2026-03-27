#!/usr/bin/env python3
"""
Training utilities for ASR, TTS, and Translation fine-tuning.

Provides:
  - Data loading & preprocessing
  - Custom collators
  - Metric computation
  - Checkpoint management
  - Learning curve tracking
"""

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger


# ============================================================================
# Audio preprocessing
# ============================================================================

class AudioPreprocessor:
    """Prepare audio for ASR/TTS training."""

    def __init__(self, sample_rate: int = 16000, max_sec: float = 30.0):
        self.sample_rate = sample_rate
        self.max_samples = int(max_sec * sample_rate)

    def load_and_preprocess(self, audio_path: str) -> dict:
        """Load audio and return array + duration."""
        import librosa
        import soundfile as sf

        try:
            audio, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                mono=True,
                dtype=np.float32,
            )
        except Exception:
            # Fallback: soundfile
            audio, sr = sf.read(audio_path, dtype=np.float32)
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

        # Clip to max length
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]

        return {
            "input_values": audio,
            "input_length": len(audio),
            "duration_sec": len(audio) / self.sample_rate,
        }


# ============================================================================
# ASR Data Collators (Whisper)
# ============================================================================

@dataclass
class DataCollatorASRWithPadding:
    """Pad audio and text for ASR training (Whisper)."""

    processor: object  # whisper_processor
    sample_rate: int = 16000
    max_audio_length: float = 30.0

    def __call__(self, batch: list) -> dict:
        """Collate batch of samples."""
        audio_preprocessor = AudioPreprocessor(self.sample_rate, self.max_audio_length)

        # Load audio
        input_features = []
        attention_mask = []
        labels = []

        for sample in batch:
            # Load audio
            audio_dict = audio_preprocessor.load_and_preprocess(sample["audio_path"])
            audio_array = audio_dict["input_values"]

            # Process with feature extractor
            feature = self.processor(
                audio_array,
                sampling_rate=self.sample_rate,
                return_attention_mask=True,
            )

            input_features.append(feature.input_features[0])
            attention_mask.append(feature.attention_mask[0])

            # Tokenize text
            text = sample.get("text", sample.get("text_clean", ""))
            token_ids = self.processor.tokenizer.encode(text)
            labels.append(token_ids)

        # Pad input features
        input_features = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        # Pad labels (use -100 for padding, ignored in loss)
        max_label_len = max(len(l) for l in labels)
        padded_labels = []
        for label_ids in labels:
            padded = label_ids + [-100] * (max_label_len - len(label_ids))
            padded_labels.append(padded[:max_label_len])

        return {
            "input_features": input_features.input_features,
            "attention_mask": input_features.attention_mask,
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


# ============================================================================
# TTS Data Collators (Fish-Speech)
# ============================================================================

@dataclass
class DataCollatorTTSWithPadding:
    """Collate TTS training data (Fish-Speech)."""

    tokenizer: object  # Text tokenizer
    sample_rate: int = 44100
    max_text_length: int = 500
    max_audio_length: float = 30.0

    def __call__(self, batch: list) -> dict:
        """Collate batch."""
        audio_preprocessor = AudioPreprocessor(self.sample_rate, self.max_audio_length)

        texts = []
        audio_arrays = []
        input_lengths = []
        label_lengths = []
        emotion_tags = []

        for sample in batch:
            # Text
            text = sample.get("text", sample.get("text_clean", ""))
            # Add emotion tag if available
            emotion = sample.get("emotion", "neutral")
            if emotion and emotion != "neutral":
                text = f"<{emotion}> {text}"
            texts.append(text)

            # Audio
            audio_dict = audio_preprocessor.load_and_preprocess(sample["audio_path"])
            audio_arrays.append(audio_dict["input_values"])
            input_lengths.append(audio_dict["input_length"])
            emotion_tags.append(emotion)

        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Pad audio
        max_audio_len = max(input_lengths)
        padded_audio = []
        for audio in audio_arrays:
            padded = np.pad(
                audio,
                (0, max_audio_len - len(audio)),
                mode="constant",
                constant_values=0,
            )
            padded_audio.append(padded)

        return {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask,
            "audio": torch.tensor(np.array(padded_audio), dtype=torch.float32),
            "input_length": torch.tensor(input_lengths, dtype=torch.long),
            "emotion_tags": emotion_tags,
        }


# ============================================================================
# Metrics
# ============================================================================

class MetricsComputer:
    """Compute training metrics."""

    @staticmethod
    def compute_wer(predictions: list[str], references: list[str]) -> float:
        """Compute Word Error Rate."""
        try:
            from jiwer import wer
            return wer(references, predictions)
        except Exception:
            logger.warning("jiwer not available; returning 0 WER")
            return 0.0

    @staticmethod
    def compute_cer(predictions: list[str], references: list[str]) -> float:
        """Compute Character Error Rate."""
        try:
            from jiwer import cer
            return cer(references, predictions)
        except Exception:
            logger.warning("jiwer not available; returning 0 CER")
            return 0.0

    @staticmethod
    def compute_speaker_similarity(
        reference_speaker_embedding: np.ndarray,
        synthesized_speaker_embedding: np.ndarray,
    ) -> float:
        """Compute cosine similarity (0-1, higher = better match)."""
        dot = np.dot(reference_speaker_embedding, synthesized_speaker_embedding)
        norm_a = np.linalg.norm(reference_speaker_embedding)
        norm_b = np.linalg.norm(synthesized_speaker_embedding)
        if norm_a < 1e-7 or norm_b < 1e-7:
            return 0.0
        return float(dot / (norm_a * norm_b))

    @staticmethod
    def compute_pesq(reference: np.ndarray, degraded: np.ndarray, sr: int = 16000) -> float:
        """Compute PESQ (Perceptual Evaluation of Speech Quality)."""
        try:
            from pesq import pesq
            return pesq(sr, reference, degraded, mode="wb")
        except Exception:
            logger.warning("PESQ unavailable; returning 0")
            return 0.0


# ============================================================================
# Checkpoint Management
# ============================================================================

class CheckpointManager:
    """Manage training checkpoints."""

    def __init__(
        self,
        save_dir: Path,
        keep_best: int = 3,
        metric_name: str = "eval_wer",
        mode: str = "min",  # min for WER, max for accuracy
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self.metric_name = metric_name
        self.mode = mode
        self.history: list[tuple[float, Path]] = []

    def save(
        self,
        model: object,
        optimizer: object,
        epoch: int,
        metrics: dict,
        is_best: bool = False,
    ):
        """Save checkpoint with metadata."""
        metric_value = metrics.get(self.metric_name, 0)
        suffix = f"_best" if is_best else f"_ep{epoch}"
        ckpt_dir = self.save_dir / f"checkpoint{suffix}"
        ckpt_dir.mkdir(exist_ok=True)

        # Save model
        model.save_pretrained(str(ckpt_dir / "model"))

        # Save optimizer state
        torch.save(optimizer.state_dict(), str(ckpt_dir / "optimizer.pt"))

        # Save metadata
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(
                {
                    "epoch": epoch,
                    "metrics": metrics,
                    "metric_value": metric_value,
                },
                f,
                indent=2,
            )

        logger.info(
            "Saved checkpoint: {} ({}={:.4f})",
            ckpt_dir.name,
            self.metric_name,
            metric_value,
        )

        # Track for pruning
        self.history.append((metric_value, ckpt_dir))
        self._prune()

    def _prune(self):
        """Keep only top-k checkpoints."""
        # Sort by metric value
        if self.mode == "min":
            self.history.sort(key=lambda x: x[0])  # Lower is better
        else:
            self.history.sort(key=lambda x: -x[0])  # Higher is better

        # Remove old ones
        for _, ckpt_dir in self.history[self.keep_best:]:
            if ckpt_dir.exists():
                logger.debug("Removing old checkpoint: {}", ckpt_dir.name)
                shutil.rmtree(ckpt_dir)

        self.history = self.history[:self.keep_best]

    def get_best(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        if not self.history:
            return None
        return self.history[0][1]  # Already sorted


# ============================================================================
# Learning Rate Scheduling
# ============================================================================

def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
):
    """Linear warmup then linear decay."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# ============================================================================
# Manifest to Dataset conversion
# ============================================================================

def load_jsonl_manifest(manifest_path: Path) -> list[dict]:
    """Load JSONL manifest file."""
    entries = []
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    return entries


def manifest_to_hf_dataset(manifest_path: Path, split_name: str = "train"):
    """Convert JSONL manifest to HuggingFace dataset."""
    from datasets import Dataset

    entries = load_jsonl_manifest(manifest_path)
    dataset = Dataset.from_dict({key: [e[key] for e in entries] for key in entries[0].keys()})
    dataset.info.splits = {split_name}
    return dataset


# ============================================================================
# Progress tracker
# ============================================================================

class TrainingProgressTracker:
    """Track and log training progress."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            "train_loss": [],
            "eval_loss": [],
            "eval_metrics": [],
            "learning_rate": [],
        }

    def log_batch(self, loss: float, learning_rate: float):
        """Log batch-level training."""
        self.history["train_loss"].append(loss)
        self.history["learning_rate"].append(learning_rate)

    def log_eval(self, loss: float, metrics: dict):
        """Log evaluation."""
        self.history["eval_loss"].append(loss)
        self.history["eval_metrics"].append(metrics)

    def save(self):
        """Save history to JSON."""
        with open(self.output_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

    def plot(self):
        """Plot learning curves (if matplotlib available)."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Loss
            axes[0].plot(self.history["train_loss"], label="Train")
            axes[0].plot(self.history["eval_loss"], label="Eval")
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title("Loss Curves")

            # Learning rate
            axes[1].plot(self.history["learning_rate"])
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("Learning Rate")
            axes[1].grid(True, alpha=0.3)
            axes[1].set_title("Learning Rate Schedule")

            fig.tight_layout()
            fig.savefig(self.output_dir / "learning_curves.png", dpi=100)
            logger.info("Saved learning curves to {}", self.output_dir / "learning_curves.png")

        except ImportError:
            logger.debug("matplotlib not available for plotting")
