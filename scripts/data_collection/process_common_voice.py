#!/usr/bin/env python3
"""
Common Voice hy-AM Download & Processing — Phase 1d

Downloads Mozilla Common Voice Armenian dataset and creates clean manifests
for ASR training. This is the "gold standard" seed data for:
  - LM training (clean Armenian text)
  - ASR fine-tuning
  - Quality reference for YouTube data validation

Usage:
    python scripts/data_collection/process_common_voice.py
    python scripts/data_collection/process_common_voice.py --version 17.0 --output-dir data/common_voice
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger


class CommonVoiceProcessor:
    """Download and process Common Voice hy-AM dataset."""

    def __init__(self, output_dir: Path, version: str = "17.0"):
        self.output_dir = output_dir
        self.version = version
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> bool:
        """Download Common Voice hy-AM using HuggingFace datasets."""
        logger.info("Downloading Common Voice {} hy-AM...", self.version)
        logger.info("This may take a while on first run (several GB).")

        try:
            from datasets import load_dataset

            ds = load_dataset(
                f"mozilla-foundation/common_voice_{self.version.replace('.', '_')}",
                "hy-AM",
                trust_remote_code=True,
                cache_dir=str(self.output_dir / "hf_cache"),
            )

            # Save splits
            for split_name in ds:
                split_dir = self.output_dir / split_name
                ds[split_name].save_to_disk(str(split_dir))
                logger.info("  {}: {} examples", split_name, len(ds[split_name]))

            return True

        except Exception as e:
            logger.error("Download failed: {}", e)
            logger.info("Alternative: download manually from https://commonvoice.mozilla.org/hy-AM/datasets")
            logger.info("Then extract to {}", self.output_dir)
            return False

    def process_split(self, split_name: str) -> list[dict]:
        """Process a single split into training manifest format."""
        split_dir = self.output_dir / split_name

        try:
            from datasets import load_from_disk
            ds = load_from_disk(str(split_dir))
        except Exception:
            logger.warning("Cannot load split {} from disk, trying TSV", split_name)
            return self._process_from_tsv(split_name)

        entries = []
        audio_out_dir = self.output_dir / "audio" / split_name
        audio_out_dir.mkdir(parents=True, exist_ok=True)

        for idx, example in enumerate(tqdm(ds, desc=f"Processing {split_name}")):
            try:
                # Extract audio
                audio = example.get("audio", {})
                sentence = example.get("sentence", "").strip()

                if not sentence:
                    continue

                # Get audio data
                if isinstance(audio, dict):
                    audio_array = np.array(audio["array"], dtype=np.float32)
                    sr = audio["sampling_rate"]
                else:
                    continue

                # Save as WAV at 16kHz
                import soundfile as sf
                import librosa

                if sr != 16000:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                    sr = 16000

                duration = len(audio_array) / sr

                # Skip very short/long
                if duration < 0.5 or duration > 30:
                    continue

                # Generate filename
                clip_id = example.get("path", example.get("client_id", f"cv_{idx:07d}"))
                if isinstance(clip_id, str):
                    clip_id = Path(clip_id).stem
                else:
                    clip_id = f"cv_{idx:07d}"

                audio_path = audio_out_dir / f"{clip_id}.wav"
                sf.write(str(audio_path), audio_array, sr)

                entry = {
                    "audio_path": str(audio_path),
                    "text": sentence,
                    "duration_sec": round(duration, 3),
                    "sample_rate": sr,
                    "split": split_name,
                    "source": "common_voice",
                    "locale": example.get("locale", "hy-AM"),
                    "gender": example.get("gender", ""),
                    "age": example.get("age", ""),
                    "up_votes": example.get("up_votes", 0),
                    "down_votes": example.get("down_votes", 0),
                }
                entries.append(entry)

            except Exception as e:
                logger.debug("Error processing example {}: {}", idx, e)
                continue

        logger.info("{}: {} valid entries ({:.1f} hours)",
                    split_name, len(entries),
                    sum(e["duration_sec"] for e in entries) / 3600)
        return entries

    def _process_from_tsv(self, split_name: str) -> list[dict]:
        """Fallback: process from extracted TSV + clips directory."""
        tsv_path = self.output_dir / f"{split_name}.tsv"
        clips_dir = self.output_dir / "clips"

        if not tsv_path.exists():
            logger.warning("TSV not found: {}", tsv_path)
            return []

        import csv
        entries = []

        with open(tsv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in tqdm(reader, desc=f"Processing {split_name} (TSV)"):
                clip_path = clips_dir / row.get("path", "")
                sentence = row.get("sentence", "").strip()

                if not clip_path.exists() or not sentence:
                    continue

                try:
                    import soundfile as sf
                    info = sf.info(str(clip_path))
                    duration = info.duration

                    if duration < 0.5 or duration > 30:
                        continue

                    entries.append({
                        "audio_path": str(clip_path),
                        "text": sentence,
                        "duration_sec": round(duration, 3),
                        "sample_rate": info.samplerate,
                        "split": split_name,
                        "source": "common_voice",
                        "gender": row.get("gender", ""),
                        "age": row.get("age", ""),
                        "up_votes": int(row.get("up_votes", 0)),
                        "down_votes": int(row.get("down_votes", 0)),
                    })
                except Exception:
                    continue

        return entries

    def extract_lm_corpus(self, entries: list[dict], output_path: Path):
        """Extract clean text corpus for LM training."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        texts = set()
        for e in entries:
            text = e.get("text", "").strip()
            if text and len(text) > 3:
                texts.add(text)

        with open(output_path, "w", encoding="utf-8") as f:
            for text in sorted(texts):
                f.write(text + "\n")

        logger.info("LM corpus: {} unique sentences -> {}", len(texts), output_path)

    def run(self) -> dict:
        """Full processing pipeline. Returns stats."""
        # Step 1: Download (if needed)
        has_data = any((self.output_dir / s).exists() for s in ["train", "test", "validation", "validated"])
        if not has_data:
            self.download()

        # Step 2: Process each split
        all_entries = {}
        splits_to_process = ["train", "validation", "test", "other", "validated", "invalidated"]

        for split in splits_to_process:
            split_dir = self.output_dir / split
            if split_dir.exists():
                entries = self.process_split(split)
                if entries:
                    all_entries[split] = entries

        # Step 3: Write manifests
        manifest_dir = self.output_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        stats = {}
        all_texts = []

        for split, entries in all_entries.items():
            manifest_path = manifest_dir / f"{split}.jsonl"
            with open(manifest_path, "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            total_hours = sum(e["duration_sec"] for e in entries) / 3600
            stats[split] = {
                "count": len(entries),
                "hours": round(total_hours, 2),
            }
            all_texts.extend(entries)

            logger.info("  {}: {} entries, {:.1f}h -> {}", split, len(entries), total_hours, manifest_path)

        # Step 4: Extract LM corpus
        self.extract_lm_corpus(all_texts, self.output_dir / "lm_corpus.txt")

        # Step 5: Save stats
        with open(self.output_dir / "processing_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        total = sum(s["count"] for s in stats.values())
        total_h = sum(s["hours"] for s in stats.values())
        logger.info("Common Voice processing complete: {} entries, {:.1f} hours", total, total_h)

        return stats


def main():
    parser = argparse.ArgumentParser(description="Common Voice hy-AM Processor")
    parser.add_argument("--output-dir", default="data/common_voice")
    parser.add_argument("--version", default="17.0", help="Common Voice version (e.g., 17.0)")

    args = parser.parse_args()
    setup_logger()

    processor = CommonVoiceProcessor(Path(args.output_dir), args.version)
    processor.run()


if __name__ == "__main__":
    main()
