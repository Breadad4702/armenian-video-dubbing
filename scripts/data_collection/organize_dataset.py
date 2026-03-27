#!/usr/bin/env python3
"""
Dataset Organizer — Train/Val/Test Splits & Unified Manifests — Phase 1e

Merges data from all sources into unified manifests:
  - Common Voice hy-AM (gold standard)
  - YouTube crawl (gold/silver/bronze tiers)
  - Validated annotations from Label Studio
  - Optional: TTS studio recordings

Creates:
  - data/splits/train.jsonl  (ASR training)
  - data/splits/val.jsonl    (validation)
  - data/splits/test.jsonl   (held-out test)
  - data/splits/tts_train.jsonl  (TTS fine-tuning subset)
  - data/splits/stats.json

Usage:
    python scripts/data_collection/organize_dataset.py
    python scripts/data_collection/organize_dataset.py --cv-dir data/common_voice --yt-dir data/youtube_crawl
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger


class DatasetOrganizer:
    """Merge all data sources and create train/val/test splits."""

    def __init__(
        self,
        output_dir: Path,
        cv_dir: Path | None = None,
        yt_dir: Path | None = None,
        studio_dir: Path | None = None,
        validated_path: Path | None = None,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        seed: int = 42,
    ):
        self.output_dir = output_dir
        self.cv_dir = cv_dir
        self.yt_dir = yt_dir
        self.studio_dir = studio_dir
        self.validated_path = validated_path
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_jsonl(self, path: Path) -> list[dict]:
        """Load entries from JSONL file."""
        entries = []
        if not path.exists():
            return entries
        with open(path) as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return entries

    def _normalize_entry(self, entry: dict, source: str) -> dict | None:
        """Normalize entry to unified format."""
        audio_path = entry.get("audio_path", "")
        text = entry.get("text", "") or entry.get("validated_text", "") or entry.get("transcription", {}).get("text_clean", "")

        if not audio_path or not text:
            return None

        # Verify audio file exists
        if not Path(audio_path).exists():
            return None

        duration = entry.get("duration_sec", 0)
        if duration <= 0:
            try:
                import soundfile as sf
                info = sf.info(audio_path)
                duration = info.duration
            except Exception:
                return None

        # Deterministic ID from audio path
        entry_id = hashlib.md5(audio_path.encode()).hexdigest()[:12]

        return {
            "id": entry_id,
            "audio_path": audio_path,
            "text": text.strip(),
            "duration_sec": round(duration, 3),
            "source": source,
            "quality_tier": entry.get("quality_tier", "gold" if source == "common_voice" else "unknown"),
            "speaker_id": entry.get("speaker_id", entry.get("client_id", "")),
            "gender": entry.get("gender", ""),
            "snr_db": entry.get("snr_db", 0),
        }

    def load_common_voice(self) -> list[dict]:
        """Load Common Voice data."""
        entries = []
        if not self.cv_dir:
            return entries

        manifest_dir = self.cv_dir / "manifests"
        if not manifest_dir.exists():
            logger.warning("No Common Voice manifests at {}", manifest_dir)
            return entries

        for split_file in sorted(manifest_dir.glob("*.jsonl")):
            split_entries = self._load_jsonl(split_file)
            for e in split_entries:
                norm = self._normalize_entry(e, "common_voice")
                if norm:
                    norm["cv_split"] = split_file.stem
                    entries.append(norm)

        logger.info("Common Voice: {} entries, {:.1f}h",
                    len(entries), sum(e["duration_sec"] for e in entries) / 3600)
        return entries

    def load_youtube(self) -> list[dict]:
        """Load YouTube crawl data (quality-bucketed)."""
        entries = []
        if not self.yt_dir:
            return entries

        quality_dir = self.yt_dir / "quality_buckets"
        if not quality_dir.exists():
            logger.warning("No YouTube quality buckets at {}", quality_dir)
            return entries

        for tier in ["gold", "silver", "bronze"]:
            tier_file = quality_dir / f"{tier}.jsonl"
            if not tier_file.exists():
                continue

            tier_entries = self._load_jsonl(tier_file)
            for e in tier_entries:
                norm = self._normalize_entry(e, f"youtube_{tier}")
                if norm:
                    entries.append(norm)

        logger.info("YouTube: {} entries, {:.1f}h",
                    len(entries), sum(e["duration_sec"] for e in entries) / 3600)
        return entries

    def load_validated(self) -> list[dict]:
        """Load human-validated annotations from Label Studio."""
        entries = []
        if not self.validated_path or not self.validated_path.exists():
            return entries

        raw = self._load_jsonl(self.validated_path)
        for e in raw:
            # Only use correct/minor_errors
            quality = e.get("quality_label", "")
            if quality not in ("correct", "minor_errors"):
                continue

            norm = self._normalize_entry(e, "validated")
            if norm:
                norm["quality_tier"] = "gold"  # Human-validated = gold
                entries.append(norm)

        logger.info("Validated: {} entries, {:.1f}h",
                    len(entries), sum(e["duration_sec"] for e in entries) / 3600)
        return entries

    def load_studio(self) -> list[dict]:
        """Load studio TTS recordings."""
        entries = []
        if not self.studio_dir or not self.studio_dir.exists():
            return entries

        for manifest in self.studio_dir.glob("*.jsonl"):
            raw = self._load_jsonl(manifest)
            for e in raw:
                norm = self._normalize_entry(e, "studio")
                if norm:
                    norm["quality_tier"] = "gold"
                    entries.append(norm)

        logger.info("Studio: {} entries, {:.1f}h",
                    len(entries), sum(e["duration_sec"] for e in entries) / 3600)
        return entries

    def deduplicate(self, entries: list[dict]) -> list[dict]:
        """Remove duplicate entries by audio path."""
        seen = set()
        unique = []
        for e in entries:
            key = e["audio_path"]
            if key not in seen:
                seen.add(key)
                unique.append(e)
        removed = len(entries) - len(unique)
        if removed > 0:
            logger.info("Deduplicated: removed {} duplicates", removed)
        return unique

    def split_data(self, entries: list[dict]) -> dict[str, list[dict]]:
        """Split into train/val/test ensuring speaker separation."""
        random.seed(self.seed)

        # Group by speaker (or video_id for YouTube)
        speaker_groups = defaultdict(list)
        for e in entries:
            speaker = e.get("speaker_id") or e.get("id", "")[:8]
            speaker_groups[speaker].append(e)

        speakers = list(speaker_groups.keys())
        random.shuffle(speakers)

        n_val = max(1, int(len(speakers) * self.val_ratio))
        n_test = max(1, int(len(speakers) * self.test_ratio))

        test_speakers = set(speakers[:n_test])
        val_speakers = set(speakers[n_test:n_test + n_val])
        train_speakers = set(speakers[n_test + n_val:])

        # SPECIAL: Common Voice test set kept separate for benchmark comparisons
        splits = {"train": [], "val": [], "test": []}

        for e in entries:
            # Respect original CV test/validation split
            if e.get("source") == "common_voice":
                cv_split = e.get("cv_split", "train")
                if cv_split == "test":
                    splits["test"].append(e)
                    continue
                elif cv_split == "validation":
                    splits["val"].append(e)
                    continue

            speaker = e.get("speaker_id") or e.get("id", "")[:8]
            if speaker in test_speakers:
                splits["test"].append(e)
            elif speaker in val_speakers:
                splits["val"].append(e)
            else:
                splits["train"].append(e)

        return splits

    def create_tts_subset(self, entries: list[dict]) -> list[dict]:
        """Create high-quality subset for TTS training.

        Criteria: gold tier, SNR > 20dB, 2-15s duration, clean text.
        """
        tts_entries = []
        for e in entries:
            if e.get("quality_tier") != "gold":
                continue
            if e.get("snr_db", 0) < 20:
                continue
            if not (2.0 <= e["duration_sec"] <= 15.0):
                continue
            if len(e["text"].split()) < 3:
                continue
            tts_entries.append(e)

        logger.info("TTS subset: {} entries ({:.1f}h) from {} total",
                    len(tts_entries),
                    sum(e["duration_sec"] for e in tts_entries) / 3600,
                    len(entries))
        return tts_entries

    def run(self) -> dict:
        """Execute full organization pipeline."""
        logger.info("=" * 60)
        logger.info("Dataset Organization Pipeline")
        logger.info("=" * 60)

        # Load all sources
        all_entries = []
        all_entries.extend(self.load_common_voice())
        all_entries.extend(self.load_youtube())
        all_entries.extend(self.load_validated())
        all_entries.extend(self.load_studio())

        if not all_entries:
            logger.error("No data loaded! Check your data directories.")
            return {}

        # Deduplicate
        all_entries = self.deduplicate(all_entries)

        total_hours = sum(e["duration_sec"] for e in all_entries) / 3600
        logger.info("Total data: {} entries, {:.1f} hours", len(all_entries), total_hours)

        # Split
        splits = self.split_data(all_entries)

        # Write split manifests
        stats = {}
        for split_name, entries in splits.items():
            random.shuffle(entries)
            manifest_path = self.output_dir / f"{split_name}.jsonl"

            with open(manifest_path, "w") as f:
                for e in entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")

            hours = sum(e["duration_sec"] for e in entries) / 3600
            stats[split_name] = {
                "count": len(entries),
                "hours": round(hours, 2),
                "sources": dict(Counter(e["source"] for e in entries)),
                "quality_tiers": dict(Counter(e["quality_tier"] for e in entries)),
            }
            logger.info("  {}: {} entries, {:.1f}h", split_name, len(entries), hours)

        # TTS subset
        tts_entries = self.create_tts_subset(splits["train"])
        tts_path = self.output_dir / "tts_train.jsonl"
        with open(tts_path, "w") as f:
            for e in tts_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        tts_hours = sum(e["duration_sec"] for e in tts_entries) / 3600
        stats["tts_train"] = {"count": len(tts_entries), "hours": round(tts_hours, 2)}

        # Save stats
        stats_path = self.output_dir / "dataset_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info("\nDataset organization complete!")
        logger.info("Stats saved to {}", stats_path)
        return stats


def main():
    parser = argparse.ArgumentParser(description="Dataset Organizer")
    parser.add_argument("--cv-dir", default="data/common_voice")
    parser.add_argument("--yt-dir", default="data/youtube_crawl")
    parser.add_argument("--studio-dir", default="data/tts_studio")
    parser.add_argument("--validated", default="data/youtube_crawl/validated_annotations.jsonl")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    setup_logger()

    organizer = DatasetOrganizer(
        output_dir=Path(args.output_dir),
        cv_dir=Path(args.cv_dir) if args.cv_dir else None,
        yt_dir=Path(args.yt_dir) if args.yt_dir else None,
        studio_dir=Path(args.studio_dir) if args.studio_dir else None,
        validated_path=Path(args.validated) if args.validated else None,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    organizer.run()


if __name__ == "__main__":
    main()
