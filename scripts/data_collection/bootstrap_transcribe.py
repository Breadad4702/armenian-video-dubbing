#!/usr/bin/env python3
"""
Bootstrap ASR Transcription + Statistical LM Refinement — Phase 1b

Implements the core pipeline from "Scaling Armenian ASR" (Hakobyan et al., CSIT 2025):
  1. Bootstrap transcription using Whisper large-v3 (pre-trained)
  2. Language identification to confirm Armenian
  3. Statistical Language Model scoring to filter hallucinations
  4. Confidence-based quality bucketing (gold / silver / bronze)

Usage:
    python scripts/data_collection/bootstrap_transcribe.py --input data/youtube_crawl/segments_filtered.jsonl
    python scripts/data_collection/bootstrap_transcribe.py --phase transcribe
    python scripts/data_collection/bootstrap_transcribe.py --phase langid
    python scripts/data_collection/bootstrap_transcribe.py --phase lm_filter
    python scripts/data_collection/bootstrap_transcribe.py --phase all
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger


# ============================================================================
# Armenian Text Utilities
# ============================================================================

# Armenian Unicode block: U+0530 – U+058F
ARMENIAN_RANGE = range(0x0530, 0x0590)
ARMENIAN_PUNCT = set("։՝՜՞՟«»—")

def armenian_char_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are Armenian."""
    if not text:
        return 0.0
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    arm_chars = [c for c in alpha_chars if ord(c) in ARMENIAN_RANGE]
    return len(arm_chars) / len(alpha_chars)


def clean_armenian_text(text: str) -> str:
    """Clean and normalize Armenian transcript text."""
    # Remove non-Armenian, non-punctuation noise
    text = text.strip()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove isolated single characters that aren't Armenian
    words = text.split()
    cleaned = []
    for w in words:
        if len(w) == 1 and not (ord(w) in ARMENIAN_RANGE or w in ",.!?։՝"):
            continue
        cleaned.append(w)

    return " ".join(cleaned)


def is_valid_armenian(text: str, min_ratio: float = 0.7, min_words: int = 2) -> bool:
    """Check if text is valid Armenian."""
    if not text or len(text.strip()) < 3:
        return False
    words = text.strip().split()
    if len(words) < min_words:
        return False
    return armenian_char_ratio(text) >= min_ratio


# ============================================================================
# Phase 1: Bootstrap Transcription
# ============================================================================

class BootstrapTranscriber:
    """Transcribe audio segments using Whisper large-v3."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        batch_size: int = 16,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.model = None

    def _load_model(self):
        """Lazy-load Whisper model."""
        if self.model is not None:
            return

        logger.info("Loading Whisper {} (faster-whisper, {})...", self.model_size, self.compute_type)
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("Whisper model loaded")

    def transcribe_segment(self, audio_path: str) -> dict:
        """Transcribe a single audio segment.

        Returns:
            Dict with keys: text, language, language_prob, segments, avg_logprob, no_speech_prob
        """
        self._load_model()

        try:
            segments_iter, info = self.model.transcribe(
                audio_path,
                language="hy",
                task="transcribe",
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=200,
                ),
            )

            segments = []
            full_text = []
            total_logprob = 0.0
            total_tokens = 0
            max_no_speech = 0.0

            for seg in segments_iter:
                segments.append({
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text.strip(),
                    "avg_logprob": round(seg.avg_logprob, 4),
                    "no_speech_prob": round(seg.no_speech_prob, 4),
                    "words": [
                        {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3), "probability": round(w.probability, 4)}
                        for w in (seg.words or [])
                    ],
                })
                full_text.append(seg.text.strip())
                total_logprob += seg.avg_logprob * len(seg.words or [1])
                total_tokens += max(len(seg.words or []), 1)
                max_no_speech = max(max_no_speech, seg.no_speech_prob)

            text = " ".join(full_text)
            avg_logprob = total_logprob / max(total_tokens, 1)

            return {
                "text": text,
                "language": info.language,
                "language_prob": round(info.language_probability, 4),
                "segments": segments,
                "avg_logprob": round(avg_logprob, 4),
                "no_speech_prob": round(max_no_speech, 4),
                "duration": round(info.duration, 3),
            }

        except Exception as e:
            logger.error("Transcription error for {}: {}", audio_path, e)
            return {
                "text": "",
                "language": "",
                "language_prob": 0.0,
                "segments": [],
                "avg_logprob": -99.0,
                "no_speech_prob": 1.0,
                "duration": 0.0,
                "error": str(e),
            }

    def run(self, manifest_path: Path, output_path: Path) -> int:
        """Transcribe all segments in manifest. Returns count."""
        segments = []
        with open(manifest_path) as f:
            for line in f:
                try:
                    segments.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        logger.info("Transcribing {} segments with Whisper {}...", len(segments), self.model_size)

        count = 0
        with open(output_path, "w") as out:
            for seg in tqdm(segments, desc="Transcribing"):
                result = self.transcribe_segment(seg["audio_path"])

                # Merge segment metadata with transcription result
                entry = {**seg, "transcription": result}
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

                # Periodic GPU cleanup
                if count % 500 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

        logger.info("Transcription complete: {} segments", count)
        return count


# ============================================================================
# Phase 2: Language Identification Filter
# ============================================================================

class LanguageFilter:
    """Filter transcriptions by Armenian language confidence."""

    def __init__(
        self,
        min_lang_prob: float = 0.7,
        min_armenian_ratio: float = 0.7,
        min_words: int = 2,
    ):
        self.min_lang_prob = min_lang_prob
        self.min_armenian_ratio = min_armenian_ratio
        self.min_words = min_words

    def run(self, input_path: Path, output_path: Path) -> tuple[int, int]:
        """Filter by language. Returns (kept, removed)."""
        kept = 0
        removed = 0

        with open(input_path) as fin, open(output_path, "w") as fout:
            for line in tqdm(fin, desc="Language filtering"):
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                tx = entry.get("transcription", {})
                text = tx.get("text", "")
                lang_prob = tx.get("language_prob", 0)
                detected_lang = tx.get("language", "")

                # Must be detected as Armenian
                if detected_lang != "hy" and lang_prob < self.min_lang_prob:
                    removed += 1
                    continue

                # Must contain Armenian characters
                if not is_valid_armenian(text, self.min_armenian_ratio, self.min_words):
                    removed += 1
                    continue

                # Clean text
                tx["text_clean"] = clean_armenian_text(text)
                entry["transcription"] = tx

                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                kept += 1

        logger.info("Language filter: kept {} ({:.1f}%), removed {}", kept, 100 * kept / max(kept + removed, 1), removed)
        return kept, removed


# ============================================================================
# Phase 3: Statistical LM Scoring & Filtering
# ============================================================================

class LMScorer:
    """
    Statistical n-gram language model for filtering Whisper hallucinations.

    Builds a character-level and word-level n-gram model from known good transcriptions
    (Common Voice), then scores YouTube transcriptions to detect:
    - Repeated phrases (hallucination loops)
    - Non-Armenian text
    - Low-confidence gibberish
    """

    def __init__(self, n: int = 3):
        self.n = n
        self.word_counts: Counter = Counter()
        self.ngram_counts: Counter = Counter()
        self.total_ngrams = 0
        self.vocab_size = 0
        self._trained = False

    def train(self, texts: list[str]):
        """Train n-gram model on known-good Armenian texts."""
        logger.info("Training {}-gram LM on {} texts...", self.n, len(texts))

        for text in texts:
            words = text.strip().split()
            self.word_counts.update(words)

            # Character n-grams
            for i in range(len(text) - self.n + 1):
                ngram = text[i:i + self.n]
                self.ngram_counts[ngram] += 1
                self.total_ngrams += 1

        self.vocab_size = len(self.ngram_counts)
        self._trained = True
        logger.info("LM trained: {} unique {}-grams, {} words", self.vocab_size, self.n, len(self.word_counts))

    def train_from_file(self, path: Path):
        """Train from a text file (one sentence per line) or JSONL manifest."""
        texts = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Try JSON
                try:
                    d = json.loads(line)
                    text = d.get("sentence", d.get("text", d.get("transcription", {}).get("text", "")))
                    if text:
                        texts.append(text)
                    continue
                except json.JSONDecodeError:
                    pass
                # Plain text
                texts.append(line)

        self.train(texts)

    def score(self, text: str) -> float:
        """Score text using character-level perplexity. Lower = more natural."""
        if not self._trained or not text:
            return float("inf")

        log_prob = 0.0
        count = 0

        for i in range(len(text) - self.n + 1):
            ngram = text[i:i + self.n]
            freq = self.ngram_counts.get(ngram, 0)
            # Laplace smoothing
            prob = (freq + 1) / (self.total_ngrams + self.vocab_size)
            log_prob += math.log(prob)
            count += 1

        if count == 0:
            return float("inf")

        # Perplexity
        avg_log_prob = log_prob / count
        perplexity = math.exp(-avg_log_prob)
        return perplexity

    def detect_repetition(self, text: str, max_repeat: int = 3) -> bool:
        """Detect repeated phrases (hallucination loops)."""
        words = text.split()
        if len(words) < 6:
            return False

        # Check for repeated 2-5 word patterns
        for pattern_len in range(2, 6):
            for i in range(len(words) - pattern_len * max_repeat):
                pattern = tuple(words[i:i + pattern_len])
                repeats = 0
                for j in range(i + pattern_len, len(words) - pattern_len + 1, pattern_len):
                    if tuple(words[j:j + pattern_len]) == pattern:
                        repeats += 1
                    else:
                        break
                if repeats >= max_repeat:
                    return True
        return False

    def run(
        self,
        input_path: Path,
        output_path: Path,
        max_perplexity: float = 500.0,
    ) -> tuple[int, int]:
        """Filter by LM score. Returns (kept, removed)."""
        kept = 0
        removed = 0

        with open(input_path) as fin, open(output_path, "w") as fout:
            for line in tqdm(fin, desc="LM filtering"):
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                text = entry.get("transcription", {}).get("text_clean", "")
                if not text:
                    removed += 1
                    continue

                # Check repetition
                if self.detect_repetition(text):
                    removed += 1
                    continue

                # Score
                ppl = self.score(text)
                entry["transcription"]["lm_perplexity"] = round(ppl, 2)

                if ppl > max_perplexity:
                    removed += 1
                    continue

                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                kept += 1

        logger.info("LM filter: kept {} ({:.1f}%), removed {}", kept, 100 * kept / max(kept + removed, 1), removed)
        return kept, removed


# ============================================================================
# Phase 4: Quality Bucketing
# ============================================================================

class QualityBucketer:
    """Assign quality tiers based on multi-signal confidence."""

    @staticmethod
    def bucket(entry: dict) -> str:
        """Assign gold/silver/bronze/reject tier."""
        tx = entry.get("transcription", {})
        text = tx.get("text_clean", "")
        lang_prob = tx.get("language_prob", 0)
        avg_logprob = tx.get("avg_logprob", -99)
        no_speech = tx.get("no_speech_prob", 1)
        ppl = tx.get("lm_perplexity", 9999)
        arm_ratio = armenian_char_ratio(text)
        snr = entry.get("snr_db", 0)
        duration = entry.get("duration_sec", 0)

        # Reject conditions
        if not text or arm_ratio < 0.6 or no_speech > 0.8:
            return "reject"

        score = 0

        # Language confidence
        if lang_prob > 0.95:
            score += 3
        elif lang_prob > 0.85:
            score += 2
        elif lang_prob > 0.7:
            score += 1

        # ASR confidence
        if avg_logprob > -0.3:
            score += 3
        elif avg_logprob > -0.5:
            score += 2
        elif avg_logprob > -0.8:
            score += 1

        # LM perplexity
        if ppl < 100:
            score += 3
        elif ppl < 200:
            score += 2
        elif ppl < 400:
            score += 1

        # Armenian character ratio
        if arm_ratio > 0.95:
            score += 2
        elif arm_ratio > 0.85:
            score += 1

        # SNR
        if snr > 25:
            score += 2
        elif snr > 15:
            score += 1

        # Duration sweet spot (3-20 seconds)
        if 3 <= duration <= 20:
            score += 1

        if score >= 11:
            return "gold"
        elif score >= 7:
            return "silver"
        elif score >= 4:
            return "bronze"
        else:
            return "reject"

    def run(self, input_path: Path, output_dir: Path) -> dict:
        """Bucket all entries. Returns counts per tier."""
        output_dir.mkdir(parents=True, exist_ok=True)

        counts = {"gold": 0, "silver": 0, "bronze": 0, "reject": 0}
        hours = {"gold": 0.0, "silver": 0.0, "bronze": 0.0, "reject": 0.0}

        files = {
            tier: open(output_dir / f"{tier}.jsonl", "w")
            for tier in counts
        }

        try:
            with open(input_path) as f:
                for line in tqdm(f, desc="Quality bucketing"):
                    try:
                        entry = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue

                    tier = self.bucket(entry)
                    entry["quality_tier"] = tier
                    counts[tier] += 1
                    hours[tier] += entry.get("duration_sec", 0) / 3600

                    files[tier].write(json.dumps(entry, ensure_ascii=False) + "\n")
        finally:
            for f in files.values():
                f.close()

        for tier in counts:
            logger.info("  {}: {} segments ({:.1f} hours)", tier.upper(), counts[tier], hours[tier])

        total_usable = counts["gold"] + counts["silver"] + counts["bronze"]
        total_hours = hours["gold"] + hours["silver"] + hours["bronze"]
        logger.info("Total usable: {} segments ({:.1f} hours)", total_usable, total_hours)

        # Save summary
        summary = {"counts": counts, "hours": {k: round(v, 2) for k, v in hours.items()}}
        with open(output_dir / "quality_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return counts


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap ASR Transcription + LM Filtering",
    )
    parser.add_argument(
        "--phase",
        choices=["transcribe", "langid", "lm_filter", "bucket", "all"],
        default="all",
    )
    parser.add_argument(
        "--input", type=str, default="data/youtube_crawl/segments_filtered.jsonl",
        help="Input manifest (from youtube_crawl.py segment+filter phase)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/youtube_crawl",
    )
    parser.add_argument(
        "--lm-train-data", type=str, default=None,
        help="LM training data (Common Voice text file or JSONL). Auto-detected if not provided.",
    )
    parser.add_argument(
        "--whisper-model", type=str, default="large-v3",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
    )
    parser.add_argument(
        "--compute-type", type=str, default="float16",
    )
    parser.add_argument(
        "--max-perplexity", type=float, default=500.0,
    )

    args = parser.parse_args()
    setup_logger()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error("Input manifest not found: {}", input_path)
        sys.exit(1)

    # Intermediate files
    transcribed_path = output_dir / "transcribed.jsonl"
    langid_path = output_dir / "langid_filtered.jsonl"
    lm_path = output_dir / "lm_filtered.jsonl"
    quality_dir = output_dir / "quality_buckets"

    # ---- Transcribe ----
    if args.phase in ("transcribe", "all"):
        logger.info("=" * 50)
        logger.info("PHASE: Bootstrap Transcription")
        logger.info("=" * 50)
        transcriber = BootstrapTranscriber(
            model_size=args.whisper_model,
            device=args.device,
            compute_type=args.compute_type,
        )
        transcriber.run(input_path, transcribed_path)

    # ---- Language ID ----
    if args.phase in ("langid", "all"):
        logger.info("=" * 50)
        logger.info("PHASE: Language Identification Filter")
        logger.info("=" * 50)
        lang_filter = LanguageFilter(min_lang_prob=0.7, min_armenian_ratio=0.7)
        source = transcribed_path if transcribed_path.exists() else input_path
        lang_filter.run(source, langid_path)

    # ---- LM Filtering ----
    if args.phase in ("lm_filter", "all"):
        logger.info("=" * 50)
        logger.info("PHASE: Statistical LM Filtering")
        logger.info("=" * 50)

        scorer = LMScorer(n=3)

        # Train LM on Common Voice or existing clean data
        lm_train = args.lm_train_data
        if lm_train is None:
            # Auto-detect Common Voice
            cv_candidates = [
                Path("data/common_voice/train"),
                Path("data/common_voice/validated"),
            ]
            for cand in cv_candidates:
                if cand.exists():
                    lm_train = str(cand)
                    break

        if lm_train:
            scorer.train_from_file(Path(lm_train))
        else:
            logger.warning("No LM training data found. Using permissive threshold.")
            args.max_perplexity = 9999

        source = langid_path if langid_path.exists() else input_path
        scorer.run(source, lm_path, max_perplexity=args.max_perplexity)

    # ---- Quality Bucketing ----
    if args.phase in ("bucket", "all"):
        logger.info("=" * 50)
        logger.info("PHASE: Quality Bucketing")
        logger.info("=" * 50)

        bucketer = QualityBucketer()
        source = lm_path if lm_path.exists() else langid_path if langid_path.exists() else input_path
        bucketer.run(source, quality_dir)

    logger.info("Bootstrap transcription pipeline complete!")


if __name__ == "__main__":
    main()
