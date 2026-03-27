#!/usr/bin/env python3
"""
TTS Studio Data Preparation — Phase 1f

Prepares high-quality multi-speaker recordings for TTS/voice-cloning fine-tuning:
  1. Process raw studio recordings (WAV/FLAC)
  2. Segment by silence/sentence boundaries
  3. Align audio with provided transcripts
  4. Validate quality (SNR, clipping, silence ratio)
  5. Tag with speaker ID, emotion, and prosody metadata
  6. Generate Fish-Speech / CosyVoice training manifests

Expected studio recording structure:
    data/tts_studio/
    ├── speaker_001/
    │   ├── recording_001.wav
    │   ├── recording_001.txt  (transcript, one sentence per line)
    │   ├── recording_002.wav
    │   └── metadata.json  (speaker info: name, gender, dialect, age_range)
    ├── speaker_002/
    │   └── ...

Usage:
    python scripts/data_collection/prepare_tts_data.py
    python scripts/data_collection/prepare_tts_data.py --input-dir data/tts_studio --output-dir data/tts_studio/processed
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger


class TTSDataProcessor:
    """Process studio recordings for TTS fine-tuning."""

    def __init__(self, input_dir: Path, output_dir: Path, target_sr: int = 44100):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_sr = target_sr
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def detect_speakers(self) -> list[Path]:
        """Find all speaker directories."""
        speakers = []
        for d in sorted(self.input_dir.iterdir()):
            if d.is_dir() and not d.name.startswith(".") and d.name != "processed":
                audio_files = list(d.glob("*.wav")) + list(d.glob("*.flac")) + list(d.glob("*.mp3"))
                if audio_files:
                    speakers.append(d)
        return speakers

    def load_speaker_metadata(self, speaker_dir: Path) -> dict:
        """Load speaker metadata.json if available."""
        meta_path = speaker_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return {
            "speaker_id": speaker_dir.name,
            "name": speaker_dir.name,
            "gender": "unknown",
            "dialect": "eastern",
            "age_range": "unknown",
        }

    def process_recording(
        self,
        audio_path: Path,
        transcript_path: Path | None,
        speaker_id: str,
    ) -> list[dict]:
        """Process a single recording file into utterance segments."""
        import librosa
        import soundfile as sf

        try:
            audio, sr = librosa.load(str(audio_path), sr=self.target_sr, mono=True)
        except Exception as e:
            logger.error("Cannot load {}: {}", audio_path, e)
            return []

        total_duration = len(audio) / sr

        # Load transcript if available
        sentences = []
        if transcript_path and transcript_path.exists():
            with open(transcript_path, encoding="utf-8") as f:
                sentences = [line.strip() for line in f if line.strip()]

        # If no transcript, treat as single utterance
        if not sentences:
            sentences = [f"[untranscribed_{audio_path.stem}]"]

        # Segment by silence
        segments = self._segment_by_silence(audio, sr)

        if not segments:
            # Fallback: equal-length chunks
            chunk_size = int(15.0 * sr)
            segments = []
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                if len(chunk) > sr:  # > 1 second
                    segments.append((i, min(i + chunk_size, len(audio))))

        # Match segments with sentences
        entries = []
        out_dir = self.output_dir / speaker_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx, (start_sample, end_sample) in enumerate(segments):
            segment_audio = audio[start_sample:end_sample]
            duration = len(segment_audio) / sr

            # Skip too short/long
            if duration < 1.0 or duration > 30.0:
                continue

            # Quality checks
            rms = np.sqrt(np.mean(segment_audio ** 2))
            peak = np.max(np.abs(segment_audio))

            # Skip near-silent or clipped
            if rms < 0.001 or peak > 0.999:
                continue

            # SNR estimate
            snr = self._estimate_snr(segment_audio)
            if snr < 15:
                continue

            # Match to sentence
            text = sentences[idx] if idx < len(sentences) else f"[segment_{idx:04d}]"

            # Detect emotion (basic energy/pitch heuristic)
            emotion = self._detect_emotion(segment_audio, sr)

            # Save segment
            seg_filename = f"{speaker_id}_{audio_path.stem}_{idx:04d}.wav"
            seg_path = out_dir / seg_filename
            sf.write(str(seg_path), segment_audio, sr)

            entries.append({
                "audio_path": str(seg_path),
                "text": text,
                "speaker_id": speaker_id,
                "duration_sec": round(duration, 3),
                "sample_rate": sr,
                "snr_db": round(snr, 1),
                "rms": round(float(rms), 6),
                "emotion": emotion,
                "source": "studio",
                "source_file": str(audio_path),
                "start_sec": round(start_sample / sr, 3),
                "end_sec": round(end_sample / sr, 3),
            })

        return entries

    def _segment_by_silence(
        self,
        audio: np.ndarray,
        sr: int,
        silence_thresh_db: float = -40,
        min_silence_ms: int = 500,
        min_segment_s: float = 2.0,
        max_segment_s: float = 20.0,
    ) -> list[tuple[int, int]]:
        """Segment audio at silence boundaries."""
        import librosa

        # Compute frame-level RMS
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Find silence frames
        is_silence = rms_db < silence_thresh_db

        # Convert frames to samples
        min_silence_frames = int(min_silence_ms / 10)
        min_segment_frames = int(min_segment_s * 1000 / 10)
        max_segment_frames = int(max_segment_s * 1000 / 10)

        # Find silence regions
        segments = []
        start = 0
        silence_count = 0

        for i, silent in enumerate(is_silence):
            if silent:
                silence_count += 1
            else:
                if silence_count >= min_silence_frames:
                    seg_len = i - silence_count - start
                    if seg_len >= min_segment_frames:
                        # Split long segments
                        seg_start = start
                        seg_end = i - silence_count
                        while seg_end - seg_start > max_segment_frames:
                            segments.append((
                                seg_start * hop_length,
                                (seg_start + max_segment_frames) * hop_length,
                            ))
                            seg_start += max_segment_frames
                        if seg_end - seg_start >= min_segment_frames:
                            segments.append((
                                seg_start * hop_length,
                                seg_end * hop_length,
                            ))
                    start = i
                silence_count = 0

        # Final segment
        if len(is_silence) - start >= min_segment_frames:
            segments.append((start * hop_length, len(audio)))

        return segments

    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate SNR using percentile method."""
        frame_size = 512
        energies = []
        for i in range(0, len(audio) - frame_size, frame_size):
            frame_e = np.mean(audio[i:i + frame_size] ** 2)
            if frame_e > 0:
                energies.append(frame_e)

        if not energies:
            return 0.0

        energies = np.array(energies)
        noise = np.percentile(energies, 10)
        signal = np.percentile(energies, 90)

        if noise <= 0:
            return 60.0

        return float(10 * np.log10(signal / noise))

    def _detect_emotion(self, audio: np.ndarray, sr: int) -> str:
        """Basic emotion detection based on energy and speaking rate."""
        import librosa

        rms = np.sqrt(np.mean(audio ** 2))
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

        # Very basic heuristic
        if rms > 0.1 and zcr > 0.1:
            return "excited"
        elif rms > 0.06:
            return "neutral"
        elif rms < 0.02:
            return "calm"
        else:
            return "neutral"

    def generate_fish_speech_manifest(self, entries: list[dict], output_path: Path):
        """Generate manifest in Fish-Speech training format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for e in entries:
                fish_entry = {
                    "audio": e["audio_path"],
                    "text": e["text"],
                    "speaker": e["speaker_id"],
                    "duration": e["duration_sec"],
                    "emotion": e.get("emotion", "neutral"),
                }
                f.write(json.dumps(fish_entry, ensure_ascii=False) + "\n")

        logger.info("Fish-Speech manifest: {} entries -> {}", len(entries), output_path)

    def generate_cosyvoice_manifest(self, entries: list[dict], output_path: Path):
        """Generate manifest in CosyVoice training format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for e in entries:
                # CosyVoice expects: utt_id | speaker | audio_path | text | duration
                utt_id = Path(e["audio_path"]).stem
                f.write(f"{utt_id}|{e['speaker_id']}|{e['audio_path']}|{e['text']}|{e['duration_sec']}\n")

        logger.info("CosyVoice manifest: {} entries -> {}", len(entries), output_path)

    def run(self) -> dict:
        """Process all studio recordings."""
        speakers = self.detect_speakers()

        if not speakers:
            logger.warning("No speaker directories found in {}", self.input_dir)
            logger.info("Expected structure:")
            logger.info("  data/tts_studio/speaker_001/recording_001.wav")
            logger.info("  data/tts_studio/speaker_001/recording_001.txt")
            logger.info("  data/tts_studio/speaker_001/metadata.json")
            return {}

        logger.info("Found {} speakers", len(speakers))

        all_entries = []
        stats = {}

        for speaker_dir in speakers:
            meta = self.load_speaker_metadata(speaker_dir)
            speaker_id = meta.get("speaker_id", speaker_dir.name)

            logger.info("Processing speaker: {} ({})", speaker_id, meta.get("gender", "?"))

            audio_files = sorted(
                list(speaker_dir.glob("*.wav")) +
                list(speaker_dir.glob("*.flac")) +
                list(speaker_dir.glob("*.mp3"))
            )

            speaker_entries = []
            for audio_file in tqdm(audio_files, desc=f"  {speaker_id}"):
                # Look for matching transcript
                txt_path = audio_file.with_suffix(".txt")

                entries = self.process_recording(audio_file, txt_path, speaker_id)
                for e in entries:
                    e.update({
                        "gender": meta.get("gender", "unknown"),
                        "dialect": meta.get("dialect", "eastern"),
                    })
                speaker_entries.extend(entries)

            all_entries.extend(speaker_entries)

            hours = sum(e["duration_sec"] for e in speaker_entries) / 3600
            stats[speaker_id] = {
                "segments": len(speaker_entries),
                "hours": round(hours, 2),
                "gender": meta.get("gender", "unknown"),
                "dialect": meta.get("dialect", "eastern"),
            }
            logger.info("  {} -> {} segments ({:.1f}h)", speaker_id, len(speaker_entries), hours)

        # Write main manifest
        manifest_path = self.output_dir / "tts_manifest.jsonl"
        with open(manifest_path, "w") as f:
            for e in all_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        # Write Fish-Speech and CosyVoice format manifests
        self.generate_fish_speech_manifest(all_entries, self.output_dir / "fish_speech_manifest.jsonl")
        self.generate_cosyvoice_manifest(all_entries, self.output_dir / "cosyvoice_manifest.txt")

        # Save stats
        total_hours = sum(e["duration_sec"] for e in all_entries) / 3600
        summary = {
            "total_segments": len(all_entries),
            "total_hours": round(total_hours, 2),
            "speakers": stats,
        }
        with open(self.output_dir / "tts_stats.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("TTS data processing complete: {} segments, {:.1f}h, {} speakers",
                    len(all_entries), total_hours, len(stats))
        return summary


def main():
    parser = argparse.ArgumentParser(description="TTS Studio Data Processor")
    parser.add_argument("--input-dir", default="data/tts_studio")
    parser.add_argument("--output-dir", default="data/tts_studio/processed")
    parser.add_argument("--sample-rate", type=int, default=44100)

    args = parser.parse_args()
    setup_logger()

    processor = TTSDataProcessor(
        Path(args.input_dir), Path(args.output_dir), args.sample_rate
    )
    processor.run()


if __name__ == "__main__":
    main()
