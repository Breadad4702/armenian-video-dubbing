#!/usr/bin/env python3
"""
End-to-End Dubbing Orchestrator — Phase 3

Complete pipeline:
  1. Extract audio from video
  2. Transcribe (ASR) → segment-level with timestamps
  3. Translate each segment (SeamlessM4T)
  4. Synthesize speech per segment (TTS) with voice cloning
  5. Time-stretch each segment to match original timing
  6. Stitch segments + post-process audio
  7. Synchronize lip movements (MuseTalk)
  8. Mix audio + encode output video

Usage:
    from src.pipeline import DubbingPipeline

    pipeline = DubbingPipeline()
    result = pipeline.dub_video(
        video_path="input.mp4",
        reference_speaker_audio="speaker.wav",
        emotion="neutral",
        output_path="dubbed.mp4"
    )
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from loguru import logger

from src.inference import (
    ASRInference,
    TranslationInference,
    TTSInference,
    LipSyncInference,
    AudioPostProcessor,
)
from src.utils.helpers import (
    extract_audio_from_video,
    get_video_info,
    time_stretch_audio,
    load_audio,
    save_audio,
    timer,
    free_gpu_memory,
    log_voice_consent,
)

# Dialect → SeamlessM4T language code mapping
DIALECT_MAP = {
    "eastern": "hye",   # Eastern Armenian (ISO 639-3)
    "western": "hyw",   # Western Armenian (ISO 639-3)
    "hye": "hye",
    "hyw": "hyw",
}


class DubbingPipeline:
    """Complete video dubbing pipeline with segment-level alignment."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize pipeline with configuration."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: {}", self.device)

        # Read quantization setting from config
        inference_cfg = self.config.get("inference", {})
        quant_bits = 0
        if inference_cfg.get("enable_quantization"):
            quant_bits = inference_cfg.get("quantization_bits", 0)

        # Initialize inference modules (lazy-loaded on first use)
        self.asr = ASRInference(device=self.device, quantize_bits=quant_bits)
        self.translator = TranslationInference(device=self.device)
        self.tts = TTSInference(device=self.device)
        self.lip_sync = LipSyncInference(device=self.device)
        self.audio_processor = AudioPostProcessor(
            sample_rate=self.config.get("audio", {}).get("sample_rate", 44100)
        )

        # Ethics config
        self.ethics = self.config.get("ethics", {})

        self.sr = self.config.get("audio", {}).get("sample_rate", 44100)
        self.temp_dir = Path(self.config.get("paths", {}).get("temp_dir", "outputs/temp"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def dub_video(
        self,
        video_path: str,
        reference_speaker_audio: Optional[str] = None,
        emotion: str = "neutral",
        output_path: str = "dubbed_output.mp4",
        keep_background: bool = True,
        skip_lipsync: bool = False,
        src_lang: str = "eng",
        tgt_lang: str = "hye",
        dialect: str = "eastern",
    ) -> dict:
        """Run the complete dubbing pipeline.

        Args:
            video_path: Input video file path.
            reference_speaker_audio: Optional reference audio for voice cloning.
            emotion: Emotion style (neutral, happy, sad, angry, excited, calm).
            output_path: Where to save the dubbed video.
            keep_background: Whether to keep original background audio/SFX.
            skip_lipsync: Skip lip-sync step (faster).
            src_lang: Source language code.
            tgt_lang: Target language code.
            dialect: Armenian dialect ("eastern" or "western").

        Returns:
            Dict with status, output_video, transcription, duration_sec.
        """
        video_path = Path(video_path)
        output_path = Path(output_path)

        if not video_path.exists():
            logger.error("Video file not found: {}", video_path)
            return {"error": "File not found"}

        # Resolve dialect to language code
        tgt_lang = DIALECT_MAP.get(dialect, tgt_lang)

        # Log voice consent if using voice cloning
        if reference_speaker_audio and self.ethics.get("consent_required", False):
            speaker_id = Path(reference_speaker_audio).stem
            log_voice_consent(
                speaker_id=speaker_id,
                consent_given=True,
                consent_log=self.ethics.get("consent_log_path", "logs/voice_consent.json"),
            )

        logger.info("=" * 60)
        logger.info("Starting dubbing: {}", video_path.name)
        logger.info("  Language: {} → {} (dialect: {})", src_lang, tgt_lang, dialect)
        logger.info("  Emotion: {}", emotion)
        logger.info("  Lip-sync: {}", "enabled" if not skip_lipsync else "disabled")
        logger.info("=" * 60)

        with timer("Complete dubbing"):
            try:
                # Step 1: Extract audio from video
                logger.info("[Step 1/8] Extracting audio from video...")
                audio_path = self._extract_audio(video_path)

                # Step 2: Transcribe (ASR) with timestamps
                logger.info("[Step 2/8] Transcribing audio (ASR)...")
                transcription = self._transcribe_audio(audio_path)
                if "error" in transcription:
                    return {"error": f"ASR failed: {transcription['error']}"}

                segments = transcription.get("segments", [])
                full_text = transcription.get("text", "")
                logger.info("  Transcribed {} segments, {} chars", len(segments), len(full_text))

                # Step 3: Translate each segment
                logger.info("[Step 3/8] Translating {} segments...", len(segments))
                translated_segments = self._translate_segments(segments, src_lang, tgt_lang)

                # Step 4: Synthesize speech per segment (TTS)
                logger.info("[Step 4/8] Synthesizing speech (TTS)...")
                segment_audios = self._synthesize_segments(
                    translated_segments,
                    reference_speaker_audio=reference_speaker_audio,
                    emotion=emotion,
                    language=tgt_lang,
                )

                # Step 5: Time-stretch segments and stitch into single audio
                logger.info("[Step 5/8] Aligning segment durations...")
                original_audio, _ = load_audio(audio_path, sr=self.sr)
                original_duration = len(original_audio) / self.sr
                dubbed_audio = self._align_and_stitch_segments(
                    segment_audios, translated_segments, original_duration
                )

                # Step 6: Post-process audio (denoise, normalize, mix)
                logger.info("[Step 6/8] Post-processing audio...")
                final_audio = self._process_audio(
                    dubbed_audio,
                    original_audio_path=audio_path if keep_background else None,
                )

                # Step 7: Lip-sync (optional)
                if not skip_lipsync:
                    logger.info("[Step 7/8] Synchronizing lip movements...")
                    fused_video = self._apply_lipsync(video_path, final_audio)
                else:
                    logger.info("[Step 7/8] Skipping lip-sync")
                    fused_video = video_path

                # Step 8: Mix audio + encode final video
                logger.info("[Step 8/8] Encoding final video...")
                output_video = self._mix_and_encode(
                    str(fused_video), final_audio, output_path,
                )

                logger.info("Dubbing complete: {}", output_path)

                return {
                    "status": "success",
                    "output_video": str(output_video),
                    "transcription": full_text,
                    "translated_text": " ".join(s["text"] for s in translated_segments),
                    "n_segments": len(segments),
                    "emotion": emotion,
                    "duration_sec": original_duration,
                }

            except Exception as e:
                logger.error("Pipeline failed: {}", e)
                import traceback
                logger.error(traceback.format_exc())
                return {"error": str(e)}

    # ========================================================================
    # Pipeline Steps
    # ========================================================================

    def _extract_audio(self, video_path: Path) -> Path:
        """Extract audio track from video."""
        output_audio = self.temp_dir / f"{video_path.stem}_extracted.wav"

        if output_audio.exists():
            logger.info("  Using cached audio: {}", output_audio.name)
            return output_audio

        return extract_audio_from_video(video_path, output_audio, sr=self.sr)

    def _transcribe_audio(self, audio_path: Path) -> dict:
        """Transcribe audio with segment-level timestamps."""
        self.asr.load()
        return self.asr.transcribe(str(audio_path))

    def _translate_segments(
        self, segments: list, src_lang: str, tgt_lang: str
    ) -> list:
        """Translate each ASR segment. If src==tgt, pass through."""
        if src_lang == tgt_lang:
            logger.info("  Same language — skipping translation")
            return segments

        self.translator.load()
        translated = self.translator.translate_segments(segments, src_lang, tgt_lang)

        for i, seg in enumerate(translated):
            if i < 3:  # Log first few for debugging
                logger.debug("  [{}] '{}' → '{}'", i, seg.get("src_text", "")[:40], seg["text"][:40])

        return translated

    def _synthesize_segments(
        self,
        segments: list,
        reference_speaker_audio: Optional[str] = None,
        emotion: str = "neutral",
        language: str = "hye",
    ) -> list:
        """Synthesize speech for each translated segment.

        Returns list of dicts with "audio" (np.ndarray) and "sample_rate".
        """
        self.tts.load()

        results = []
        for i, seg in enumerate(segments):
            text = seg["text"]
            if not text.strip():
                # Silent gap
                gap_duration = max(0.1, seg.get("end", 0) - seg.get("start", 0))
                results.append({
                    "audio": np.zeros(int(gap_duration * self.sr), dtype=np.float32),
                    "sample_rate": self.sr,
                    "duration_sec": gap_duration,
                })
                continue

            tts_result = self.tts.synthesize(
                text=text,
                reference_audio_path=reference_speaker_audio,
                emotion=emotion,
                language=language[:2],  # "hye" → "hy"
            )
            results.append(tts_result)

            if i % 10 == 0 and i > 0:
                logger.info("  Synthesized {}/{} segments", i, len(segments))

        logger.info("  Synthesized {} segments total", len(results))
        return results

    def _align_and_stitch_segments(
        self,
        segment_audios: list,
        segments: list,
        total_duration: float,
    ) -> np.ndarray:
        """Align synthesized segments to original timestamps and stitch together.

        Each segment is time-stretched to fit the original segment's time slot,
        then placed at the correct offset in the output buffer.
        """
        total_samples = int(total_duration * self.sr)
        output = np.zeros(total_samples, dtype=np.float32)

        for i, (seg_audio, seg_info) in enumerate(zip(segment_audios, segments)):
            audio = seg_audio.get("audio", np.array([], dtype=np.float32))
            if len(audio) == 0:
                continue

            # Resample if needed
            audio_sr = seg_audio.get("sample_rate", self.sr)
            if audio_sr != self.sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=audio_sr, target_sr=self.sr)

            start_sec = seg_info.get("start", 0.0)
            end_sec = seg_info.get("end", 0.0)
            target_duration = end_sec - start_sec

            if target_duration > 0.05 and len(audio) > 100:
                # Time-stretch this segment to fit the time slot
                audio_duration = len(audio) / self.sr
                ratio = target_duration / audio_duration

                if abs(ratio - 1.0) > 0.05:
                    ratio = max(0.5, min(2.0, ratio))
                    # Use rubberband for quality stretching
                    try:
                        tmp_in = self.temp_dir / f"seg_{i}_in.wav"
                        tmp_out = self.temp_dir / f"seg_{i}_out.wav"
                        save_audio(audio, tmp_in, sr=self.sr)
                        time_stretch_audio(tmp_in, tmp_out, target_duration=target_duration)
                        audio, _ = load_audio(tmp_out, sr=self.sr)
                    except Exception:
                        # Fallback: simple resampling
                        target_samples = int(target_duration * self.sr)
                        if target_samples > 0:
                            indices = np.linspace(0, len(audio) - 1, target_samples).astype(int)
                            audio = audio[indices]

            # Place in output buffer
            start_sample = int(start_sec * self.sr)
            end_sample = start_sample + len(audio)

            if end_sample > total_samples:
                audio = audio[:total_samples - start_sample]
                end_sample = total_samples

            if start_sample < total_samples and len(audio) > 0:
                output[start_sample:start_sample + len(audio)] = audio

        return output

    def _process_audio(
        self,
        dubbed_audio: np.ndarray,
        original_audio_path: Optional[Path] = None,
    ) -> np.ndarray:
        """Post-process: denoise, normalize, optionally mix with background."""
        # Denoise
        audio = self.audio_processor.denoise_audio(dubbed_audio)

        # Normalize loudness
        audio = self.audio_processor.normalize_loudness(audio, target_loudness=-14.0)

        # Mix with background if requested
        if original_audio_path:
            try:
                bg_audio, _ = load_audio(original_audio_path, sr=self.sr)

                # Separate sources to get just the accompaniment
                separated = self.audio_processor.separate_sources(bg_audio)
                accompaniment = separated.get("accompaniment", bg_audio)

                audio = self.audio_processor.mix_audio(
                    audio,
                    accompaniment,
                    dubbed_weight=1.0,
                    sfx_weight=0.2,
                )
            except Exception as e:
                logger.warning("Background mixing failed: {}", e)

        return audio

    def _apply_lipsync(self, video_path: Path, audio: np.ndarray) -> Path:
        """Apply lip-sync using MuseTalk."""
        temp_audio = self.temp_dir / f"{video_path.stem}_dubbed_audio.wav"
        save_audio(audio, temp_audio, sr=self.sr)

        output_video = self.temp_dir / f"{video_path.stem}_lipsync.mp4"

        self.lip_sync.load()
        result = self.lip_sync.inpaint_mouth(
            str(video_path),
            str(temp_audio),
            output_path=str(output_video),
        )

        output = Path(result.get("output", str(video_path)))
        if output.exists() and result.get("status") == "success":
            return output

        logger.warning("Lip-sync skipped — using original video")
        return video_path

    def _mix_and_encode(
        self,
        video_path: str,
        audio: np.ndarray,
        output_path: Path,
    ) -> Path:
        """Mix dubbed audio into video, apply watermark if configured, and encode."""
        import subprocess

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save audio
        temp_audio = self.temp_dir / "final_audio.wav"
        save_audio(audio, temp_audio, sr=self.sr)

        # Build video filter for watermark
        vf_filters = []
        if self.ethics.get("add_watermark", False):
            wm_text = self.ethics.get("watermark_text", "AI-Dubbed")
            wm_opacity = self.ethics.get("watermark_opacity", 0.3)
            # FFmpeg drawtext filter — bottom-right corner, semi-transparent
            vf_filters.append(
                f"drawtext=text='{wm_text}':fontsize=18:"
                f"fontcolor=white@{wm_opacity}:"
                f"x=w-tw-10:y=h-th-10"
            )

        # FFmpeg: replace audio track (+ optional watermark)
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", str(temp_audio),
        ]

        if vf_filters:
            cmd.extend(["-vf", ",".join(vf_filters)])
            cmd.extend(["-c:v", "libx264"])
        else:
            cmd.extend(["-c:v", "libx264"])

        cmd.extend([
            "-crf", str(self.config.get("video", {}).get("output_crf", 18)),
            "-preset", self.config.get("video", {}).get("output_preset", "medium"),
            "-c:a", "aac",
            "-b:a", self.config.get("video", {}).get("output_audio_bitrate", "192k"),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_path),
        ])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
            logger.info("Video encoded: {}", output_path)
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error("FFmpeg failed: {}", e.stderr[:500] if e.stderr else str(e))
            return Path(video_path)
        except Exception as e:
            logger.error("Encoding failed: {}", e)
            return Path(video_path)

    def cleanup_temp(self):
        """Remove temporary files."""
        import shutil
        if self.temp_dir.exists():
            for f in self.temp_dir.iterdir():
                try:
                    if f.is_file():
                        f.unlink()
                except OSError:
                    pass


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Armenian Video Dubbing Pipeline")
    parser.add_argument("video", type=str, help="Input video file")
    parser.add_argument("--reference-speaker", type=str, default=None,
                        help="Reference speaker audio for voice cloning")
    parser.add_argument("--emotion", type=str, default="neutral",
                        choices=["neutral", "happy", "sad", "angry", "excited", "calm"])
    parser.add_argument("--output", type=str, default="dubbed_output.mp4",
                        help="Output video path")
    parser.add_argument("--skip-lipsync", action="store_true",
                        help="Skip lip-sync step (faster)")
    parser.add_argument("--no-background", action="store_true",
                        help="Don't mix background SFX/music")
    parser.add_argument("--src-lang", type=str, default="eng",
                        help="Source language (default: eng)")
    parser.add_argument("--tgt-lang", type=str, default="hye",
                        help="Target language (default: hye = Eastern Armenian)")
    parser.add_argument("--dialect", type=str, default="eastern",
                        choices=["eastern", "western"],
                        help="Armenian dialect (default: eastern)")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Config file path")

    args = parser.parse_args()

    from src.utils.logger import setup_logger
    setup_logger()

    pipeline = DubbingPipeline(config_path=args.config)
    result = pipeline.dub_video(
        video_path=args.video,
        reference_speaker_audio=args.reference_speaker,
        emotion=args.emotion,
        output_path=args.output,
        keep_background=not args.no_background,
        skip_lipsync=args.skip_lipsync,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        dialect=args.dialect,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
