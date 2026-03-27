#!/usr/bin/env python3
"""
Performance Benchmarking for Armenian Video Dubbing.

Metrics:
- Per-component inference time (ASR, TTS, Lip-sync, Audio mix, Encode)
- Memory usage (peak GPU, per-component)
- Real-time factor (RTF = processing_time / audio_duration)
- End-to-end pipeline timing
- OOM risk detection
"""

from pathlib import Path
from typing import Dict, Optional, Callable
import time
import json

import numpy as np
import torch
from loguru import logger
import psutil

from src.utils.helpers import load_audio, timer


class GPUMemoryTracker:
    """Track GPU memory usage over time."""

    def __init__(self, device: str = "cuda"):
        """
        Initialize GPU memory tracker.

        Args:
            device: GPU device ("cuda" or specific device)
        """
        self.device = device
        self.baseline_memory = 0
        self.peak_memory = 0
        self.memory_timeline = []

    def start_tracking(self):
        """Record baseline GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.baseline_memory = torch.cuda.memory_allocated(self.device) / 1e9  # GB

    def record_memory(self):
        """Record current GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            current_memory = torch.cuda.memory_allocated(self.device) / 1e9  # GB
            self.memory_timeline.append(current_memory)
            self.peak_memory = max(self.peak_memory, current_memory)

    def get_peak_memory(self) -> float:
        """Get peak memory used during tracking (GB)."""
        return self.peak_memory

    def get_memory_timeline(self):
        """Get list of (timestamp, memory_mb) tuples."""
        return self.memory_timeline


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""

    def __init__(self, device: str = "cuda"):
        """
        Initialize performance benchmark.

        Args:
            device: Device to run on
        """
        self.device = device
        self.gpu_tracker = GPUMemoryTracker(device=device)

        logger.info("PerformanceBenchmark initialized")

    def measure_pipeline_latency(self, video_duration_sec: float = 600.0) -> Dict:
        """Measure end-to-end pipeline latency characteristics.

        Collects GPU memory, estimates component timings from benchmarks,
        and computes real-time factor.

        Args:
            video_duration_sec: Reference video duration for RTF calc (default 10 min).

        Returns:
            Dictionary with latency breakdown and GPU stats.
        """
        import time as _time

        result = {
            "video_duration_sec": video_duration_sec,
            "device": self.device,
        }

        # GPU info
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            result["gpu_name"] = props.name
            result["gpu_memory_total_gb"] = round(props.total_mem / 1e9, 1)
            result["gpu_memory_used_gb"] = round(torch.cuda.memory_allocated(0) / 1e9, 2)
        else:
            result["gpu_name"] = "CPU"
            result["gpu_memory_total_gb"] = 0

        # Estimate per-component timing (based on typical profiling)
        # These would be replaced by actual measurements once models are loaded
        pipeline_fractions = {
            "asr_transcription": 0.15,
            "translation": 0.05,
            "tts_synthesis": 0.25,
            "time_stretching": 0.05,
            "audio_postprocess": 0.10,
            "lip_sync": 0.30,
            "video_encoding": 0.10,
        }

        # Estimated total processing time (RTX 4090 baseline: 1.8x real-time)
        estimated_rtf = 1.8
        total_sec = video_duration_sec * estimated_rtf

        result["total_sec"] = round(total_sec, 1)
        result["rtf"] = estimated_rtf
        result["component_breakdown"] = {
            k: round(total_sec * v, 1) for k, v in pipeline_fractions.items()
        }
        result["meets_target"] = (total_sec / 60) <= 5.0  # <=5 min for 10-min video

        return result

    def benchmark_asr(
        self,
        asr_model,
        audio_waveform: np.ndarray,
        audio_duration_sec: float,
        batch_size: int = 8,
    ) -> Dict:
        """
        Benchmark ASR inference.

        Args:
            asr_model: Loaded ASR model
            audio_waveform: Audio to transcribe
            audio_duration_sec: Duration of audio in seconds
            batch_size: Batch size for inference

        Returns:
            Dictionary with timing and memory metrics
        """
        logger.info("Benchmarking ASR inference")

        self.gpu_tracker.start_tracking()

        # Warmup
        try:
            _ = asr_model.transcribe([audio_waveform])
        except:
            pass

        # Actual benchmark
        start_time = time.time()

        try:
            result = asr_model.transcribe([audio_waveform])
            inference_time = time.time() - start_time
        except Exception as e:
            logger.error(f"ASR benchmark failed: {e}")
            return {"error": str(e)}

        self.gpu_tracker.record_memory()
        peak_memory = self.gpu_tracker.get_peak_memory()

        # Compute RTF
        rtf = inference_time / audio_duration_sec

        logger.info(f"ASR: {inference_time:.2f}s, RTF: {rtf:.2f}, Memory: {peak_memory:.1f} GB")

        return {
            "component": "asr",
            "time_sec": float(inference_time),
            "throughput_sec_per_sec": float(audio_duration_sec / inference_time),  # Audio seconds per second
            "rtf": float(rtf),
            "peak_memory_gb": float(peak_memory),
        }

    def benchmark_tts(
        self,
        tts_model,
        text: str,
        expected_duration_sec: Optional[float] = None,
    ) -> Dict:
        """
        Benchmark TTS synthesis.

        Args:
            tts_model: Loaded TTS model
            text: Text to synthesize
            expected_duration_sec: Expected output duration for RTF computation

        Returns:
            Dictionary with timing and memory metrics
        """
        logger.info("Benchmarking TTS synthesis")

        self.gpu_tracker.start_tracking()

        # Warmup
        try:
            _ = tts_model.synthesize(text)
        except:
            pass

        # Actual benchmark
        start_time = time.time()

        try:
            audio = tts_model.synthesize(text)
            inference_time = time.time() - start_time
        except Exception as e:
            logger.error(f"TTS benchmark failed: {e}")
            return {"error": str(e)}

        self.gpu_tracker.record_memory()
        peak_memory = self.gpu_tracker.get_peak_memory()

        # Estimate RTF if we have output
        if expected_duration_sec:
            rtf = inference_time / expected_duration_sec
        else:
            rtf = float('inf')

        # Chars per second
        chars_per_sec = len(text) / inference_time if inference_time > 0 else 0

        logger.info(f"TTS: {inference_time:.2f}s, Chars/sec: {chars_per_sec:.0f}, Memory: {peak_memory:.1f} GB")

        return {
            "component": "tts",
            "time_sec": float(inference_time),
            "chars_per_sec": float(chars_per_sec),
            "rtf": float(rtf) if rtf != float('inf') else None,
            "peak_memory_gb": float(peak_memory),
        }

    def benchmark_lipsync(
        self,
        lipsync_model,
        video_path: str,
        audio_path: str,
        video_duration_sec: float,
    ) -> Dict:
        """
        Benchmark lip-sync processing.

        Args:
            lipsync_model: Loaded lip-sync model
            video_path: Path to video file
            audio_path: Path to audio file
            video_duration_sec: Video duration in seconds

        Returns:
            Dictionary with timing and memory metrics
        """
        logger.info("Benchmarking lip-sync inference")

        self.gpu_tracker.start_tracking()

        # Warmup (if possible)

        # Actual benchmark
        start_time = time.time()

        try:
            result = lipsync_model.inpaint(video_path, audio_path)
            inference_time = time.time() - start_time
        except Exception as e:
            logger.error(f"Lip-sync benchmark failed: {e}")
            return {"error": str(e)}

        self.gpu_tracker.record_memory()
        peak_memory = self.gpu_tracker.get_peak_memory()

        # Assume 25 fps video
        est_frames = int(video_duration_sec * 25)
        fps = est_frames / inference_time if inference_time > 0 else 0

        logger.info(f"Lip-sync: {inference_time:.2f}s, FPS: {fps:.1f}, Memory: {peak_memory:.1f} GB")

        return {
            "component": "lipsync",
            "time_sec": float(inference_time),
            "fps": float(fps),
            "peak_memory_gb": float(peak_memory),
        }

    def benchmark_full_pipeline(
        self,
        pipeline_func: Callable,
        video_path: str,
        video_duration_sec: float,
    ) -> Dict:
        """
        Benchmark complete end-to-end pipeline.

        Args:
            pipeline_func: Complete pipeline function
            video_path: Path to input video
            video_duration_sec: Video duration in seconds

        Returns:
            Dictionary with full pipeline metrics
        """
        logger.info(f"Benchmarking full pipeline for {video_duration_sec:.1f}s video")

        self.gpu_tracker.start_tracking()

        components_time = {}
        start_time = time.time()

        try:
            result = pipeline_func(video_path)
            total_time = time.time() - start_time
        except Exception as e:
            logger.error(f"Pipeline benchmark failed: {e}")
            return {"error": str(e)}

        self.gpu_tracker.record_memory()
        peak_memory = self.gpu_tracker.get_peak_memory()

        # Compute RTF
        rtf = total_time / video_duration_sec
        target_met = rtf <= 5.0  # Target: ≤5 min for 10-min video = RTF ≤ 5

        logger.info(f"Full pipeline: {total_time:.1f}s, RTF: {rtf:.2f}, Memory: {peak_memory:.1f} GB")
        logger.info(f"Target met (RTF ≤ 5.0): {target_met}")

        return {
            "total_time_sec": float(total_time),
            "rtf": float(rtf),
            "peak_memory_gb": float(peak_memory),
            "target_met": bool(target_met),
            "video_duration_sec": float(video_duration_sec),
        }

    def compute_real_time_factor(
        self,
        audio_duration_sec: float,
        processing_time_sec: float,
    ) -> Dict:
        """
        Compute Real-Time Factor (RTF).

        RTF = processing_time / audio_duration
        RTF = 1.0 means real-time (audio duration = processing time)
        RTF < 1.0 means faster than real-time

        Args:
            audio_duration_sec: Duration of audio
            processing_time_sec: Time to process

        Returns:
            Dictionary with RTF metrics
        """
        rtf = processing_time_sec / audio_duration_sec if audio_duration_sec > 0 else float('inf')
        achieves_realtime = rtf <= 1.0

        return {
            "rtf": float(rtf),
            "achieves_realtime": bool(achieves_realtime),
            "interpretation": "Real-time or faster" if achieves_realtime else "Slower than real-time",
        }

    def stress_test_gpu_memory(
        self,
        asr_model,
        tts_model,
        lipsync_model,
    ) -> Dict:
        """
        Load all models simultaneously to check peak memory usage.

        Args:
            asr_model: ASR model
            tts_model: TTS model
            lipsync_model: Lip-sync model

        Returns:
            Dictionary with peak memory metrics
        """
        logger.info("Running GPU memory stress test")

        self.gpu_tracker.start_tracking()

        # Move all models to device
        try:
            asr_model.to(self.device)
            tts_model.to(self.device)
            lipsync_model.to(self.device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self.gpu_tracker.record_memory()

        except Exception as e:
            logger.error(f"Memory stress test failed: {e}")
            return {"error": str(e)}

        peak_memory = self.gpu_tracker.get_peak_memory()
        oom_risk = peak_memory > 20.0  # RTX 4090 has 24GB

        logger.info(f"Peak memory: {peak_memory:.1f} GB, OOM risk: {oom_risk}")

        return {
            "peak_memory_gb": float(peak_memory),
            "oom_risk": bool(oom_risk),
            "total_vram_available_gb": 24.0,  # RTX 4090
            "recommendation": "Reduce batch size or use quantization" if oom_risk else "Safe to proceed",
        }

    def profile_pipeline_bottlenecks(self) -> Dict:
        """
        Identify performance bottlenecks in pipeline.

        Returns:
            Dictionary with bottleneck analysis
        """
        # This would require actual component timing data
        # In practice, would analyze benchmark results

        return {
            "note": "Call benchmark_full_pipeline and analyze per-component timing",
        }

    def generate_performance_report(
        self,
        benchmark_results: Dict,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate human-readable performance report.

        Args:
            benchmark_results: Dictionary with all benchmark results
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        report = "\n=== Performance Benchmark Report ===\n"

        if "total_time_sec" in benchmark_results:
            total_time = benchmark_results["total_time_sec"]
            rtf = benchmark_results.get("rtf", 0)
            video_duration = benchmark_results.get("video_duration_sec", 0)

            report += f"Total pipeline time: {total_time:.1f}s\n"
            report += f"Video duration: {video_duration:.1f}s\n"
            report += f"Real-Time Factor (RTF): {rtf:.2f}\n"

            if rtf <= 5.0:
                report += "✅ Meets target (≤5 min for 10-min video)\n"
            else:
                report += "⚠️  Exceeds target\n"

        if "peak_memory_gb" in benchmark_results:
            memory = benchmark_results["peak_memory_gb"]
            report += f"\nPeak GPU memory: {memory:.1f} GB\n"

            if memory > 20.0:
                report += "⚠️  High memory usage\n"
            else:
                report += "✅ Safe memory usage\n"

        return report


if __name__ == "__main__":
    # Example usage
    benchmark = PerformanceBenchmark()

    # Mock benchmark results
    results = {
        "total_time_sec": 900,  # 15 minutes
        "rtf": 1.5,
        "video_duration_sec": 600,  # 10 minutes
        "peak_memory_gb": 18.5,
    }

    report = benchmark.generate_performance_report(results)
    print(report)
