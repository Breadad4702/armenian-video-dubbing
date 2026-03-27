#!/usr/bin/env python3
"""
Cost Estimation for Armenian Video Dubbing at Scale

Calculates cost-per-minute for various cloud GPU providers.

Usage:
    python scripts/deployment/cost_estimate.py
"""

# ============================================================================
# Cost per minute estimates (March 2026 pricing)
# ============================================================================

PROVIDER_COSTS = {
    "runpod_rtx4090": {
        "name": "RunPod RTX 4090",
        "gpu_cost_per_hour": 0.74,
        "minutes_per_10min_video": 18,  # ~18 min processing for 10 min video
    },
    "runpod_a100_80gb": {
        "name": "RunPod A100 80GB",
        "gpu_cost_per_hour": 1.64,
        "minutes_per_10min_video": 10,  # faster with more VRAM
    },
    "aws_g5_xlarge": {
        "name": "AWS g5.xlarge (A10G)",
        "gpu_cost_per_hour": 1.006,
        "minutes_per_10min_video": 22,
    },
    "aws_p4d_24xlarge": {
        "name": "AWS p4d.24xlarge (8xA100)",
        "gpu_cost_per_hour": 32.77,
        "minutes_per_10min_video": 3,  # multi-GPU parallel
    },
    "gcp_a100_40gb": {
        "name": "GCP A100 40GB",
        "gpu_cost_per_hour": 2.48,
        "minutes_per_10min_video": 12,
    },
    "local_rtx4090": {
        "name": "Local RTX 4090 (electricity only)",
        "gpu_cost_per_hour": 0.05,  # ~350W * $0.15/kWh
        "minutes_per_10min_video": 18,
    },
}

# Pipeline step breakdown (% of total processing time)
PIPELINE_STEPS = {
    "asr_transcription": 0.15,
    "translation": 0.05,
    "tts_synthesis": 0.25,
    "time_stretching": 0.05,
    "audio_postprocess": 0.10,
    "lip_sync": 0.30,
    "video_encoding": 0.10,
}


def estimate_cost(provider_key: str, video_minutes: float = 10.0) -> dict:
    """Estimate cost for dubbing a video."""
    provider = PROVIDER_COSTS[provider_key]

    processing_minutes = provider["minutes_per_10min_video"] * (video_minutes / 10.0)
    processing_hours = processing_minutes / 60.0
    total_cost = processing_hours * provider["gpu_cost_per_hour"]
    cost_per_minute = total_cost / video_minutes

    return {
        "provider": provider["name"],
        "video_duration_min": video_minutes,
        "processing_time_min": round(processing_minutes, 1),
        "total_cost_usd": round(total_cost, 4),
        "cost_per_minute_usd": round(cost_per_minute, 4),
        "gpu_cost_per_hour": provider["gpu_cost_per_hour"],
    }


def print_cost_table():
    """Print cost comparison table."""
    print("=" * 80)
    print("Armenian Video Dubbing — Cost per Minute Estimates")
    print("=" * 80)
    print()
    print(f"{'Provider':<30} {'$/hr GPU':<10} {'Process':<10} {'$/10min':<10} {'$/min':<10}")
    print("-" * 70)

    for key in PROVIDER_COSTS:
        e = estimate_cost(key, video_minutes=10.0)
        print(
            f"{e['provider']:<30} "
            f"${e['gpu_cost_per_hour']:<9.2f} "
            f"{e['processing_time_min']:<9.1f}m "
            f"${e['total_cost_usd']:<9.4f} "
            f"${e['cost_per_minute_usd']:<9.4f}"
        )

    print()
    print("Notes:")
    print("  - Processing times assume RTX 4090 baseline (~18 min for 10-min video)")
    print("  - With 4-bit quantization + ONNX export: ~40% faster (target: ≤5 min)")
    print("  - Batch processing amortizes model loading across videos")
    print("  - Prices as of March 2026; check provider for current rates")

    print()
    print("Pipeline Breakdown (% of processing time):")
    for step, pct in PIPELINE_STEPS.items():
        bar = "#" * int(pct * 40)
        print(f"  {step:<25} {pct:>5.0%} {bar}")


if __name__ == "__main__":
    print_cost_table()
