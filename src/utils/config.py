#!/usr/bin/env python3
"""
Configuration loader for Armenian Video Dubbing AI.

Loads YAML config with environment variable overrides.
Usage:
    from src.utils.config import get_config
    cfg = get_config()
    print(cfg.asr.backend)  # "whisper"
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class DotDict(dict):
    """Dictionary with dot-notation access."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
            if isinstance(val, dict) and not isinstance(val, DotDict):
                val = DotDict(val)
                self[key] = val
            return val
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")


_CONFIG_CACHE: DotDict | None = None


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _resolve_env_vars(d: dict) -> dict:
    """Replace ${ENV_VAR} patterns with environment variable values."""
    resolved = {}
    for k, v in d.items():
        if isinstance(v, dict):
            resolved[k] = _resolve_env_vars(v)
        elif isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            env_key = v[2:-1]
            resolved[k] = os.environ.get(env_key, v)
        else:
            resolved[k] = v
    return resolved


def load_config(
    config_path: str | Path | None = None,
    override_path: str | Path | None = None,
) -> DotDict:
    """Load configuration from YAML file(s).

    Args:
        config_path: Path to main config YAML. Defaults to configs/config.yaml.
        override_path: Optional path to override YAML (merged on top).

    Returns:
        DotDict configuration object.
    """
    if config_path is None:
        # Check env var first, then default
        config_path = os.environ.get(
            "ARMTTS_CONFIG",
            Path(__file__).parent.parent.parent / "configs" / "config.yaml",
        )

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    if override_path:
        override_path = Path(override_path)
        if override_path.exists():
            with open(override_path) as f:
                overrides = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, overrides)

    cfg = _resolve_env_vars(cfg)
    return DotDict(cfg)


def get_config(**kwargs) -> DotDict:
    """Get or create cached configuration singleton."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config(**kwargs)
    return _CONFIG_CACHE


def reset_config() -> None:
    """Reset cached configuration (for testing)."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None
