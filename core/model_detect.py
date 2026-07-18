"""
Auto-detection of model family and backend arguments from model files.

When a base config has no extra_args, we read the model's config.json (or GGUF
metadata) to identify the family, then apply the matching preset from FAMILY_PRESETS.
"""
import json
from pathlib import Path
from typing import Optional

from core.detect import FAMILY_PRESETS, detect_model


def get_family(model_path: str) -> str:
    """Detect model family from path. Returns a FAMILY_PRESETS key."""
    path = Path(model_path)
    if not path.exists():
        return "generic"
    info = detect_model(path)
    return info.get("family", "generic")


def get_auto_extra_args(cfg: dict, backend: str) -> list:
    """
    Return extra_args for a model config.
    If the config already has extra_args, return them as-is (user override).
    Otherwise detect from model files and apply the family preset.
    """
    if "extra_args" in cfg:
        return cfg["extra_args"]

    model_path = cfg.get("path") or cfg.get("model", "")
    if not model_path:
        return []

    family = get_family(model_path)
    preset = FAMILY_PRESETS.get(family, FAMILY_PRESETS["generic"])

    if backend == "vllm":
        return preset.get("vllm_extra_args", [])
    else:
        return preset.get("llama_extra_args", [])


def get_auto_max_model_len(cfg: dict) -> Optional[int]:
    """
    Return max_model_len for a vLLM config.
    If set in config, return as-is. Otherwise read from config.json and apply
    a conservative cap based on model size.
    """
    if "max_model_len" in cfg:
        return int(cfg["max_model_len"])

    model_path = cfg.get("path", "")
    if not model_path:
        return 131072  # safe default

    cfg_json = Path(model_path) / "config.json"
    if not cfg_json.exists():
        return 131072

    try:
        data = json.loads(cfg_json.read_text())
        tc = data.get("text_config", data)
        native_ctx = tc.get("max_position_embeddings")
        if not native_ctx:
            return 131072
        # Conservative cap: avoid OOM on large context models
        return min(native_ctx, 131072)
    except Exception:
        return 131072


def get_family_label(model_path: str) -> str:
    """Human-readable family label for logging."""
    family = get_family(model_path)
    return FAMILY_PRESETS.get(family, {}).get("label", "Generic")
