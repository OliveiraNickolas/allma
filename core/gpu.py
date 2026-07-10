"""
GPU detection and VRAM allocation functions.
"""
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

from core.config import logger, BASE_MODELS
import core.state as state


def _calc_model_size_gb(model_path: str) -> float:
    """Sum .safetensors files in a model directory to estimate model size in GB.

    Excludes .cache/ and hidden directories to avoid double-counting HuggingFace
    download cache files which are symlinks or duplicates of the real weights.
    """
    total_bytes = 0
    for root, dirs, files in os.walk(model_path):
        # Skip hidden dirs (.cache, .git, etc.) in-place so os.walk won't descend
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.endswith(".safetensors"):
                try:
                    fpath = os.path.join(root, f)
                    # Skip symlinks — they point to the real file already counted
                    if not os.path.islink(fpath):
                        total_bytes += os.path.getsize(fpath)
                except OSError:
                    pass
    return total_bytes / (1024 ** 3)


def _estimate_kv_cache_gb(model_path: str, max_model_len: int, kv_dtype: str = "auto") -> float:
    """Estimate KV cache memory in GB by reading model config.json.

    Handles hybrid models (e.g. Qwen3.5) that only allocate KV cache on full
    attention layers, not on linear/recurrent layers.
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return max_model_len * 65536 / (1024 ** 3)  # rough fallback
    try:
        config = json.loads(config_path.read_text())
        # Multimodal models nest text config under "text_config"
        tc = config.get("text_config", config)
        num_layers = tc.get("num_hidden_layers", 32)
        num_kv_heads = tc.get("num_key_value_heads") or tc.get("num_attention_heads", 8)
        hidden_size = tc.get("hidden_size", 4096)
        num_attn_heads = tc.get("num_attention_heads", 16)
        head_dim = tc.get("head_dim") or (hidden_size // num_attn_heads)
        # Hybrid models (linear attention + full attention): only full attention layers
        # consume KV cache that scales with sequence length.
        full_attn_interval = tc.get("full_attention_interval")
        if full_attn_interval:
            num_kv_layers = max(1, num_layers // full_attn_interval)
        else:
            num_kv_layers = num_layers
        # q8_0 and fp8 = 1 byte/element; fp16/bf16/auto = 2 bytes/element
        dtype_bytes = 1 if kv_dtype in ("fp8", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1") else 2
        # Sliding window attention (e.g. Gemma4): local attention layers only cache
        # `sliding_window` tokens, while global attention layers cache the full context.
        # Typical pattern: ~1/6 of layers are global, rest are local sliding window.
        sliding_window = tc.get("sliding_window")
        if sliding_window and sliding_window < max_model_len:
            n_global = max(1, round(num_kv_layers / 6))
            n_local = num_kv_layers - n_global
            kv_bytes = (n_global * max_model_len + n_local * sliding_window) * 2 * num_kv_heads * head_dim * dtype_bytes
            return kv_bytes / (1024 ** 3)
        kv_bytes_per_token = 2 * num_kv_heads * head_dim * num_kv_layers * dtype_bytes
        return max_model_len * kv_bytes_per_token / (1024 ** 3)
    except Exception:
        return max_model_len * 65536 / (1024 ** 3)


def _get_kv_dtype(cfg: dict) -> str:
    """Extract kv-cache-dtype from extra_args list."""
    extra_args = cfg.get("extra_args", [])
    if "--kv-cache-dtype" in extra_args:
        idx = extra_args.index("--kv-cache-dtype")
        if idx + 1 < len(extra_args):
            return extra_args[idx + 1]
    return "auto"


def get_free_gpu_memory() -> list[dict]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,index", "--format=csv,nounits,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        free_mbs = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = line.split(",")
                try:
                    gpu_id = int(parts[-1].strip()) if len(parts) > 1 else 0
                    free_mb = float(parts[0].strip()) if parts[0].strip() else 0.0
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse nvidia-smi line '{line}': {e}")
                    continue
                free_mbs.append({"index": gpu_id, "free_mb": free_mb, "free_gb": free_mb / 1024})
        return free_mbs
    except Exception as e:
        logger.error(f"Error reading VRAM: {e}")
        return []


def get_best_gpu() -> int:
    """Get GPU with most free memory."""
    gpus = get_free_gpu_memory()
    if not gpus:
        logger.warning("No GPU info available, using GPU 0")
        return 0
    best = max(gpus, key=lambda g: g["free_gb"])
    logger.debug(f"Selected GPU {best['index']} with {best['free_gb']:.1f}GB free")
    return best["index"]


def get_all_gpus() -> list[dict]:
    """Get all GPUs with their VRAM info, respecting ALLMA_VISIBLE_DEVICES env var."""
    visible_devices = os.environ.get("ALLMA_VISIBLE_DEVICES", None)
    visible_gpus = None
    if visible_devices:
        try:
            visible_gpus = set(int(x.strip()) for x in visible_devices.split(","))
            logger.debug(f"ALLMA_VISIBLE_DEVICES={visible_devices}, restricting to GPUs: {visible_gpus}")
        except ValueError as e:
            logger.error(f"Invalid ALLMA_VISIBLE_DEVICES: {visible_devices}. Must be comma-separated integers. {e}")
            visible_gpus = None

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = line.split(",")
                try:
                    gpu_id = int(parts[0].strip()) if parts[0].strip() else 0
                    total_mb = float(parts[1].strip()) if parts[1].strip() else 24576
                    free_mb = float(parts[2].strip()) if parts[2].strip() else 0.0
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse nvidia-smi line '{line}': {e}")
                    continue
                if visible_gpus is None or gpu_id in visible_gpus:
                    gpus.append({
                        "index": gpu_id,
                        "free_mb": free_mb,
                        "free_gb": free_mb / 1024,
                        "total_mb": total_mb,
                        "total_gb": total_mb / 1024,
                    })
        if visible_gpus is not None:
            logger.info(f"Found {len(gpus)} GPU(s) visible to ALLMA: {[g['index'] for g in gpus]}")
        return gpus
    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")
        return []



def find_optimal_tp_and_gpus(base_name: str, skip_gpu: int | None = None) -> tuple[int, int]:
    """
    Find the optimal tensor parallel size and GPU allocation for a model.
    Returns (adjusted_tp, selected_gpu).
    """
    cfg = state.effective_base_cfg(base_name)
    backend = cfg.get("backend", "vllm")

    if backend != "vllm":
        gpus = get_all_gpus()
        if gpus:
            best = max(gpus, key=lambda g: g["free_gb"])
            return 1, best["index"]
        return 1, 0

    model_path = cfg.get("path", "")
    if not model_path or not os.path.isdir(model_path):
        gpus = get_all_gpus()
        if gpus:
            best = max(gpus, key=lambda g: g["free_gb"])
            return 1, best["index"]
        return 1, 0

    gpu_mem_util = float(cfg.get("gpu_memory_utilization", "0.90"))
    requested_tp = int(cfg.get("tensor_parallel", "1"))

    all_gpus = get_all_gpus()
    if not all_gpus:
        return 1, 0

    all_gpus.sort(key=lambda g: g["free_gb"], reverse=True)

    if skip_gpu is not None:
        all_gpus = [g for g in all_gpus if g["index"] != skip_gpu]
    if not all_gpus:
        all_gpus = get_all_gpus()
        all_gpus.sort(key=lambda g: g["free_gb"], reverse=True)

    total_free_gb = sum(g["free_gb"] for g in all_gpus)
    max_gpu_gb = max(g["total_gb"] for g in all_gpus)

    try:
        total_size_gb = _calc_model_size_gb(model_path)
        kv_cache_gb = _estimate_kv_cache_gb(
            model_path,
            int(cfg.get("max_model_len", "40960")),
            _get_kv_dtype(cfg),
        )
        required_gb = total_size_gb * 1.06 + kv_cache_gb + 1.0
    except Exception as e:
        logger.warning(f"Could not estimate model size for {base_name}: {e}")
        best = all_gpus[0] if all_gpus else {"index": 0}
        return requested_tp, best["index"]

    usable_per_gpu_gb = max_gpu_gb * gpu_mem_util
    min_tp_needed = max(1, math.ceil(required_gb / usable_per_gpu_gb))
    effective_tp = max(requested_tp, min_tp_needed)

    if effective_tp > requested_tp:
        logger.info(
            f"{base_name}: Auto-upgrading TP {requested_tp}→{effective_tp} "
            f"(model needs {required_gb:.1f}GB, single GPU has {usable_per_gpu_gb:.1f}GB usable)"
        )

    if len(all_gpus) < effective_tp:
        logger.error(
            f"❌ {base_name}: Need TP={effective_tp} but only {len(all_gpus)} GPU(s) available. "
            f"Model requires {required_gb:.1f}GB, system has {total_free_gb:.1f}GB free total."
        )
        best = all_gpus[0] if all_gpus else {"index": 0}
        return len(all_gpus), best["index"]

    all_gpus_by_index = sorted(all_gpus, key=lambda g: g["index"])
    for i in range(len(all_gpus_by_index) - effective_tp + 1):
        candidate = all_gpus_by_index[i:i + effective_tp]
        indices = [g["index"] for g in candidate]
        is_consecutive = all(indices[j] + 1 == indices[j + 1] for j in range(len(indices) - 1))
        if is_consecutive:
            if skip_gpu is not None and skip_gpu in indices:
                continue
            return effective_tp, indices[0]

    logger.warning(f"{base_name}: No consecutive GPU group found for TP={effective_tp}, using GPU 0")
    best = all_gpus[0] if all_gpus else {"index": 0}
    return effective_tp, best["index"]


def get_vram_breakdown(cfg: Dict[str, Any], base_name: str = "") -> Dict[str, float]:
    """Estimate VRAM usage split by component. All values in GB.

    Returns a dict with:
        weights_gb    — model weights as loaded (quantized size for GGUF,
                        file bytes + load overhead for vLLM)
        kv_cache_gb   — KV cache at the configured context length
        mmproj_gb     — llama.cpp vision projector, when configured
        vision_gb     — vLLM multimodal encoder working memory
        cudagraph_gb  — CUDA graph capture workspace (vLLM, eager off)
        mtp_gb        — speculative/MTP draft context
        overhead_gb   — fixed backend overhead
        total_gb      — sum of the above

    The single-number `get_model_vram_need` wraps this and returns total_gb.
    """
    backend = cfg.get("backend", "vllm")
    extra_args = cfg.get("extra_args", [])
    zero = {
        "weights_gb": 0.0, "kv_cache_gb": 0.0, "mmproj_gb": 0.0,
        "vision_gb": 0.0, "cudagraph_gb": 0.0, "mtp_gb": 0.0,
        "overhead_gb": 0.0, "total_gb": 0.0,
    }

    def _finish(b: Dict[str, float]) -> Dict[str, float]:
        b["total_gb"] = sum(v for k, v in b.items() if k != "total_gb")
        return b

    try:
        if backend == "vllm":
            model_path = cfg.get("path", "")
            if not model_path or not os.path.isdir(model_path):
                return _finish({**zero, "weights_gb": 3.0, "overhead_gb": 1.0})

            b = dict(zero)
            b["weights_gb"] = _calc_model_size_gb(model_path) * 1.06
            model_len = int(cfg.get("max_model_len", "40960"))
            b["kv_cache_gb"] = _estimate_kv_cache_gb(model_path, model_len, _get_kv_dtype(cfg))
            b["overhead_gb"] = 1.0

            # Multimodal encoder (Qwen-VL etc): activations + image-token buffers
            # beyond the encoder weights already counted in weights_gb.
            try:
                mconf = json.loads((Path(model_path) / "config.json").read_text())
                if "vision_config" in mconf:
                    b["vision_gb"] = 0.8
            except Exception:
                pass

            # CUDA graph capture workspace. enforce_eager disables it entirely.
            enforce_eager = (
                bool(cfg.get("enforce_eager"))
                and str(cfg.get("enforce_eager")).lower() not in ("false", "0", "no")
            ) or "--enforce-eager" in extra_args
            if not enforce_eager:
                # Scales with capture size; the vLLM default (512) is heavy.
                cap = 512
                if "--max-cudagraph-capture-size" in extra_args:
                    idx = extra_args.index("--max-cudagraph-capture-size")
                    if idx + 1 < len(extra_args):
                        try:
                            cap = int(extra_args[idx + 1])
                        except ValueError:
                            pass
                b["cudagraph_gb"] = 0.5 if cap <= 16 else (1.0 if cap <= 128 else 1.8)

            # Speculative decoding / MTP draft context.
            if "--speculative-config" in extra_args:
                b["mtp_gb"] = 0.6

            return _finish(b)

        elif backend == "llama.cpp":
            model_file = cfg.get("model", "")
            if not model_file or not os.path.isfile(model_file):
                return _finish({**zero, "weights_gb": 3.0, "overhead_gb": 1.0})
            # CPU-only (n_gpu_layers=0): the model runs entirely in system RAM and
            # uses essentially no VRAM. Without this, a CPU model is sized by its file
            # bytes and wrongly gated by the GPU-free check — refused to load when the
            # GPUs are full, even though it never touches the GPU.
            try:
                if int(str(cfg.get("n_gpu_layers", "-1")).strip()) == 0:
                    return _finish({**zero, "overhead_gb": 0.3})
            except (TypeError, ValueError):
                pass

            b = dict(zero)
            # GGUF loads exactly its quantized size — only 1% overhead
            b["weights_gb"] = os.path.getsize(model_file) / (1024 ** 3) * 1.01
            mmproj_file = cfg.get("mmproj", "")
            if mmproj_file and os.path.isfile(mmproj_file):
                b["mmproj_gb"] = os.path.getsize(mmproj_file) / (1024 ** 3)
            n_ctx = int(cfg.get("n_ctx", "40960"))
            model_dir = str(Path(model_file).parent)

            # Read actual --cache-type-k value from extra_args (not just presence)
            kv_dtype = "auto"
            if "--cache-type-k" in extra_args:
                idx = extra_args.index("--cache-type-k")
                kv_dtype = extra_args[idx + 1] if idx + 1 < len(extra_args) else "q8_0"

            kv_cache_gb = _estimate_kv_cache_gb(model_dir, n_ctx, kv_dtype)
            # If config.json not found in GGUF dir, fall back using real bytes/element
            # q4_0/q4_1 = 0.5 B/elem; q5_x = 0.625; q8_0/fp8 = 1.0; fp16 = 2.0
            if kv_cache_gb == n_ctx * 65536 / (1024 ** 3):
                _dtype_bytes = {"q4_0": 0.5, "q4_1": 0.5, "q5_0": 0.625, "q5_1": 0.625,
                                "q8_0": 1.0, "fp8": 1.0}.get(kv_dtype, 2.0)
                kv_cache_gb = (n_ctx * 2 * 32 * 128 * _dtype_bytes) / (1024 ** 3)
            b["kv_cache_gb"] = kv_cache_gb

            # MTP / speculative draft context (llama-server logs ~0.5-0.6 GB
            # for "estimated memory usage of MTP context").
            if "--spec-type" in extra_args:
                b["mtp_gb"] = 0.6

            b["overhead_gb"] = 0.25
            return _finish(b)
    except Exception as e:
        logger.error(f"Error estimating VRAM for {base_name}: {e}")
    return _finish({**zero, "weights_gb": 3.0, "overhead_gb": 1.0})


def get_model_vram_need(cfg: Dict[str, Any], base_name: str) -> float:
    """Single-number VRAM estimate — sum of the per-component breakdown."""
    return get_vram_breakdown(cfg, base_name)["total_gb"]
