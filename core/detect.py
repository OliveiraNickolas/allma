"""
Model detection, family presets and hardware-aware suggestions.

Single source of truth consumed by the TUI, the config wizard
(create_config.py) and the CLI. Everything here is stdlib-only so importing
it never drags in backend or server dependencies.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path


# ==============================================================================
# Presets by model family
# Sources: HuggingFace model cards + official backend docs
# ==============================================================================
FAMILY_PRESETS = {
    # ── Qwen3.5 (text) ───────────────────────────────────────────────────────
    "qwen3_5": {
        "label": "Qwen3.5 (text)",
        "vllm_extra_args": [
            "--reasoning-parser", "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--kv-cache-dtype", "fp8",
            "--enable-prefix-caching",
        ],
        "llama_extra_args": [
            "--chat-template", "chatml", "--jinja",
            "--flash-attn", "on",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        ],
        "sampling": {
            "temperature": "0.7",
            "top_p": "0.8",
            "top_k": "20",
            "min_p": "0.0",
            "presence_penalty": "1.5",
        },
        "profile_variants": {
            "default":   {"temperature": "0.7", "top_p": "0.8",  "top_k": "20", "presence_penalty": "1.5"},
            "Instruct":  {"temperature": "0.7", "top_p": "0.8",  "top_k": "20"},
            "Reasoning": {"temperature": "0.6", "top_p": "0.95", "top_k": "20"},
            "Code":      {"temperature": "0.6", "top_p": "0.95", "top_k": "20"},
        },
    },
    # ── Qwen3.5-VL / Qwen3-VL (vision) ──────────────────────────────────────
    "qwen3_vl": {
        "label": "Qwen3-VL (vision)",
        # Production lessons baked in:
        # - hermes parser: VL models emit standard Qwen3 tool calls, not the
        #   coder XML dialect (qwen3_coder broke tool calls on Qwythos).
        # - mm-processor-kwargs caps images at ~1448x1448 (~2.6K tokens each
        #   instead of ~16K at the 12MP default) — without it, 3 images blow
        #   any consumer-GPU context and force chunked prefill stalls.
        # - limit-mm-per-prompt guards against runaway multimodal batches.
        # - NO prefix caching: it misbehaves with vision token invalidation.
        "vllm_extra_args": [
            "--reasoning-parser", "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--kv-cache-dtype", "fp8",
            "--mm-encoder-tp-mode", "data",
            "--generation-config", "vllm",
            "--mm-processor-kwargs", '{"max_pixels": 2097152, "min_pixels": 3136}',
            "--limit-mm-per-prompt", '{"image": 5, "video": 1}',
            "--async-scheduling",
            "--disable-custom-all-reduce",
        ],
        "llama_extra_args": [
            "--chat-template", "chatml", "--jinja",
            "--flash-attn", "on",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        ],
        "sampling": {
            "temperature": "0.7",
            "top_p": "0.8",
            "top_k": "20",
            "min_p": "0.0",
            "presence_penalty": "1.5",
        },
        "profile_variants": {
            "default": {"temperature": "0.7", "top_p": "0.8",  "top_k": "20", "presence_penalty": "1.5"},
            "Caption": {"temperature": "0.3", "top_p": "0.9",  "top_k": "20"},
        },
    },
    # ── Qwen3 (MoE / A3B) ────────────────────────────────────────────────────
    "qwen3_moe": {
        "label": "Qwen3 MoE",
        "vllm_extra_args": [
            "--reasoning-parser", "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--kv-cache-dtype", "fp8",
            "--enable-prefix-caching",
        ],
        "llama_extra_args": [
            "--chat-template", "chatml", "--jinja",
            "--flash-attn", "on",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        ],
        "sampling": {
            "temperature": "0.7",
            "top_p": "0.8",
            "top_k": "20",
            "min_p": "0.0",
        },
        "profile_variants": {
            "default":   {"temperature": "0.7", "top_p": "0.8",  "top_k": "20"},
            "Reasoning": {"temperature": "0.6", "top_p": "0.95", "top_k": "20"},
        },
    },
    # ── DeepSeek-R1 / V3 ──────────────────────────────────────────────────────
    "deepseek": {
        "label": "DeepSeek R1/V3",
        "vllm_extra_args": [
            "--reasoning-parser", "deepseek_r1",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ],
        "llama_extra_args": ["--chat-template", "deepseek", "--jinja"],
        "sampling": {
            "temperature": "0.6",
            "top_p": "0.95",
            "top_k": "40",
            "min_p": "0.0",
        },
        "profile_variants": {
            "default":   {"temperature": "0.6", "top_p": "0.95", "top_k": "40"},
            "Reasoning": {"temperature": "0.5", "top_p": "0.95", "top_k": "20"},
        },
    },
    # ── Llama 3.x ────────────────────────────────────────────────────────────
    "llama": {
        "label": "Llama 3.x",
        "vllm_extra_args": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "llama3_json",
        ],
        "llama_extra_args": [],
        "sampling": {
            "temperature": "0.7",
            "top_p": "0.9",
            "top_k": "40",
            "min_p": "0.0",
        },
        "profile_variants": {
            "default": {"temperature": "0.7", "top_p": "0.9", "top_k": "40"},
            "Code":    {"temperature": "0.3", "top_p": "0.9", "top_k": "20"},
        },
    },
    # ── Mistral ───────────────────────────────────────────────────────────────
    "mistral": {
        "label": "Mistral / Mixtral",
        "vllm_extra_args": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "mistral",
        ],
        "llama_extra_args": [],
        "sampling": {
            "temperature": "0.7",
            "top_p": "0.9",
            "top_k": "40",
            "min_p": "0.0",
        },
        "profile_variants": {
            "default": {"temperature": "0.7", "top_p": "0.9", "top_k": "40"},
        },
    },
    # ── Gemma ─────────────────────────────────────────────────────────────────
    "gemma": {
        "label": "Gemma",
        "vllm_extra_args": [],
        "llama_extra_args": [],
        "sampling": {
            "temperature": "1.0",
            "top_p": "0.95",
            "top_k": "64",
            "min_p": "0.0",
        },
        "profile_variants": {
            "default": {"temperature": "1.0", "top_p": "0.95", "top_k": "64"},
        },
    },
    # ── Phi-4 / Phi-3 ─────────────────────────────────────────────────────────
    "phi": {
        "label": "Microsoft Phi",
        "vllm_extra_args": [
            "--trust-remote-code",
        ],
        "llama_extra_args": ["--jinja"],
        "sampling": {
            "temperature": "0.0",
            "top_p": "1.0",
            "top_k": "0",
            "min_p": "0.0",
        },
        "profile_variants": {
            "default": {"temperature": "0.0", "top_p": "1.0"},
        },
    },
    # ── Generic (fallback) ────────────────────────────────────────────────────
    "generic": {
        "label": "Generic",
        "vllm_extra_args": [],
        "llama_extra_args": [],
        "sampling": {
            "temperature": "0.7",
            "top_p": "0.9",
            "top_k": "40",
            "min_p": "0.0",
        },
        "profile_variants": {
            "default": {"temperature": "0.7", "top_p": "0.9", "top_k": "40"},
        },
    },
}

# Mapping: model_type / architecture → family
ARCH_TO_FAMILY = {
    # Qwen3.5
    "qwen3_5":                          "qwen3_5",
    "qwen3_5forcausallm":               "qwen3_5",
    "qwen3_5forconditionalgeneration":  "qwen3_5",
    # Qwen3 VL
    "qwen3_vl":                         "qwen3_vl",
    "qwen3vlforconditionalgeneration":  "qwen3_vl",
    # Qwen2 VL (compatible)
    "qwen2_vl":                         "qwen3_vl",
    "qwen2vlforconditionalgeneration":  "qwen3_vl",
    # Qwen3 / Qwen2 MoE
    "qwen3moeforconditionalgeneration": "qwen3_moe",
    "qwen3_moe":                        "qwen3_moe",
    "qwen2moeforconditionalgeneration": "qwen3_moe",
    # Qwen3 text (no MoE, no VL)
    "qwen3forcausallm":                 "qwen3_5",
    "qwen3":                            "qwen3_5",
    # Qwen2
    "qwen2forcausallm":                 "qwen3_5",
    "qwen2":                            "qwen3_5",
    # DeepSeek
    "deepseekv3forcausallm":            "deepseek",
    "deepseek_v3":                      "deepseek",
    "deepseekr1":                       "deepseek",
    "deepseek_r1":                      "deepseek",
    "deepseekv2forcausallm":            "deepseek",
    # Llama
    "llamaforcausallm":                 "llama",
    "llama":                            "llama",
    "llama3":                           "llama",
    # Mistral
    "mistralformcausallm":              "mistral",
    "mistral":                          "mistral",
    "mixtralformcausallm":              "mistral",
    "mixtral":                          "mistral",
    # Gemma
    "gemmaforcausallm":                 "gemma",
    "gemma2forcausallm":                "gemma",
    "gemma3forcausallm":                "gemma",
    "gemma":                            "gemma",
    # Phi
    "phiforcausallm":                   "phi",
    "phi3forcausallm":                  "phi",
    "phi4forcausallm":                  "phi",
    "phi":                              "phi",
}


# ==============================================================================
# GPU detection via nvidia-smi
# ==============================================================================
def get_gpus() -> list[dict]:
    """Returns list of GPUs with index, name, total_gb and free_gb."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                gpus.append({
                    "index":    int(parts[0]),
                    "name":     parts[1],
                    "total_gb": int(parts[2]) / 1024,
                    "free_gb":  int(parts[3]) / 1024,
                })
        return gpus
    except Exception:
        return []


# ==============================================================================
# Model detection
# ==============================================================================
def detect_model(path: Path) -> dict:
    """Analyse the model directory and return a dict with detected information."""
    info = {
        "path":         str(path),
        "backend":      None,
        "family":       "generic",
        "model_type":   None,
        "architectures": [],
        "max_ctx":      None,
        "has_vision":   False,
        # True when the chat template opts into a chain-of-thought mode
        # (Qwen3 `enable_thinking`, DeepSeek-R1 `<think>` tags, Gemma-flash
        # `reasoning_effort`, ...). Callers use this to pick a lower-
        # temperature preset — reasoning models fall apart with T=1.0.
        "has_reasoning": False,
        # True when the model ships a Multi-Token-Prediction head — enables
        # speculative decoding via draft-mtp on llama.cpp / mtp on vLLM.
        "has_mtp":      False,
        "gguf_files":   [],
        "mmproj_files": [],
        "size_gb":      0.0,
    }

    # ── Backend: GGUF → llama.cpp, safetensors → vLLM ──────────────────────
    gguf_files   = sorted(path.glob("*.gguf"))
    sf_files     = list(path.rglob("*.safetensors"))

    info["gguf_files"]   = [str(f) for f in gguf_files]
    info["mmproj_files"] = [str(f) for f in gguf_files if "mmproj" in f.name.lower()]

    if gguf_files:
        info["backend"] = "llama.cpp"
        # Size: largest .gguf that is not mmproj
        model_ggufs = [f for f in gguf_files if "mmproj" not in f.name.lower()]
        if model_ggufs:
            info["size_gb"] = model_ggufs[0].stat().st_size / (1024 ** 3)
        else:
            info["size_gb"] = gguf_files[0].stat().st_size / (1024 ** 3)
    elif sf_files:
        info["backend"] = "vllm"
        info["size_gb"] = sum(f.stat().st_size for f in sf_files) / (1024 ** 3)
    else:
        info["backend"] = "vllm"  # assume — no detectable files

    # ── config.json ─────────────────────────────────────────────────────────
    cfg_path = path / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            info["model_type"]    = cfg.get("model_type", "")
            info["architectures"] = cfg.get("architectures", [])
            info["has_vision"]    = "vision_config" in cfg
            # MTP head is either declared in config (mtp_num_hidden_layers,
            # num_nextn_predict_layers) or discoverable in the safetensors
            # index (keys prefixed with `mtp.` / `nextn.`).
            tc = cfg.get("text_config", cfg)
            if tc.get("mtp_num_hidden_layers") or tc.get("num_nextn_predict_layers"):
                info["has_mtp"] = True
            info["max_ctx"] = tc.get("max_position_embeddings")
        except Exception as e:
            print(f"⚠  Could not read config.json: {e}", file=sys.stderr)

    # ── chat template — reasoning-mode heuristic ────────────────────────────
    # A model is a "reasoning" model if its chat template opts into a
    # thinking / chain-of-thought block. Vendor-specific tokens keep
    # changing (Qwen3 `enable_thinking`, DeepSeek `<think>`, Gemma
    # `reasoning_effort`), so we OR every known signal — false positives
    # here just mean a slightly conservative sampling preset.
    _REASONING_HINTS = ("enable_thinking", "<think>", "</think>",
                        "reasoning_effort", "<|reasoning|>")
    for template_name in ("chat_template.jinja", "chat_template_v2.jinja"):
        t = path / template_name
        if t.exists():
            try:
                blob = t.read_text(errors="replace")
                if any(h in blob for h in _REASONING_HINTS):
                    info["has_reasoning"] = True
                    break
            except Exception:
                pass
    # tokenizer_config.json embeds the template as a string in many
    # HF-native repos — check there too when a separate .jinja isn't shipped.
    if not info["has_reasoning"]:
        tok_cfg = path / "tokenizer_config.json"
        if tok_cfg.exists():
            try:
                blob = tok_cfg.read_text(errors="replace")
                if any(h in blob for h in _REASONING_HINTS):
                    info["has_reasoning"] = True
            except Exception:
                pass

    # ── Family — tries model_type, then architectures, then folder name ──
    candidates = [info["model_type"] or ""] + [a.lower() for a in info["architectures"]]
    for c in candidates:
        key = c.lower().replace("-", "_").replace(" ", "_")
        if key in ARCH_TO_FAMILY:
            info["family"] = ARCH_TO_FAMILY[key]
            break

    # Fallback: infer family from folder name / .gguf filename
    if info["family"] == "generic":
        name_lower = path.name.lower()
        gguf_names = " ".join(Path(f).stem.lower() for f in info["gguf_files"])
        combined = name_lower + " " + gguf_names

        NAME_PATTERNS = [
            # Qwen3 VL (before generic qwen3)
            (("qwen3.vl", "qwen3vl", "qwen3-vl", "qwen3_vl",
              "qwen3.5vl", "qwen3.5-vl", "qwen3.5_vl"), "qwen3_vl"),
            # Qwen3.5 / Qwen3
            (("qwen3.5", "qwen3-5", "qwen3_5", "qwen3"), "qwen3_5"),
            # Qwen2 VL
            (("qwen2.vl", "qwen2vl", "qwen2-vl"), "qwen3_vl"),
            # Qwen2
            (("qwen2",), "qwen3_5"),
            # DeepSeek
            (("deepseek-r1", "deepseek_r1", "deepseek-v3", "deepseek_v3", "deepseek"), "deepseek"),
            # Llama
            (("llama-3", "llama3", "llama-2", "llama2", "llama"), "llama"),
            # Mistral / Mixtral
            (("mixtral", "mistral"), "mistral"),
            # Gemma
            (("gemma",), "gemma"),
            # Phi
            (("phi-4", "phi-3", "phi4", "phi3", "phi"), "phi"),
        ]
        for patterns, family in NAME_PATTERNS:
            if any(p in combined for p in patterns):
                info["family"] = family
                break

    return info


# ==============================================================================
# VRAM math (shared with core/gpu.py)
# ==============================================================================
def calc_model_size_gb(model_path: str) -> float:
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


def estimate_kv_cache_gb(model_path: str, max_model_len: int, kv_dtype: str = "auto") -> float:
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


# ==============================================================================
# Suggested tensor_parallel and max_model_len
# ==============================================================================
def suggest_backend(info: dict, platform_info: dict | None = None) -> str:
    """Pick the backend for a freshly-detected model.

    Rules:
      - Explicit gguf → llama.cpp (only backend that can load it)
      - CPU-only host → llama.cpp (vLLM requires CUDA/ROCm)
      - Otherwise honour the file format's natural home (safetensors → vllm)
    """
    if info.get("gguf_files"):
        return "llama.cpp"
    if platform_info is None:
        try:
            platform_info = detect_platform()
        except Exception:
            platform_info = {"accelerator": "cpu"}
    if platform_info.get("accelerator") == "cpu":
        return "llama.cpp"
    return info.get("backend") or "vllm"


def suggest_tp(size_gb: float, gpus: list) -> int:
    """Suggest minimum tensor_parallel to fit the model across available GPUs."""
    if not gpus:
        return 1
    # Estimate required VRAM: size * 1.15 (activation overhead + basic KV cache)
    need = size_gb * 1.15
    total_free = sum(g["free_gb"] for g in gpus)
    max_single  = max(g["free_gb"] for g in gpus)

    if max_single >= need:
        return 1
    # How many GPUs do we need?
    n_gpus = len(gpus)
    for tp in [2, 4, 8]:
        if tp > n_gpus:
            break
        # For vLLM TP, split evenly — estimate each GPU gets size/tp
        per_gpu = need / tp
        if max_single >= per_gpu:
            return tp
    return n_gpus  # use all


def suggest_max_len(max_ctx: int | None, size_gb: float, tp: int, gpus: list,
                    model_path: str | None = None, kv_dtype: str = "fp8") -> int:
    """Suggest the largest max_model_len that fits the target GPUs' free VRAM.

    When the model ships a config.json (safetensors), the real KV geometry is
    read and the context is solved from the VRAM left after weights. GGUF and
    unknown layouts fall back to conservative tier caps.
    """
    native = max_ctx or 131072

    def _tier_cap() -> int:
        total_free = sum(g["free_gb"] for g in gpus) if gpus else 0
        if total_free < 32:
            cap = 65536
        elif total_free < 64:
            cap = 131072
        else:
            cap = 262144
        return min(native, cap)

    # CPU-only: KV cache lives in system RAM, which is much scarcer than the
    # aggregate multi-GPU budgets the tier caps assume. Cap much lower so a
    # first-time user on a laptop doesn't try to allocate 16 GB of KV cache.
    if not gpus:
        return min(native, 8192)

    if not model_path or not (Path(model_path) / "config.json").exists():
        return _tier_cap()

    # Budget on the tp GPUs with most free VRAM: vLLM claims
    # gpu_memory_utilization (0.90) of each; weights + load/runtime overhead
    # come out first, the KV cache lives in what remains.
    ranked = sorted(gpus, key=lambda g: g["free_gb"], reverse=True)[:max(1, tp)]
    budget = sum(g["free_gb"] for g in ranked) * 0.90 - size_gb * 1.05 - 1.5
    if budget <= 0:
        return min(native, 8192)

    # KV cost grows (at worst) linearly with context — probe a reference
    # length and solve, then walk down if the exact estimate still overshoots
    # (sliding-window models are cheaper at scale, so this stays conservative).
    ref = 32768
    kv_ref = estimate_kv_cache_gb(model_path, ref, kv_dtype)
    if kv_ref <= 0:
        return _tier_cap()
    ctx = int(budget / kv_ref * ref)
    ctx = min(native, max(8192, (ctx // 4096) * 4096))
    while ctx > 8192 and estimate_kv_cache_gb(model_path, ctx, kv_dtype) > budget:
        ctx -= 4096
    return ctx




# ==============================================================================
# Name/dir heuristics (shared with the TUI model table)
# ==============================================================================
def quant_of(text: str) -> str:
    m = re.search(r"(Q\d_K_[SML]|Q\d_\d|IQ\d\w*|Q\d|FP8|FP16|BF16|AWQ|GPTQ)", text, re.I)
    return m.group(1).upper() if m else "—"


def params_of(text: str) -> str:
    m = re.search(r"(\d+(?:\.\d+)?)\s*[Bb](?![a-z0-9])", text)
    return f"{m.group(1)}B" if m else "—"


def is_moe(text: str) -> bool:
    """MoE sparse models carry the active-params marker in the name (e.g. 35B-A3B)."""
    return bool(re.search(r"a\d+(\.\d+)?b", text.lower()))


def detect_vision(model_dir: Path, cfg: dict | None = None) -> bool:
    """Vision by actual capability: config.json beats filename heuristics.

    Multimodal safetensors models (e.g. Qwen3.6) embed vision without any
    mmproj file — the presence of mmproj is only a llama.cpp packaging detail.
    """
    if cfg and cfg.get("mmproj"):
        return True
    try:
        c = json.loads((model_dir / "config.json").read_text())
        if "vision_config" in c:
            return True
        blob = (str(c.get("model_type", "")) + " "
                + " ".join(c.get("architectures", []))).lower()
        if "vl" in blob or "vision" in blob:
            return True
    except Exception:
        pass
    if list(model_dir.glob("*mmproj*")):
        return True
    name = model_dir.name.lower()
    return "vl" in name.replace("vllm", "") or "vision" in name


# ==============================================================================
# Platform + backend availability probe (multi-platform readiness)
# ==============================================================================
import platform as _platform
import shutil as _shutil


def detect_platform() -> dict:
    """Identify the host and its accelerator without assuming anything.

    Returns a dict with:
        os              — "linux", "darwin", "windows"
        arch            — machine architecture ("x86_64", "arm64", …)
        accelerator     — "cuda" | "rocm" | "metal" | "cpu"
        gpu_vendor      — "nvidia" | "amd" | "apple" | None
        driver_version  — vendor driver version string, or ""
        runtime_version — CUDA / ROCm / Metal version string, or ""

    Every branch swallows its own errors: probing for a tool that isn't
    installed must never raise. Downstream code can check ``accelerator``
    and decide whether to attempt a GPU load at all.
    """
    info = {
        "os": _platform.system().lower(),
        "arch": _platform.machine().lower(),
        "accelerator": "cpu",
        "gpu_vendor": None,
        "driver_version": "",
        "runtime_version": "",
    }

    # NVIDIA (Linux, Windows)
    if _shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                out = r.stdout
                info["accelerator"] = "cuda"
                info["gpu_vendor"] = "nvidia"
                m = re.search(r"Driver Version:\s+([0-9.]+)", out)
                if m:
                    info["driver_version"] = m.group(1)
                m = re.search(r"CUDA Version:\s+([0-9.]+)", out)
                if m:
                    info["runtime_version"] = m.group(1)
                return info
        except Exception:
            pass

    # AMD / ROCm (Linux)
    if _shutil.which("rocm-smi") or _shutil.which("rocminfo"):
        try:
            r = subprocess.run(["rocm-smi", "--showdriverversion"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                info["accelerator"] = "rocm"
                info["gpu_vendor"] = "amd"
                m = re.search(r"Driver Version:\s*([0-9.]+)", r.stdout)
                if m:
                    info["driver_version"] = m.group(1)
                return info
        except Exception:
            info["accelerator"] = "rocm"
            info["gpu_vendor"] = "amd"
            return info

    # Apple Silicon (macOS on arm64 = Metal-capable)
    if info["os"] == "darwin" and info["arch"] == "arm64":
        info["accelerator"] = "metal"
        info["gpu_vendor"] = "apple"
        return info

    return info


def detect_backends() -> dict:
    """Which inference backends this machine can actually run right now.

    Returns per-backend:
        available — bool
        path      — command/binary or python module, or ""
        version   — best-effort version string, or ""
    """
    out: dict[str, dict] = {"vllm": {}, "llama_cpp": {}}

    # vLLM: either the CLI shim or the python package suffices.
    vllm_path = _shutil.which("vllm") or ""
    vllm_version = ""
    if not vllm_path:
        try:
            import importlib.util
            if importlib.util.find_spec("vllm") is not None:
                vllm_path = "python-module:vllm"
        except Exception:
            pass
    if vllm_path:
        try:
            r = subprocess.run([_sys := _shutil.which("python") or "python", "-c",
                                "import vllm; print(vllm.__version__)"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                vllm_version = r.stdout.strip()
        except Exception:
            pass
    out["vllm"] = {"available": bool(vllm_path), "path": vllm_path, "version": vllm_version}

    # llama.cpp: the llama-server binary is what allma actually spawns.
    llama_path = _shutil.which("llama-server") or ""
    if not llama_path:
        for candidate in ("/usr/local/bin/llama-server",
                          str(Path.home() / "AI/llama.cpp/build/bin/llama-server")):
            if Path(candidate).exists():
                llama_path = candidate
                break
    llama_version = ""
    if llama_path:
        try:
            r = subprocess.run([llama_path, "--version"],
                               capture_output=True, text=True, timeout=5)
            blob = (r.stdout + r.stderr).strip()
            m = re.search(r"version:\s*(\S+)", blob) or re.search(r"b\d{3,5}", blob)
            if m:
                llama_version = m.group(0) if m.re.pattern.startswith("b") else m.group(1)
        except Exception:
            pass
    out["llama_cpp"] = {"available": bool(llama_path), "path": llama_path, "version": llama_version}
    return out
