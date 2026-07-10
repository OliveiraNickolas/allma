#!/usr/bin/env python3
"""
Allma Config Creator — generates .allm files for downloaded models.

Usage:
    python create_config.py /path/to/model
    python create_config.py /path/to/model --name MyModel
    python create_config.py /path/to/model --yes   # accept all defaults
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


# ==============================================================================
# Terminal colors
# ==============================================================================
C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_DIM    = "\033[2m"
C_GREEN  = "\033[92m"
C_YELLOW = "\033[93m"
C_CYAN   = "\033[96m"
C_RED    = "\033[91m"
C_BLUE   = "\033[94m"


def bold(s):    return f"{C_BOLD}{s}{C_RESET}"
def green(s):   return f"{C_GREEN}{s}{C_RESET}"
def yellow(s):  return f"{C_YELLOW}{s}{C_RESET}"
def cyan(s):    return f"{C_CYAN}{s}{C_RESET}"
def red(s):     return f"{C_RED}{s}{C_RESET}"
def dim(s):     return f"{C_DIM}{s}{C_RESET}"


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
def get_gpus():
    """Returns list of GPUs with index, total_gb and free_gb."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                gpus.append({
                    "index":    int(parts[0]),
                    "total_gb": int(parts[1]) / 1024,
                    "free_gb":  int(parts[2]) / 1024,
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

            # max_position_embeddings may be in text_config (VL models)
            tc = cfg.get("text_config", cfg)
            info["max_ctx"] = tc.get("max_position_embeddings")
        except Exception as e:
            print(yellow(f"⚠  Could not read config.json: {e}"))

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
# Suggested tensor_parallel and max_model_len
# ==============================================================================
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


def suggest_max_len(max_ctx: int | None, size_gb: float, tp: int, gpus: list) -> int:
    """Suggest a conservative max_model_len based on native context and available VRAM."""
    native = max_ctx or 131072

    # Conservative cap: above 256k it's rare to have enough VRAM for KV cache
    # Simplified formula: KV cache ≈ layers * heads * ctx * 4 bytes
    # For simplicity: cap at 128k if < 32GB total, 256k if >= 32GB
    total_free = sum(g["free_gb"] for g in gpus) if gpus else 0
    if total_free < 32:
        cap = 65536
    elif total_free < 64:
        cap = 131072
    else:
        cap = 262144

    return min(native, cap)


# ==============================================================================
# .allm file generation
# ==============================================================================
def render_flag_lines(args: list) -> list[str]:
    """Convert a flat ['--flag', 'value', '--bool-flag', ...] list into
    .allm v2 lines: 'flag value' / 'bool-flag' (leading dashes dropped —
    the parser re-adds them)."""
    lines: list[str] = []
    i = 0
    while i < len(args):
        tok = str(args[i])
        if tok.startswith("-"):
            flag = tok.lstrip("-")
            # value = next token unless it's another flag
            if i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
                lines.append(f"{flag} {args[i + 1]}")
                i += 2
            else:
                lines.append(flag)
                i += 1
        else:
            # stray value without a flag — keep as-is
            lines.append(tok)
            i += 1
    return lines


def generate_base_allm(
    name: str,
    info: dict,
    preset: dict,
    tp: int,
    max_len: int,
    gguf_path: str | None,
    mmproj_path: str | None,
) -> str:
    """Generate the content of a base .allm file (v2 syntax)."""
    lines = [f"# Base model: {name} ({info['backend']} backend)"]

    if info["backend"] == "vllm":
        lines += [
            "@vllm",
            f"@path {info['path']}",
            "",
            f"tensor-parallel-size {tp}",
            f"max-model-len {max_len}",
            "max-num-seqs 8",
            "gpu-memory-utilization 0.90",
        ]
        lines += render_flag_lines(preset.get("vllm_extra_args", []))

    else:  # llama.cpp
        lines += [
            "@llamacpp",
            f"@path {gguf_path}",
            "",
            "-ngl -1",
            f"-c {min(max_len, 40960)}",
            "-b 1024",
            "-t 16",
        ]
        if mmproj_path:
            lines.append(f"--mmproj {mmproj_path}")
        lines += render_flag_lines(preset.get("llama_extra_args", []))
        # MTP-head GGUFs (name carries "MTP"): enable speculative decoding.
        # n_max=3 measured ~2x decode speed vs 6 (low draft acceptance
        # wastes compute at higher depths).
        if gguf_path and "mtp" in Path(gguf_path).name.lower():
            lines += ["spec-type draft-mtp", "spec-draft-n-max 3"]

    return "\n".join(lines) + "\n"


def generate_profile_allm(
    profile_name: str,
    base_name: str,
    sampling: dict,
) -> str:
    """Generate the content of a profile .allm file (v2 syntax)."""
    lines = [
        f"# Profile: {profile_name}",
        f"@name {profile_name}",
        f"@base {base_name}",
    ]
    # Convention: profiles named *Instruct* run without chain-of-thought.
    if "instruct" in profile_name.lower():
        lines.append("@thinking-off")
    lines.append("")
    for k, v in sampling.items():
        lines.append(f"{k.replace('_', '-')} {v}")
    return "\n".join(lines) + "\n"


# ==============================================================================
# Interactive prompt utilities
# ==============================================================================
def ask(prompt: str, default: str, auto: bool) -> str:
    """Ask the user with a default. If auto=True, use the default."""
    if auto:
        print(f"  {prompt}: {green(default)}")
        return default
    val = input(f"  {prompt} [{cyan(default)}]: ").strip()
    return val if val else default


def ask_int(prompt: str, default: int, auto: bool) -> int:
    """Ask the user for an integer with validation."""
    default_str = str(default)
    if auto:
        print(f"  {prompt}: {green(default_str)}")
        return default
    while True:
        val = input(f"  {prompt} [{cyan(default_str)}]: ").strip()
        if not val:
            return default
        try:
            return int(val)
        except ValueError:
            print(f"    {red('✗ Invalid integer')} — try again")


def ask_list(prompt: str, default: list, auto: bool) -> list:
    """Ask for a JSON list. If empty, use the default."""
    default_str = json.dumps(default)
    raw = ask(prompt + " (JSON)", default_str, auto)
    try:
        result = json.loads(raw)
        return result if isinstance(result, list) else default
    except json.JSONDecodeError:
        print(yellow("  ⚠  Invalid JSON — using default"))
        return default


def ask_yes(prompt: str, auto: bool) -> bool:
    if auto:
        print(f"  {prompt}: {green('y')}")
        return True
    val = input(f"  {prompt} [Y/n]: ").strip().lower()
    return val not in ("n", "no")


def pick_gguf(gguf_files: list, auto: bool) -> str | None:
    """Ask the user to choose the main .gguf file."""
    model_ggufs = [f for f in gguf_files if "mmproj" not in Path(f).name.lower()]
    if not model_ggufs:
        return None
    if len(model_ggufs) == 1:
        print(f"  GGUF detected: {green(model_ggufs[0])}")
        return model_ggufs[0]

    print(bold("\n  Multiple GGUFs found — choose the main model file:"))
    for i, f in enumerate(model_ggufs):
        size = Path(f).stat().st_size / (1024 ** 3)
        print(f"    {cyan(str(i))} — {Path(f).name}  {dim(f'{size:.1f}GB')}")
    if auto:
        print(f"  → Using {green(model_ggufs[0])}")
        return model_ggufs[0]
    idx = input(f"  Index [0]: ").strip()
    try:
        return model_ggufs[int(idx)] if idx else model_ggufs[0]
    except (ValueError, IndexError):
        return model_ggufs[0]


# ==============================================================================
# Main flow
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Allma .allm config generator for downloaded models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
            Examples:
              python create_config.py /path/to/Qwen3.5-9b
              python create_config.py /path/to/model --name MyModel --yes
        """),
    )
    parser.add_argument("model_path", help="Path to the downloaded model directory")
    parser.add_argument("--name", "-n", help="Base name for the configs (e.g. Qwen3.5-9b)")
    parser.add_argument("--yes",  "-y", action="store_true", help="Accept all defaults without prompting")
    parser.add_argument("--config-dir", default="configs", help="Allma configs directory (default: configs)")
    args = parser.parse_args()

    auto = args.yes
    config_dir = Path(args.config_dir)
    model_path = Path(args.model_path).resolve()

    if not model_path.exists():
        print(red(f"❌  Directory not found: {model_path}"))
        sys.exit(1)

    print(bold(f"\n{'═'*60}"))
    print(bold(f"  Allma Config Creator"))
    print(bold(f"{'═'*60}"))
    print(f"  Path: {cyan(str(model_path))}")

    # ── Detect model ─────────────────────────────────────────────────────────
    info = detect_model(model_path)
    gpus = get_gpus()

    print(f"\n{bold('📦 Auto-detection:')}")
    print(f"  Backend    : {green(info['backend'])}")
    print(f"  Family     : {green(FAMILY_PRESETS[info['family']]['label'])}")
    print(f"  model_type : {dim(info['model_type'] or 'not detected')}")
    print(f"  Context    : {dim(str(info['max_ctx']) if info['max_ctx'] else 'not detected')}")
    size_str = f"{info['size_gb']:.1f}GB"
    print(f"  Size       : {dim(size_str)}")
    print(f"  Vision     : {dim('yes' if info['has_vision'] else 'no')}")
    if gpus:
        gpu_str = ", ".join(f"GPU{g['index']} {g['free_gb']:.0f}GB free" for g in gpus)
        print(f"  GPUs       : {dim(gpu_str)}")

    preset = FAMILY_PRESETS[info["family"]]
    tp_suggested   = suggest_tp(info["size_gb"], gpus) if info["backend"] == "vllm" else 1
    len_suggested  = suggest_max_len(info["max_ctx"], info["size_gb"], tp_suggested, gpus)

    # ── Physical name ─────────────────────────────────────────────────────────
    default_name = args.name or model_path.name
    # Strip problematic characters
    default_name = default_name.replace("/", "-").replace(" ", "-")

    print(f"\n{bold('⚙️  Physical model configuration:')}")
    phys_name = ask("Physical name (file in configs/base/)", default_name, auto)

    # ── Backend ──────────────────────────────────────────────────────────────
    backend = ask("Backend (vllm / llama.cpp)", info["backend"], auto)

    # ── GGUF ─────────────────────────────────────────────────────────────────
    gguf_path   = None
    mmproj_path = None
    if backend == "llama.cpp":
        gguf_path = pick_gguf(info["gguf_files"], auto)
        if not gguf_path:
            gguf_path = ask(".gguf file path", "", auto)
        if info["mmproj_files"]:
            print(f"  mmproj detected: {green(info['mmproj_files'][0])}")
            mmproj_path = info["mmproj_files"][0]
            if not auto:
                custom = input(f"  mmproj [{cyan(mmproj_path)}]: ").strip()
                if custom:
                    mmproj_path = custom

    # ── Tensor parallel / max_model_len ──────────────────────────────────────
    if backend == "vllm":
        tp       = ask_int("tensor_parallel", tp_suggested, auto)
        max_len  = ask_int("max_model_len", len_suggested, auto)
        gpu_util = ask("gpu_memory_utilization", "0.90", auto)
        max_seqs = ask("max_num_seqs", "8", auto)

        extra_args_default = preset.get("vllm_extra_args", [])
        extra_args = ask_list("extra_args vLLM", extra_args_default, auto)
    else:
        tp = 1
        max_len = ask_int("n_ctx", min(len_suggested, 40960), auto)
        n_threads = ask("n_threads", "16", auto)
        extra_args_default = preset.get("llama_extra_args", [])
        extra_args = ask_list("extra_args llama.cpp", extra_args_default, auto)

    # Override preset with user values
    preset_copy = dict(preset)
    if backend == "vllm":
        preset_copy["vllm_extra_args"] = extra_args
    else:
        preset_copy["llama_extra_args"] = extra_args

    # ── Generate base.allm ────────────────────────────────────────────────
    phys_content = generate_base_allm(
        name=phys_name,
        info={**info, "backend": backend},
        preset=preset_copy,
        tp=tp,
        max_len=max_len,
        gguf_path=gguf_path,
        mmproj_path=mmproj_path,
    )
    if backend == "vllm":
        # Replace custom values
        phys_content = phys_content.replace(
            'gpu_memory_utilization = "0.90"',
            f'gpu_memory_utilization = "{gpu_util}"',
        ).replace(
            'max_num_seqs = "8"',
            f'max_num_seqs = "{max_seqs}"',
        )
    else:
        phys_content = phys_content.replace(
            'n_threads = "16"',
            f'n_threads = "{n_threads}"',
        )

    # ── Profile variants ──────────────────────────────────────────────────────
    print(f"\n{bold('🧩 Profile models (sampling):')}")
    variants = preset.get("profile_variants", {"default": preset["sampling"]})

    # Derive logical base name from physical name (e.g. "Qwen3.5-9b" → "Qwen3.5:9b")
    import re
    m = re.search(r"-(\d+\.?\d*[bBmM])", phys_name)
    profile_base = phys_name[:m.start()] + ":" + phys_name[m.start() + 1:] if m else phys_name

    profile_configs = []
    for variant_key, variant_sampling in variants.items():
        if variant_key == "default":
            default_profile = profile_base
        else:
            default_profile = f"{profile_base}-{variant_key}"

        print(f"\n  {bold(f'Variant: {variant_key}')}")
        log_name = ask("  Profile name", default_profile, auto)

        # Sampling
        sampling = dict(preset["sampling"])
        sampling.update(variant_sampling)
        print(f"  Suggested sampling: {dim(str(sampling))}")
        if not auto:
            print("  Press Enter to accept, or edit field by field:")
            for k, v in list(sampling.items()):
                new_v = input(f"    {k} [{cyan(str(v))}]: ").strip()
                if new_v:
                    sampling[k] = new_v

        log_content = generate_profile_allm(log_name, phys_name, sampling)
        profile_configs.append((log_name, log_content))

    # ── Preview ───────────────────────────────────────────────────────────────
    print(f"\n{bold('📄 Preview — base:')}")
    for line in phys_content.splitlines():
        print(f"  {dim(line)}")

    for log_name, log_content in profile_configs:
        print(f"\n{bold(f'📄 Preview — profile ({log_name}):')}")
        for line in log_content.splitlines():
            print(f"  {dim(line)}")

    # ── Confirm and write ─────────────────────────────────────────────────────
    print()
    if not ask_yes("Write files?", auto):
        print(yellow("  Cancelled."))
        sys.exit(0)

    phys_dir = config_dir / "base"
    log_dir  = config_dir / "profile"
    phys_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    phys_file = phys_dir / f"{phys_name}.allm"
    if phys_file.exists() and not auto:
        if not ask_yes(f"  {phys_file} already exists — overwrite?", auto):
            print(yellow("  Base config not overwritten."))
        else:
            phys_file.write_text(phys_content)
            print(green(f"  ✔ {phys_file}"))
    else:
        phys_file.write_text(phys_content)
        print(green(f"  ✔ {phys_file}"))

    for log_name, log_content in profile_configs:
        log_file = log_dir / f"{log_name.replace(':', '-')}.allm"
        if log_file.exists() and not auto:
            if not ask_yes(f"  {log_file} already exists — overwrite?", auto):
                print(yellow(f"  Profile '{log_name}' not overwritten."))
                continue
        log_file.write_text(log_content)
        print(green(f"  ✔ {log_file}"))

    print(f"\n{green(bold('✅ Done!'))} Restart Allma to load the new configs.\n")


if __name__ == "__main__":
    main()
