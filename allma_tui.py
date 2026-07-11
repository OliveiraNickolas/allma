#!/usr/bin/env python3
"""
Allma TUI — 3-column configuration panel (LM Studio style).

  ┌─[ EXPLORE ]──┬─[ MY MODELS ]──────────────┬─[ SETUP ]──────────────┐
  │ filters      │ models table               │ LOAD | PROFILES tabs   │
  │ [MODEL INFO] │ 📁 browsable folder        │ sliders · extra args   │
  │ [DOWNLOAD]   │                            │                        │
  └──────────────┴────────────────────────────┴────────────────────────┘

LOAD  = base model config (.allm in configs/base/)
PROFILES = sampling/variants (.allm in configs/profile/)
"Load once" sends ephemeral overrides via /v1/load — nothing touches disk.
"""
import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from rich.markup import escape as rescape
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import (
    Button, Collapsible, DataTable, DirectoryTree, Input, Select, Static,
    TabbedContent, TabPane,
)

# ──────────────────────────────────────────────────────────────────────────────
# Paths / constants
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "configs"
STATE_FILE = BASE_DIR / "cache" / "tui_state.json"
ALLMA_URL  = f"http://127.0.0.1:{os.environ.get('ALLMA_PORT', '9000')}"

# Keep huggingface_hub from printing progress bars over the TUI
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from create_config import FAMILY_PRESETS, detect_model  # noqa: E402

# Accent palette — the ALLMA logo rainbow (C64/Apple II retro style)
ACC_RED    = "#e52529"
ACC_ORANGE = "#f7941d"
ACC_YELLOW = "#b89000"   # darkened yellow, readable on the cream background
ACC_GREEN  = "#43b047"
ACC_BLUE   = "#009ddc"
LOGO_COLORS = [ACC_RED, ACC_ORANGE, "#f7d000", ACC_GREEN, ACC_BLUE]


def logo_markup() -> str:
    """▮▮▮▮▮ ALLMA — blocks and letters in the logo colors."""
    blocks = "".join(f"[{c}]▮[/]" for c in LOGO_COLORS)
    word = "".join(f"[bold {c}]{ch}[/]" for ch, c in zip("ALLMA", LOGO_COLORS))
    return f"{blocks}  {word}"


def _load_tui_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def _save_tui_state(state: dict) -> None:
    try:
        STATE_FILE.parent.mkdir(exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


_state = _load_tui_state()
DEFAULT_MODELS_DIR = Path(_state.get("models_dir")
                          or os.environ.get("ALLMA_MODELS_DIR")
                          or str(Path.home() / "AI" / "Models"))

# ──────────────────────────────────────────────────────────────────────────────
# Field specs — sliders: (key, label, min, max, step, is_int, default)
# ──────────────────────────────────────────────────────────────────────────────
LLAMA_SLIDERS = [
    ("n_ctx",        "Context Length",      1024, 262144, 1024, True, 40960),
    ("n_gpu_layers", "GPU Offload (layers)",  -1,     99,    1, True,    -1),
    ("n_batch",      "Batch Size",            64,   4096,   64, True,  1024),
    ("n_ubatch",     "µBatch Size",           64,   2048,   64, True,   512),
    ("n_threads",    "CPU Threads",            1,     32,    1, True,    16),
]
VLLM_SLIDERS = [
    ("max_model_len",          "Context Length",   1024, 262144, 1024, True, 131072),
    ("gpu_memory_utilization", "GPU Mem Util",     0.50,   0.98, 0.01, False,  0.90),
    ("max_num_seqs",           "Max Sequences",       1,     64,    1, True,      8),
    ("max_num_batched_tokens", "Max Batch Tokens", 1024,  65536, 1024, True,  32768),
    ("tensor_parallel",        "Tensor Parallel",     1,      8,    1, True,      1),
]
SAMPLING_SLIDERS = [
    ("temperature",        "Temperature",        0.0, 2.0, 0.05, False, 0.7),
    ("top_p",              "Top-P",              0.0, 1.0, 0.01, False, 0.9),
    ("top_k",              "Top-K",                0, 200,    1, True,   40),
    ("min_p",              "Min-P",              0.0, 1.0, 0.01, False, 0.0),
    ("presence_penalty",   "Presence Penalty",  -2.0, 2.0, 0.05, False, 0.0),
    ("repetition_penalty", "Repetition Penalty", 0.8, 1.5, 0.01, False, 1.0),
]

# Extra-args catalog per backend: (flag, default_args, description for the (i)).
# The list shows the name without "--"; serialization keeps the original spelling.
LLAMA_EXTRA_CATALOG = [
    ("--jinja",          [],
     "Use the Jinja chat template embedded in the GGUF (recommended for tool calling and new models)."),
    ("--flash-attn",     ["on"],
     "Optimized attention: less VRAM and faster prefill. Values: on/off/auto."),
    ("--cache-type-k",   ["q8_0"],
     "Quantize the K half of the KV cache. Values: f16, q8_0, q4_0… Saves VRAM for longer context."),
    ("--cache-type-v",   ["q8_0"],
     "Quantize the V half of the KV cache (requires flash-attn). Values: f16, q8_0, q4_0…"),
    ("--no-kv-offload",  [],
     "Keep the KV cache in RAM instead of VRAM. Frees VRAM but slows down generation."),
    ("--no-mmap",        [],
     "Load the whole model into RAM instead of memory-mapping the file. Slower start, steadier access."),
    ("--mlock",          [],
     "Prevent the OS from swapping the model out (LM Studio's 'Keep Model in Memory')."),
    ("--cont-batching",  [],
     "Process multiple requests in parallel on the same model (better multi-client throughput)."),
    ("--parallel",       ["4"],
     "Number of simultaneous request slots (LM Studio's 'Max Concurrent Predictions')."),
    ("--kv-unified",     [],
     "Unified KV cache across parallel slots (LM Studio's 'Unified KV Cache')."),
    ("--seed",           ["-1"],
     "Random generator seed. -1 = random on every start (LM Studio's 'Seed')."),
    ("--rope-freq-base", ["0"],
     "RoPE frequency base. 0 = use the model's value (LM Studio's 'RoPE Frequency Base')."),
    ("--rope-freq-scale", ["1.0"],
     "RoPE frequency scale for stretching context. 1.0 = no scaling ('RoPE Frequency Scale')."),
    ("--rope-scaling",   ["yarn"],
     "Context extension method: none, linear or yarn (YaRN stretches beyond native context)."),
    ("--cache-reuse",    ["256"],
     "Reuse KV cache chunks across similar prompts (min. tokens for reuse via shifting)."),
    ("--no-warmup",      [],
     "Skip the warm-up inference at startup. Loads faster; first response is a bit slower."),
    ("--no-webui",       [],
     "Disable llama-server's built-in web UI (API only)."),
    ("--metrics",        [],
     "Expose performance metrics at /metrics in Prometheus format."),
    ("--embeddings",     [],
     "Enable the /embeddings endpoint to generate embedding vectors with the model."),
    ("--split-mode",     ["row"],
     "How to split the model across GPUs: none, layer (default) or row."),
    ("--defrag-thold",   ["0.1"],
     "Defragment the KV cache when fragmentation exceeds the threshold (0.1 = 10%)."),
    ("--threads-batch",  ["16"],
     "CPU threads for batch/prompt processing (defaults to --threads when unset)."),
    ("--n-cpu-moe",      ["0"],
     "Keep the N most-used MoE expert layers on GPU, the rest on CPU. Fit a big MoE in less VRAM."),
    ("--override-tensor", ["exps=CPU"],
     "Force specific tensors onto a device by regex (e.g. exps=CPU offloads MoE experts to RAM)."),
    ("--keep",           ["-1"],
     "Tokens from the start of the prompt to always keep when the context fills. -1 = keep all."),
    ("--ctx-shift",      [],
     "Slide the context window when it fills instead of stopping (infinite-generation mode)."),
    ("--repeat-penalty", ["1.0"],
     "Server-side default repetition penalty (per-request values override it)."),
    ("--temp",           ["0.7"],
     "Server-side default sampling temperature (per-request values override it)."),
    ("--spec-type",      ["draft-mtp"],
     "Speculative decoding type (e.g. draft-mtp for models with MTP heads)."),
    ("--spec-draft-n-max", ["3"],
     "Max draft tokens per speculative decoding step."),
    ("--model-draft",    [],
     "Path to a small draft GGUF for speculative decoding (external-draft alternative to MTP)."),
    ("--draft-max",      ["16"],
     "Max tokens the external draft model proposes per step (with --model-draft)."),
    ("--device",         [],
     "Restrict to specific backend devices (e.g. CUDA0,CUDA1). Overrides auto GPU selection."),
]
VLLM_EXTRA_CATALOG = [
    ("--reasoning-parser",         ["qwen3"],
     "Split reasoning content (<think>) from the final text. Value: parser name (qwen3)."),
    ("--enable-auto-tool-choice",  [],
     "Let the model decide on its own when to call tools (required for tool calling)."),
    ("--tool-call-parser",         ["qwen3_coder"],
     "Parser that extracts tool calls from generated text. Value: model format (qwen3_coder)."),
    ("--enable-prefix-caching",    [],
     "Reuse the KV cache of repeated prefixes across requests (great for long system prompts)."),
    ("--kv-cache-dtype",           ["fp8"],
     "Store the KV cache in FP8: ~half the cache VRAM, allowing more context/sequences."),
    ("--enable-chunked-prefill",   [],
     "Split long prompts into chunks interleaved with generation. Better latency under load."),
    ("--enforce-eager",            [],
     "Disable CUDA graphs. Less VRAM and faster startup, slightly slower generation."),
    ("--max-cudagraph-capture-size", ["8"],
     "Cap CUDA-graph capture batch size. Small values (8) cut graph VRAM a lot; needs eager OFF."),
    ("--quantization",             ["fp8"],
     "Quantize a full-precision model on load (fp8, awq, gptq…). Halves weight VRAM on BF16 models."),
    ("--mm-processor-kwargs",      ['{"max_pixels": 2097152, "min_pixels": 3136}'],
     "Vision preprocessing. max_pixels caps image resolution (2097152 ≈ 1448²) so images don't blow the context."),
    ("--limit-mm-per-prompt",      ['{"image": 5, "video": 1}'],
     "Cap how many images/videos a single prompt may carry (JSON)."),
    ("--mm-encoder-tp-mode",       ["data"],
     "How the vision encoder is sharded under tensor parallel. 'data' is usually fastest."),
    ("--speculative-config",       ['{"method": "mtp", "num_speculative_tokens": 1}'],
     "Speculative decoding config (JSON). MTP models: {\"method\":\"mtp\",\"num_speculative_tokens\":1}."),
    ("--generation-config",        ["vllm"],
     "Where default sampling comes from: 'vllm' ignores the model's generation_config.json."),
    ("--async-scheduling",         [],
     "Overlap scheduling with GPU work. Lower latency for streaming; slight complexity cost."),
    ("--trust-remote-code",        [],
     "Allow custom Python code from the model repository to run (required by some models)."),
    ("--disable-log-requests",     [],
     "Don't log request contents (smaller logs and more privacy)."),
    ("--swap-space",               ["4"],
     "Reserve N GB of RAM as KV-cache 'swap' for preempted sequences."),
    ("--cpu-offload-gb",           ["4"],
     "Offload N GB of weights to RAM — run a model bigger than VRAM, at a speed cost."),
    ("--seed",                     ["0"],
     "Random generator seed for the server."),
    ("--disable-custom-all-reduce", [],
     "Use NCCL's standard all-reduce for tensor parallel (more compatible, sometimes slower)."),
    ("--disable-sliding-window",   [],
     "Disable sliding-window attention, using full attention over the whole context."),
]

# Flags with a fixed set of valid values → rendered as a drop list
FLAG_CHOICES = {
    "--flash-attn":     ["on", "off", "auto"],
    "--cache-type-k":   ["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"],
    "--cache-type-v":   ["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"],
    "--rope-scaling":   ["none", "linear", "yarn"],
    "--split-mode":     ["none", "layer", "row"],
    "--kv-cache-dtype": ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
    "--quantization":   ["fp8", "awq", "gptq", "bitsandbytes"],
    "--mm-encoder-tp-mode": ["data", "weights"],
}

# Human-readable dropdowns for flags whose real value is fiddly JSON. Each entry
# is (label shown to the user, actual value string). The FlagRow renders the
# labels in a Select and maps back to the value on save — the user never types
# JSON. First entry is the default when the flag is switched on.
FLAG_PRESETS = {
    "--mm-processor-kwargs": [
        ("balanced ~1448px", '{"max_pixels": 2097152, "min_pixels": 3136}'),
        ("low VRAM ~1024px", '{"max_pixels": 1048576, "min_pixels": 3136}'),
        ("hi-detail ~2048px", '{"max_pixels": 4194304, "min_pixels": 3136}'),
        ("max ~2560px", '{"max_pixels": 6553600, "min_pixels": 3136}'),
    ],
    "--limit-mm-per-prompt": [
        ("5 img · 1 vid", '{"image": 5, "video": 1}'),
        ("3 img · 0 vid", '{"image": 3, "video": 0}'),
        ("10 img · 1 vid", '{"image": 10, "video": 1}'),
        ("1 img only", '{"image": 1, "video": 0}'),
    ],
    "--speculative-config": [
        ("MTP · 1 token", '{"method": "mtp", "num_speculative_tokens": 1}'),
        ("MTP · 2 tokens", '{"method": "mtp", "num_speculative_tokens": 2}'),
    ],
}

# extra_args flags that duplicate base-config fields: absorbed into the sliders
# (and written back as proper .allm fields on save)
LLAMA_FIELD_ALIASES = {
    "--ubatch-size": "n_ubatch", "-ub": "n_ubatch",
    "--batch-size":  "n_batch",  "-b":  "n_batch",
    "--ctx-size":    "n_ctx",    "-c":  "n_ctx",
    "--threads":     "n_threads", "-t": "n_threads",
    "--n-gpu-layers": "n_gpu_layers", "-ngl": "n_gpu_layers",
    "--gpu-layers":  "n_gpu_layers",
}


def absorb_field_aliases(leftover: list, aliases: dict) -> tuple[dict, list]:
    """Pull `--flag value` pairs that duplicate config fields out of the leftovers.

    Returns ({cfg_key: value}, remaining_tokens)."""
    absorbed: dict[str, str] = {}
    rest: list[str] = []
    i = 0
    while i < len(leftover):
        tok = leftover[i]
        if tok in aliases and i + 1 < len(leftover) and not leftover[i + 1].startswith("-"):
            absorbed[aliases[tok]] = leftover[i + 1]
            i += 2
        else:
            rest.append(tok)
            i += 1
    return absorbed, rest


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _as_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "on")
    return bool(v)


def _http(method: str, path: str, body: dict | None = None, timeout: float = 8.0):
    """Returns (ok, json). Never raises."""
    try:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"{ALLMA_URL}{path}", data=data, method=method)
        req.add_header("User-Agent", "AllmaTUI/2.0")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return False, json.loads(e.read() or b"{}")
        except Exception:
            return False, {"error": f"HTTP {e.code}"}
    except Exception as e:
        return False, {"error": str(e)}


def get_gpus() -> list[dict]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
        gpus = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                gpus.append({"index": int(parts[0]), "name": parts[1],
                             "total_gb": int(parts[2]) / 1024,
                             "free_gb": int(parts[3]) / 1024})
        return gpus
    except Exception:
        return []


def _parse_allm(f: Path) -> dict:
    from configs.allm_parser import parse_allm
    return parse_allm(f.read_text(), f.name)


def _quant_of(text: str) -> str:
    m = re.search(r"(Q\d_K_[SML]|Q\d_\d|IQ\d\w*|Q\d|FP8|FP16|BF16|AWQ|GPTQ)", text, re.I)
    return m.group(1).upper() if m else "—"


def _params_of(text: str) -> str:
    m = re.search(r"(\d+(?:\.\d+)?)\s*[Bb](?![a-z0-9])", text)
    return f"{m.group(1)}B" if m else "—"


def _is_moe(text: str) -> bool:
    """MoE sparse models carry the active-params marker in the name (e.g. 35B-A3B)."""
    return bool(re.search(r"a\d+(\.\d+)?b", text.lower()))


def _detect_vision(model_dir: Path, cfg: dict | None = None) -> bool:
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


def scan_models(models_dir: Path) -> list[dict]:
    """Unified list: each item is a configured base OR an unconfigured directory."""
    models: list[dict] = []
    known_paths: set[str] = set()

    base_dir = CONFIG_DIR / "base"
    if base_dir.exists():
        for f in sorted(base_dir.glob("*.allm")):
            try:
                cfg = _parse_allm(f)
            except Exception:
                continue
            if "backend" not in cfg:
                continue
            backend = cfg.get("backend", "vllm")
            path = cfg.get("path") or cfg.get("model", "")
            if path:
                known_paths.add(str(Path(path)))
                known_paths.add(str(Path(path).parent))  # gguf: mark the directory
            size_gb = 0.0
            p = Path(path) if path else None
            if p and p.exists():
                size_gb = (p.stat().st_size / 1024**3 if p.is_file() else
                           sum(x.stat().st_size for x in p.rglob("*.safetensors")) / 1024**3)
            model_dir = (p.parent if p and p.is_file() else p) if p else None
            name_all = f.stem + " " + Path(path).name
            models.append({
                "key": f.stem, "configured": True, "backend": backend,
                "format": "gguf" if backend == "llama.cpp" else "safetensors",
                "path": path, "file": f, "cfg": cfg, "size_gb": size_gb,
                "quant": _quant_of(name_all), "params": _params_of(name_all),
                "vision": _detect_vision(model_dir, cfg) if model_dir else False,
                "moe": _is_moe(name_all),
                "mtp": "mtp" in f.stem.lower(),
            })

    if models_dir.exists():
        for d in sorted(models_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            if str(d) in known_paths:
                continue
            ggufs = [g for g in d.glob("*.gguf") if "mmproj" not in g.name.lower()]
            if ggufs and all(str(g) in known_paths for g in ggufs):
                continue
            has_st = any(d.rglob("*.safetensors"))
            if not ggufs and not has_st:
                continue
            size_gb = (max(g.stat().st_size for g in ggufs) / 1024**3 if ggufs
                       else sum(x.stat().st_size for x in d.rglob("*.safetensors")) / 1024**3)
            name_all = d.name + " " + " ".join(g.name for g in ggufs[:1])
            models.append({
                "key": d.name, "configured": False,
                "backend": "llama.cpp" if ggufs else "vllm",
                "format": "gguf" if ggufs else "safetensors",
                "path": str(d), "file": None, "cfg": {}, "size_gb": size_gb,
                "quant": _quant_of(name_all), "params": _params_of(name_all),
                "vision": _detect_vision(d),
                "moe": _is_moe(name_all),
                "mtp": "mtp" in d.name.lower(),
            })
    return models


def scan_profiles() -> list[dict]:
    out = []
    prof_dir = CONFIG_DIR / "profile"
    if prof_dir.exists():
        for f in sorted(prof_dir.glob("*.allm")):
            try:
                cfg = _parse_allm(f)
            except Exception:
                continue
            if "base" in cfg:
                out.append({"name": cfg.get("name") or f.stem, "base": cfg["base"],
                            "cfg": cfg, "file": f})
    return out


# ── .allm writing (line-based; multi-line arrays handled explicitly) ──────────
def _strip_allm_key(lines: list[str], section: str | None, key: str) -> list[str]:
    """Remove a key (including multi-line arrays) from a list of lines."""
    out, current, skipping_array = [], None, False
    for line in lines:
        s = line.strip()
        if skipping_array:
            if s.rstrip().endswith("]"):
                skipping_array = False
            continue
        if s.startswith("[") and s.endswith("]") and "=" not in s:
            current = s[1:-1].strip()
        if current == section and "=" in s and not s.startswith("#"):
            if s.split("=", 1)[0].strip() == key:
                val = s.split("=", 1)[1].strip()
                if val.startswith("[") and not val.rstrip().endswith("]"):
                    skipping_array = True
                continue
        out.append(line)
    return out


def update_allm_param(path: Path, section: str | None, key: str, value) -> None:
    """Update/insert key = value in an .allm file, respecting sections and arrays."""
    lines = _strip_allm_key(path.read_text(encoding="utf-8").splitlines(), section, key)
    if section is None:
        # insert before the first section header (or at the end)
        idx = next((i for i, l in enumerate(lines)
                    if l.strip().startswith("[") and l.strip().endswith("]")
                    and "=" not in l), len(lines))
        lines.insert(idx, f"{key} = {value}")
    else:
        try:
            idx = next(i for i, l in enumerate(lines) if l.strip() == f"[{section}]")
            lines.insert(idx + 1, f"{key} = {value}")
        except StopIteration:
            lines += ["", f"[{section}]", f"{key} = {value}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def remove_allm_param(path: Path, section: str | None, key: str) -> None:
    lines = _strip_allm_key(path.read_text(encoding="utf-8").splitlines(), section, key)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── extra_args: parse/serialize against the catalog ───────────────────────────
def parse_extra_args(args: list, catalog: list) -> tuple[dict, list]:
    """Returns ({flag: current_args}, leftovers). Consumes each catalog flag and
    the following tokens that don't start with '--' as its arguments."""
    enabled: dict[str, list] = {}
    leftover: list[str] = []
    flags = {c[0] for c in catalog}
    i = 0
    args = [str(a) for a in (args or [])]
    while i < len(args):
        tok = args[i]
        if tok in flags:
            vals = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                vals.append(args[i])
                i += 1
            enabled[tok] = vals
        else:
            leftover.append(tok)
            i += 1
    return enabled, leftover


def _split_flag_value(raw: str) -> list:
    """Split a flag's value string into tokens WITHOUT mangling JSON.

    shlex.split() strips the quotes inside {"max_pixels": ...} and shatters
    it on spaces — the exact corruption that turned a valid config into
    `{max_pixels:`. When the value looks like JSON (starts with { or [),
    keep it as a single token; otherwise fall back to shlex."""
    raw = raw.strip()
    if raw.startswith("{") or raw.startswith("["):
        return [raw]
    try:
        return shlex.split(raw)
    except ValueError:
        return [raw]


def serialize_extra_args(enabled: dict, custom: str) -> list:
    out: list[str] = []
    for flag, vals in enabled.items():
        out.append(flag)
        out.extend(vals)
    out.extend(_split_flag_value(custom) if custom else [])
    return out


def _fmt_args_list(args: list) -> str:
    return "[" + ", ".join(f'"{a}"' for a in args) + "]"


def _fmt_size(n: int | None) -> str:
    if not n:
        return "?"
    return f"{n / 1024**3:.1f} GB" if n >= 1024**3 else f"{n / 1024**2:.0f} MB"


def _dir_size(d: Path) -> int:
    try:
        return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    except Exception:
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# Custom widgets
# ──────────────────────────────────────────────────────────────────────────────
class Toggle(Static):
    """● on / ○ off (bold text when on).

    show_mark=False hides the dot — used by the EXPLORE filters, where the
    state is the fully colored line (class 'line').
    """
    can_focus = True

    class Changed(Message):
        def __init__(self, toggle: "Toggle", value: bool) -> None:
            super().__init__()
            self.toggle = toggle
            self.value = value

        @property
        def control(self):
            return self.toggle

    def __init__(self, label: str, value: bool = False, show_mark: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self.value = value
        self.show_mark = show_mark

    def render(self):
        text = rescape(self.label)
        if not self.show_mark:
            return f" {text}"
        mark = "[#007878]●[/]" if self.value else "[#6a5a48]○[/]"
        return f" {mark}  {text}"

    def _sync(self):
        self.set_class(self.value, "-on")
        self.refresh()

    def on_mount(self) -> None:
        self._sync()

    def toggle_value(self) -> None:
        self.value = not self.value
        self._sync()
        self.post_message(self.Changed(self, self.value))

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.toggle_value()

    def on_key(self, event: events.Key) -> None:
        if event.key in ("enter", "space"):
            event.stop()
            self.toggle_value()


class RadioRow(Horizontal):
    """Row of exclusive options: ● auto  ○ GPU 0  ○ GPU 1."""

    class Changed(Message):
        def __init__(self, radio: "RadioRow", value: str) -> None:
            super().__init__()
            self.radio = radio
            self.value = value

        @property
        def control(self):
            return self.radio

    def __init__(self, options: list[tuple[str, str]], value: str = "", **kwargs):
        super().__init__(classes="radio-row", **kwargs)
        self._options = options
        if value not in {v for _l, v in options}:
            value = options[0][1] if options else ""
        self.value = value

    def compose(self) -> ComposeResult:
        for label, val in self._options:
            yield Toggle(label, value=(val == self.value), classes="radio-opt")

    def on_toggle_changed(self, event: Toggle.Changed) -> None:
        event.stop()
        toggles = list(self.query(Toggle))
        idx = toggles.index(event.toggle)
        self.value = self._options[idx][1]
        for i, t in enumerate(toggles):
            t.value = (i == idx)
            t._sync()
        self.post_message(self.Changed(self, self.value))


class _InfoMark(Static):
    """Clickable (i) at the end of the line — shows the flag's purpose in a toast."""

    def __init__(self, title: str, desc: str, **kwargs):
        super().__init__("(i)", classes="info-mark", **kwargs)
        self._title = title
        self._desc = desc

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.app.notify(self._desc, title=self._title, timeout=8)


class FlagRow(Horizontal):
    """Extra-args flag row: ● flag-name  [editable value]  (i).

    The name is shown without '--'; serialization keeps the original spelling.
    """

    def __init__(self, flag: str, defaults: list, desc: str, on: bool,
                 vals: list | None, **kwargs):
        super().__init__(classes="xrow", **kwargs)
        self.flag = flag
        self.defaults = defaults
        self._desc = desc
        self._on = on
        self._vals = list(vals) if vals else list(defaults)
        self._has_val = bool(defaults or vals)

    def compose(self) -> ComposeResult:
        yield Toggle(self.flag.lstrip("-"), value=self._on)
        if self._has_val:
            presets = FLAG_PRESETS.get(self.flag)
            choices = FLAG_CHOICES.get(self.flag)
            if presets:
                # Human labels in the dropdown; the raw value is carried on the
                # option value so we can map back on save. Keep an unknown
                # current value visible as its own entry.
                cur = " ".join(self._vals) if self._vals else presets[0][1]
                opts = [(lbl, val) for lbl, val in presets]
                if cur not in {v for _l, v in opts}:
                    opts = [(f"custom: {cur[:24]}…", cur)] + opts
                yield Select(opts, value=cur, allow_blank=False, classes="flag-sel")
            elif choices:
                current = self._vals[0] if self._vals else (self.defaults or [choices[0]])[0]
                opts = list(dict.fromkeys([current] + choices))  # keep unknown current
                yield Select([(c, c) for c in opts], value=current,
                             allow_blank=False, classes="flag-sel")
            else:
                yield Input(value=" ".join(self._vals), classes="flag-val")
        yield _InfoMark(self.flag, self._desc)

    @property
    def enabled(self) -> bool:
        try:
            return self.query_one(Toggle).value
        except Exception:
            return self._on

    def args(self) -> list:
        if not self._has_val:
            return []
        try:
            sel = self.query_one(".flag-sel", Select)
            val = str(sel.value)
            # preset dropdown values may be a whole JSON blob (one token with
            # spaces) — keep it as a single arg; plain choices are one word.
            return [val] if self.flag in FLAG_PRESETS else [val]
        except Exception:
            pass
        try:
            raw = self.query_one(".flag-val", Input).value
            return _split_flag_value(raw) if raw.strip() else list(self.defaults)
        except Exception:
            return self._vals

    def set_state(self, on: bool, vals: list | None = None) -> None:
        t = self.query_one(Toggle)
        t.value = on
        t._sync()
        if vals and self._has_val:
            want = " ".join(vals)
            try:
                sel = self.query_one(".flag-sel", Select)
                known = {v for _l, v in sel._options}
                if want in known:
                    sel.value = want
                elif vals[0] in known:
                    sel.value = vals[0]
                return
            except Exception:
                pass
            try:
                self.query_one(".flag-val", Input).value = want
            except Exception:
                pass

    class Changed(Message):
        """A flag's toggle or value changed — lets the panel refresh anything
        derived from extra_args (e.g. the VRAM breakdown bar)."""
        def __init__(self, row: "FlagRow") -> None:
            super().__init__()
            self.row = row

    def on_input_changed(self, event: Input.Changed) -> None:
        event.stop()
        self.post_message(self.Changed(self))

    def on_select_changed(self, event: Select.Changed) -> None:
        event.stop()
        self.post_message(self.Changed(self))


class _CtrBtn(Static):
    """[-]/[+] mini-button for ParamCounter — plain Static, without the
    min-width/border baggage of Textual Buttons (which overflow 1-cell rows)."""

    def __init__(self, glyph: str, sign: int):
        super().__init__(f" {glyph} ", classes="ctr-btn")
        self._sign = sign

    def on_click(self, event: events.Click) -> None:
        event.stop()
        parent = self.parent
        if isinstance(parent, ParamCounter):
            parent.bump(self._sign)


class ParamCounter(Horizontal):
    """Discrete numeric field: label  [-] N [+] (for small values)."""

    class Changed(Message):
        def __init__(self, pc: "ParamCounter", value) -> None:
            super().__init__()
            self.param = pc
            self.key = pc.key
            self.value = value

        @property
        def control(self):
            return self.param

    def __init__(self, key: str, label: str, vmin, vmax, step, is_int: bool,
                 value, **kwargs):
        super().__init__(classes="param-counter", **kwargs)
        self.key = key
        self.label = label
        self.vmin, self.vmax, self.step, self.is_int = vmin, vmax, step, is_int
        try:
            v = float(str(value))
            self.value = int(v) if is_int else v
        except (TypeError, ValueError):
            self.value = vmin
        self.value = max(vmin, min(vmax, self.value))

    @property
    def value_str(self) -> str:
        return str(int(self.value)) if self.is_int else f"{self.value:g}"

    def compose(self) -> ComposeResult:
        yield Static(self.label, classes="ctr-label")
        yield _CtrBtn("-", -1)
        yield Static(self.value_str, classes="ctr-val")
        yield _CtrBtn("+", +1)

    def set_value(self, v, announce: bool = False) -> None:
        try:
            v = float(str(v))
        except (TypeError, ValueError):
            return
        v = int(v) if self.is_int else v
        self.value = max(self.vmin, min(self.vmax, v))
        try:
            self.query_one(".ctr-val", Static).update(self.value_str)
        except Exception:
            pass
        if announce:
            self.post_message(self.Changed(self, self.value))

    def bump(self, sign: int) -> None:
        self.set_value(self.value + sign * self.step, announce=True)


class SliderBar(Widget):
    """Clickable/draggable ──●── track. Internal to ParamSlider."""
    can_focus = True

    class Moved(Message):
        def __init__(self, bar: "SliderBar", fraction: float) -> None:
            super().__init__()
            self.bar = bar
            self.fraction = fraction

    def __init__(self, fraction: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.fraction = max(0.0, min(1.0, fraction))
        self._dragging = False

    def set_fraction(self, f: float) -> None:
        self.fraction = max(0.0, min(1.0, f))
        self.refresh()

    def render(self):
        w = max(self.size.width, 8)
        knob = round(self.fraction * (w - 1))
        t = Text()
        t.append("─" * knob, style="#007878")
        t.append("●", style="bold #007878")
        t.append("─" * (w - 1 - knob), style="#a89878")
        return t

    def _emit_from_x(self, x: int) -> None:
        w = max(self.size.width, 2)
        self.post_message(self.Moved(self, max(0.0, min(1.0, x / (w - 1)))))

    def on_mouse_down(self, event: events.MouseDown) -> None:
        event.stop()
        self._dragging = True
        self.capture_mouse()
        self._emit_from_x(event.x)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._dragging:
            self._emit_from_x(event.x)

    def on_mouse_up(self, event: events.MouseUp) -> None:
        self._dragging = False
        self.release_mouse()

    def on_key(self, event: events.Key) -> None:
        if event.key in ("left", "right"):
            event.stop()
            delta = -1 if event.key == "left" else 1
            self.post_message(self.Moved(self, None))  # keyboard step signal
            # the real step is resolved by ParamSlider (stores the sign)
            self._key_delta = delta


class ParamSlider(Vertical):
    """Label + editable value + slider, with two-way sync.

    n_gpu_layers uses -1 = 'all' (right end of the track).
    """

    class Changed(Message):
        def __init__(self, ps: "ParamSlider", value) -> None:
            super().__init__()
            self.param = ps
            self.key = ps.key
            self.value = value

        @property
        def control(self):
            return self.param

    def __init__(self, key: str, label: str, vmin, vmax, step, is_int: bool,
                 value, **kwargs):
        super().__init__(classes="param-slider", **kwargs)
        self.key = key
        self.label = label
        self.vmin, self.vmax, self.step, self.is_int = vmin, vmax, step, is_int
        self.value = self._clamp(self._parse(value, vmin))
        self._syncing = False

    # — value helpers —
    def _parse(self, v, fallback):
        try:
            f = float(str(v).strip())
            return int(f) if self.is_int else f
        except (ValueError, TypeError):
            return fallback

    def _clamp(self, v):
        return max(self.vmin, min(self.vmax, v))

    @property
    def value_str(self) -> str:
        if self.is_int:
            return str(int(self.value))
        return f"{self.value:.2f}".rstrip("0").rstrip(".") or "0"

    def _display(self) -> str:
        if self.key == "n_gpu_layers" and int(self.value) == -1:
            return "all"
        return self.value_str

    def _fraction(self) -> float:
        if self.vmax == self.vmin:
            return 0.0
        return (self.value - self.vmin) / (self.vmax - self.vmin)

    # — UI —
    def compose(self) -> ComposeResult:
        with Horizontal(classes="ps-head"):
            yield Static(self.label, classes="ps-label")
            yield Input(value=self._display(), classes="ps-value")
        yield SliderBar(self._fraction(), classes="ps-bar")

    def set_value(self, v, announce: bool = False) -> None:
        self.value = self._clamp(self._parse(v, self.value))
        self._syncing = True
        try:
            self.query_one(".ps-value", Input).value = self._display()
            self.query_one(SliderBar).set_fraction(self._fraction())
        except Exception:
            pass
        self._syncing = False
        if announce:
            self.post_message(self.Changed(self, self.value))

    def on_slider_bar_moved(self, event: SliderBar.Moved) -> None:
        event.stop()
        if event.fraction is None:           # ←/→ key
            delta = getattr(event.bar, "_key_delta", 1) * self.step
            self.set_value(self.value + delta, announce=True)
            return
        raw = self.vmin + event.fraction * (self.vmax - self.vmin)
        stepped = round(raw / self.step) * self.step
        self.set_value(stepped, announce=True)

    def on_input_changed(self, event: Input.Changed) -> None:
        event.stop()
        if self._syncing:
            return
        txt = event.value.strip().lower()
        if self.key == "n_gpu_layers" and txt in ("all", "-1"):
            v = -1
        else:
            v = self._parse(txt, None)
        if v is None:
            return
        v = self._clamp(v)
        if v == self.value:
            # initial mount / no-op typing — don't announce (a spurious Changed
            # from n_gpu_layers would switch off the automatic coupling)
            return
        self.value = v
        self._syncing = True
        try:
            self.query_one(SliderBar).set_fraction(self._fraction())
        finally:
            self._syncing = False
        self.post_message(self.Changed(self, self.value))


class _DirsOnlyTree(DirectoryTree):
    def filter_paths(self, paths):
        return [p for p in paths if p.is_dir() and not p.name.startswith(".")]


class _GgufTree(DirectoryTree):
    def filter_paths(self, paths):
        return [p for p in paths
                if (p.is_dir() and not p.name.startswith(".")) or p.suffix == ".gguf"]


class PickerScreen(ModalScreen):
    """Modal browser: pick a directory (mode='dir') or a .gguf file (mode='gguf')."""
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, root: Path, mode: str = "dir", title: str = "Select"):
        super().__init__()
        self.root = root if root.exists() else Path.home()
        self.mode = mode
        self.picker_title = title
        self.current: Path | None = None

    def compose(self) -> ComposeResult:
        tree_cls = _DirsOnlyTree if self.mode == "dir" else _GgufTree
        with Vertical(id="picker-box") as box:
            box.border_title = f"\\[ {self.picker_title} ]"
            with Horizontal(id="picker-nav"):
                yield Button("⬆ Up", id="picker-up", classes="mini")
                yield Static(rescape(str(self.root)), id="picker-root")
            yield tree_cls(str(self.root), id="picker-tree")
            yield Static("", id="picker-current")
            with Horizontal(classes="btn-row"):
                yield Button("✔ Select", id="picker-ok")
                yield Button("✕ Cancel", id="picker-cancel", classes="warn")

    def _go_up(self) -> None:
        parent = self.root.parent
        if parent == self.root:
            return
        self.root = parent
        tree = self.query_one("#picker-tree", DirectoryTree)
        tree.path = str(parent)
        self.query_one("#picker-root", Static).update(rescape(str(parent)))
        if self.mode == "dir":
            self._set_current(parent)

    def _set_current(self, p: Path) -> None:
        self.current = p
        self.query_one("#picker-current", Static).update(f"▸ {rescape(str(p))}")

    def on_directory_tree_directory_selected(self, ev: DirectoryTree.DirectorySelected) -> None:
        if self.mode == "dir":
            self._set_current(Path(ev.path))

    def on_directory_tree_file_selected(self, ev: DirectoryTree.FileSelected) -> None:
        if self.mode == "gguf":
            self._set_current(Path(ev.path))

    def on_button_pressed(self, ev: Button.Pressed) -> None:
        if ev.button.id == "picker-up":
            self._go_up()
        elif ev.button.id == "picker-ok" and self.current:
            self.dismiss(str(self.current))
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ──────────────────────────────────────────────────────────────────────────────
# CSS — allma cream/teal palette
# ──────────────────────────────────────────────────────────────────────────────
CSS = """
Screen { background: #0a0a08; }
#root  { background: #d0c4a8; height: 100%; }
#topbar {
    height: 3; background: #c8b898; padding: 0 1 0 2;
}
#topbar-status {
    width: 1fr; height: 3;
    color: #007878; text-style: bold;
    content-align: left middle;
}
.top-btn { margin: 1 0 0 1; min-width: 11; }
#columns { height: 1fr; }

.panel {
    border: double #008888;
    border-title-color: #007878;
    border-title-style: bold;
    background: #e8dfc8;
    padding: 0 1;
}
#col-left  { width: 1fr; min-width: 24; max-width: 40; }
#col-mid   { width: 2fr; min-width: 30; }
#col-right { width: 1fr; min-width: 32; max-width: 66; }

/* ── EXPLORE / toggles ── */
#explore { height: auto; }
Toggle {
    height: 1; color: #6a5a48; background: #e8dfc8; padding: 0 1;
}
Toggle:hover  { background: #ddd4b8; }
Toggle:focus  { text-style: underline; }
Toggle.-on    { color: #1a1408; text-style: bold; }
Toggle.line.-on {
    background: #007878; color: #f0e8d0; text-style: bold;
}
.radio-row { height: 1; margin: 0 2 0 0; }
.radio-row Toggle { width: auto; padding: 0 2 0 1; }
.xrow { height: 1; margin: 0 2 0 0; }
.xrow Toggle { width: 1fr; }
.flag-val {
    width: 14; height: 1; border: none;
    background: #f0e8d0; color: #007878; padding: 0 1;
}
/* border: none on focus too — the global Input:focus rule adds a border and
   inflates the 1-row field to 3 rows, covering the line below */
.flag-val:focus { background: #fff8e0; border: none; }
.flag-sel { width: 14; height: 1; margin: 0; }
.flag-sel SelectCurrent {
    border: none; height: 1; padding: 0 1;
    background: #f0e8d0; color: #007878;
}
.flag-sel SelectCurrent Static { background: #f0e8d0; color: #007878; }
/* border: none on focus too — the global Select:focus rule would inflate the
   1-row field to 3 rows (same trap as Input:focus) */
.flag-sel:focus SelectCurrent { border: none; background: #fff8e0; }
.flag-sel:focus SelectCurrent Static { background: #fff8e0; }
/* overlay is wider than the collapsed field so long preset labels fit on
   one line; it floats above the row so the extra width doesn't reflow it */
.flag-sel SelectOverlay { width: 24; height: auto; max-height: 12; }
.info-mark { width: 4; height: 1; color: #6a5a48; }
.info-mark:hover { color: #007878; text-style: bold; }
.param-counter { height: 1; margin: 0 2 1 0; }
.ctr-label { width: 1fr; color: #1a1408; }
.ctr-btn {
    width: 3; height: 1; padding: 0;
    background: #c8b898; color: #007878;
    text-align: center; text-style: bold;
}
.ctr-btn:hover { background: #007878; color: #f0e8d0; }
.ctr-val { width: 7; height: 1; text-align: center; color: #007878; text-style: bold; }

/* ── MODEL INFO / DOWNLOAD ── */
#model-info {
    height: 1fr; color: #1a1408;
    scrollbar-color: #007878; scrollbar-background: #c8b898;
    scrollbar-size: 1 1;
}
#download { height: auto; }
#dl-url { height: 3; }
.dl-row { height: 3; margin: 1 0 0 0; }
#dl-pick { margin: 1 0 0 0; }

/* ── MY MODELS ── */
#models-table {
    background: #e8dfc8; color: #1a1408; border: none; height: 1fr;
    scrollbar-color: #007878; scrollbar-background: #c8b898;
    scrollbar-size: 1 1;
}
#models-table > .datatable--header {
    background: #c8b898; color: #007878; text-style: bold;
}
#models-table > .datatable--cursor { background: #007878; color: #f0e8d0; }
#models-table > .datatable--even-row { background: #ddd4b8; }
#models-footer {
    height: 1; background: #c8b898; color: #007878; padding: 0 1;
}
#models-footer:hover { background: #b8a888; text-style: bold; }

/* ── SETUP ── */
TabbedContent { height: 1fr; }
Tabs { background: #c8b898; }
Tab  {
    width: 1fr;
    color: #6a5a48;
    content-align: center middle;
    text-align: center;
    margin: 0;
}
Tab.-active { color: #007878; text-style: bold; }
/* no right padding — the scrollbar hugs the panel border */
TabPane { background: #e8dfc8; padding: 0 0 0 1; }
.form-scroll {
    height: 1fr;
    scrollbar-color: #007878; scrollbar-background: #c8b898;
    scrollbar-size: 1 1;
}
/* content↔scrollbar gap goes on each row type's right margin — a broad margin
   rule would override the vertical margins (margin is one Spacing per rule) */
.section-hdr {
    color: #007878; text-style: bold;
    border-bottom: solid #008888; margin: 1 2 0 0;
}
.field-hint { color: #6a5a48; margin: 0 2 0 0; }
.vram-line  { color: #1a1408; margin: 0 2 1 0; }

/* sliders */
.param-slider { height: 3; margin: 0 2 0 0; }
.ps-head  { height: 1; }
.ps-label { width: 1fr; color: #1a1408; }
.ps-value {
    width: 12; height: 1; border: none;
    background: #f0e8d0; color: #007878; text-style: bold;
    padding: 0 1;
}
.ps-value:focus { background: #fff8e0; border: none; }
.ps-bar { height: 1; margin: 0 0 1 0; }
.ps-bar:focus { background: #ddd4b8; }

Input { background: #f0e8d0; color: #007878; border: solid #008888; height: 3; }
Input:focus { border: solid #007878; }
Input > .input--cursor { background: #007878; color: #f0e8d0; }
Input > .input--selection { background: #00a0a0; color: #f0e8d0; }
Select { background: #e8dfc8; }
Select SelectCurrent {
    background: #f0e8d0; color: #007878; border: solid #008888;
}
Select SelectCurrent Static { color: #007878; background: #f0e8d0; }
Select:focus SelectCurrent { border: solid #007878; }
.form-scroll > Select { margin: 0 2 0 0; }
SelectOverlay {
    background: #f0e8d0; color: #1a1408; border: solid #008888;
    scrollbar-color: #007878; scrollbar-background: #c8b898;
}
SelectOverlay > .option-list--option-highlighted { background: #007878; color: #f0e8d0; }
Collapsible {
    background: #e8dfc8; border: solid #008888; margin: 0 2 1 0; padding: 0;
}
CollapsibleTitle { color: #007878; text-style: bold; }
/* slim chip-style buttons: 1 row, no fat border */
Button {
    background: #007878; color: #f0e8d0;
    border: none; height: 1;
    min-width: 8; padding: 0 2;
    text-style: bold; margin: 0 1 0 0;
}
Button:hover { background: #7a1818; }
Button.warn  { background: #7a1818; }
Button.mini  { min-width: 4; }
.btn-row { height: 1; margin: 1 2 1 0; }
.mmproj-row { height: 3; margin: 0 2 0 0; }
.mmproj-row Input { width: 1fr; }
.mmproj-row Button { margin: 1 0 0 0; }
#statusline { height: 1; background: #c8b898; color: #007878; padding: 0 2; }

/* picker modal */
PickerScreen { align: center middle; }
#picker-box {
    width: 96; height: 34;
    border: double #008888;
    border-title-color: #007878;
    border-title-style: bold;
    background: #e8dfc8; padding: 1;
}
#picker-nav { height: 1; margin: 0 0 1 0; }
#picker-nav Button { min-width: 8; }
#picker-root { height: 1; content-align: left middle; color: #6a5a48; padding: 0 1; }
#picker-tree {
    height: 1fr; background: #e8dfc8; color: #1a1408;
    scrollbar-color: #007878; scrollbar-background: #c8b898;
    scrollbar-size: 1 1;
}
#picker-tree > .tree--cursor { background: #007878; color: #f0e0d0; }
#picker-current { height: 1; color: #007878; text-style: bold; }
"""


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────
class AllmaTUI(App):
    CSS = CSS
    TITLE = "Allma"
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("f5", "rescan", "Rescan"),
        Binding("ctrl+l", "load_one_time", "Load once"),
        Binding("ctrl+s", "save_base", "Save base"),
    ]

    FILTERS = [
        ("gguf",   "GGUF",         True),
        ("st",     "Safetensors",  True),
        ("vision", "Vision",       False),
        ("moe",    "MoE",          False),
        ("mtp",    "MTP",          False),
        ("loaded", "Loaded",       False),
        ("new",    "Not configured", False),
    ]

    def __init__(self):
        super().__init__()
        self._rebuild_lock = asyncio.Lock()
        self.models_dir = DEFAULT_MODELS_DIR
        self.models: list[dict] = []
        self.profiles: list[dict] = []
        self.selected: dict | None = None
        self.loaded_status: dict[str, dict] = {}
        self.server_online = False
        self._detect_cache: dict[str, dict] = {}
        self.gpus = get_gpus()
        # automatic coupling: turns off once the user touches the target manually
        self._auto_layers = True     # llama: n_ctx → n_gpu_layers
        self._auto_seqs = True       # vllm:  max_model_len → max_num_seqs
        self._kv_budget: float | None = None
        # download state
        self._dl_repo: str | None = None
        self._dl_files: dict | None = None
        self._dl_busy = False
        self._dl_dest: Path | None = None
        self._dl_expected = 0
        self._dl_timer = None

    # ── layout ────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        with Vertical(id="root"):
            with Horizontal(id="topbar"):
                yield Static("", id="topbar-status")
                yield Button("▶ Load", id="btn-top-load", classes="top-btn")
                yield Button("⟳ Server", id="btn-top-reload", classes="top-btn")
            with Horizontal(id="columns"):
                with Vertical(id="col-left"):
                    with Vertical(id="explore", classes="panel"):
                        for fid, label, default in self.FILTERS:
                            yield Toggle(label, value=default, show_mark=False,
                                         id=f"flt-{fid}", classes="line")
                    with ScrollableContainer(id="model-info", classes="panel"):
                        yield Static("Select a model…", id="info-text")
                    with Vertical(id="download", classes="panel"):
                        yield Input(placeholder="HF repo-id / URL / .gguf link",
                                    id="dl-url")
                        with Horizontal(classes="btn-row dl-row"):
                            yield Button("⇣ Fetch", id="btn-dl")
                with Vertical(id="col-mid", classes="panel"):
                    yield DataTable(id="models-table", cursor_type="row", zebra_stripes=True)
                    yield Static("", id="models-footer")
                with Vertical(id="col-right", classes="panel"):
                    with TabbedContent(id="setup-tabs"):
                        with TabPane("LOAD", id="tab-load"):
                            yield ScrollableContainer(id="load-form", classes="form-scroll")
                        with TabPane("PROFILES", id="tab-profiles"):
                            yield ScrollableContainer(id="profiles-form", classes="form-scroll")
            yield Static("", id="statusline")

    def on_mount(self) -> None:
        # column titles (colors set in CSS for contrast with the background)
        self.query_one("#explore").border_title = "\\[ EXPLORE ]"
        self.query_one("#model-info").border_title = "\\[ MODEL INFO ]"
        self.query_one("#download").border_title = "\\[ DOWNLOAD ]"
        self.query_one("#col-mid").border_title = "\\[ MY MODELS ]"
        self.query_one("#col-right").border_title = "\\[ SETUP ]"
        self._update_footer()
        table = self.query_one("#models-table", DataTable)
        table.add_columns(" ", "Model", "Backend", "Params", "Quant", "Size")
        self.action_rescan()
        table.focus()
        self.set_interval(3.0, self._poll_server)
        self._poll_server()

    def _update_footer(self) -> None:
        self.query_one("#models-footer", Static).update(
            f"📁 {rescape(str(self.models_dir))}  [#6a5a48](click to change)[/]"
        )

    # ── server status ─────────────────────────────────────────────────────
    @work(thread=True, exclusive=True, group="poll")
    def _poll_server(self) -> None:
        ok, _ = _http("GET", "/health", timeout=2)
        loaded: dict[str, dict] = {}
        if ok:
            ok2, ps = _http("GET", "/v1/ps", timeout=3)
            if ok2:
                for s in ps.get("servers", []):
                    if s.get("alive"):
                        loaded[s["name"]] = s
        gpus = get_gpus()
        self.app.call_from_thread(self._apply_server_status, ok, loaded, gpus)

    def _apply_server_status(self, online: bool, loaded: dict, gpus: list) -> None:
        changed = (online != self.server_online) or (set(loaded) != set(self.loaded_status))
        self.server_online = online
        self.loaded_status = loaded
        self.gpus = gpus or self.gpus
        srv = (f"[{ACC_GREEN}]● online[/]" if online
               else f"[{ACC_RED}]○ offline — run `allma serve`[/]")
        gpu_txt = "  ·  ".join(
            f"[{ACC_BLUE}]GPU{g['index']}[/] {g['free_gb']:.0f}/{g['total_gb']:.0f}G free"
            for g in self.gpus) or "no GPU"
        n_color = ACC_ORANGE if loaded else "#6a5a48"
        self.query_one("#topbar-status", Static).update(
            f"{logo_markup()}   ·   server {srv}   ·   "
            f"[{n_color}]{len(loaded)} loaded[/]   ·   {gpu_txt}"
        )
        if changed:
            self._refresh_table()

    # ── scan / table ──────────────────────────────────────────────────────
    def action_rescan(self) -> None:
        self.models = scan_models(self.models_dir)
        self.profiles = scan_profiles()
        self._refresh_table()
        self._status(f"{len(self.models)} models · {len(self.profiles)} profiles")

    def _filter_state(self) -> dict:
        return {fid: self.query_one(f"#flt-{fid}", Toggle).value
                for fid, _l, _d in self.FILTERS}

    def _refresh_table(self) -> None:
        f = self._filter_state()
        table = self.query_one("#models-table", DataTable)
        prev_key = self.selected["key"] if self.selected else None
        table.clear()
        for m in self.models:
            if m["format"] == "gguf" and not f["gguf"]:
                continue
            if m["format"] == "safetensors" and not f["st"]:
                continue
            if f["vision"] and not m["vision"]:
                continue
            if f["moe"] and not m["moe"]:
                continue
            if f["mtp"] and not m["mtp"]:
                continue
            if f["loaded"] and m["key"] not in self.loaded_status:
                continue
            if f["new"] and m["configured"]:
                continue
            if m["key"] in self.loaded_status:
                dot = (f"[{ACC_ORANGE}]✱[/]" if self.loaded_status[m["key"]].get("custom_load")
                       else f"[{ACC_GREEN}]●[/]")
            elif not m["configured"]:
                dot = f"[{ACC_YELLOW}]+[/]"
            else:
                dot = " "
            be_color = ACC_GREEN if m["backend"] == "llama.cpp" else ACC_BLUE
            table.add_row(
                dot, m["key"],
                f"[{be_color}]{m['backend']}[/]",
                f"[{ACC_BLUE}]{m['params']}[/]",
                f"[{ACC_ORANGE}]{m['quant']}[/]",
                f"{m['size_gb']:.1f}G",
                key=m["key"],
            )
        if prev_key is not None:
            try:
                table.move_cursor(row=table.get_row_index(prev_key))
            except Exception:
                pass

    def on_toggle_changed(self, event: Toggle.Changed) -> None:
        tid = event.toggle.id or ""
        if tid.startswith("flt-"):
            self._refresh_table()
            return
        # A FlagRow toggle flips extra_args → VRAM costs change (mtp,
        # cudagraph, kv dtype...). Walk up to detect it.
        w = event.toggle
        while w is not None:
            if isinstance(w, FlagRow):
                if self.selected:
                    self._update_vram_line(self.selected)
                return
            w = w.parent

    def on_flag_row_changed(self, event: "FlagRow.Changed") -> None:
        # Flag VALUE edited (Select/Input inside the row) — e.g. changing
        # --cache-type-k from q8_0 to q4_0 halves the KV estimate.
        if self.selected:
            self._update_vram_line(self.selected)

    async def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is None:
            return
        key = event.row_key.value
        model = next((m for m in self.models if m["key"] == key), None)
        if model and model is not self.selected:
            self.selected = model
            await self._show_model(model)

    # footer click → change models folder
    def on_click(self, event: events.Click) -> None:
        w = event.widget
        while w is not None:
            if getattr(w, "id", None) == "models-footer":
                self._pick_models_dir()
                return
            w = w.parent

    def _pick_models_dir(self) -> None:
        def _done(path: str | None) -> None:
            if path:
                self.models_dir = Path(path)
                _state["models_dir"] = path
                _save_tui_state(_state)
                self._update_footer()
                self.action_rescan()
        self.push_screen(PickerScreen(self.models_dir, "dir", "Models folder"), _done)

    # ── selection → panels ────────────────────────────────────────────────
    async def _show_model(self, m: dict) -> None:
        async with self._rebuild_lock:
            self._auto_layers = True
            self._auto_seqs = True
            self._kv_budget = None
            self._render_info(m, detecting=True)
            await self._build_load_form(m)
            await self._build_profiles_form(m)
        self._detect_async(m)

    def _detect_path(self, m: dict) -> str:
        return m["path"] if (m["backend"] == "vllm" or not m["configured"]) \
            else str(Path(m["path"]).parent)

    @work(thread=True, exclusive=True, group="detect")
    def _detect_async(self, m: dict) -> None:
        path = self._detect_path(m)
        if path not in self._detect_cache:
            try:
                self._detect_cache[path] = detect_model(Path(path))
            except Exception:
                self._detect_cache[path] = {}
        self.app.call_from_thread(self._render_info, m, False)

    def _render_info(self, m: dict, detecting: bool) -> None:
        if self.selected is not m:
            return
        info = self._detect_cache.get(self._detect_path(m), {})
        a, d = "#007878", "#6a5a48"

        def row(k, v):
            return f"[{d}]{k:<12}[/][{a}]{v}[/]"

        rows = [
            row("Model", rescape(m["key"])),
            row("File", rescape(Path(m["path"]).name)),
            row("Format", m["format"].upper()),
            f"[{d}]{'Quant':<12}[/][{ACC_ORANGE}]{m['quant']}[/]",
            f"[{d}]{'Params':<12}[/][{ACC_BLUE}]{m['params']}[/]",
            row("Backend", m["backend"]),
            row("Size", f"{m['size_gb']:.2f} GB"),
        ]
        if detecting and not info:
            rows.append(f"[{d}]… detecting metadata[/]")
        if info:
            if info.get("max_ctx"):
                rows.append(row("Max ctx", f"{info['max_ctx']:,}"))
            ggufs = info.get("gguf_files", [])
            if ggufs:
                rows.append(row("GGUFs", str(len(ggufs))))
            mm = info.get("mmproj_files", []) or ([m["cfg"]["mmproj"]] if m["cfg"].get("mmproj") else [])
            if mm:
                rows.append(row("mmproj", rescape(Path(str(mm[0])).name)))
            caps = (["vision"] if (info.get("has_vision") or m["vision"]) else []) \
                 + (["moe"] if m["moe"] else []) \
                 + (["mtp"] if m["mtp"] else []) + (["tool use"])
            rows.append(row("Capabilities", " · ".join(caps)))
        rows.append(f"[{d}]Path[/] {rescape(m['path'])}")
        if not m["configured"]:
            rows.append(f"\n[{ACC_YELLOW}]+ no config — adjust LOAD and save[/]")
        elif m["key"] in self.loaded_status:
            s = self.loaded_status[m["key"]]
            mark = " (one-time config)" if s.get("custom_load") else ""
            mark_c = ACC_ORANGE if s.get("custom_load") else ACC_GREEN
            rows.append(f"\n[{ACC_GREEN}]● running :{s.get('port')} GPU {s.get('gpu')}[/]"
                        f"[{mark_c}]{mark}[/]")
        self.query_one("#info-text", Static).update("\n".join(rows))

    # ── LOAD form ─────────────────────────────────────────────────────────
    def _slider_specs(self, m: dict) -> list:
        specs = LLAMA_SLIDERS if m["backend"] == "llama.cpp" else VLLM_SLIDERS
        info = self._detect_cache.get(self._detect_path(m), {})
        max_ctx = info.get("max_ctx") or 262144
        out = []
        for key, label, vmin, vmax, step, is_int, default in specs:
            if key in ("n_ctx", "max_model_len"):
                vmax = max_ctx
            if key == "tensor_parallel":
                vmax = max(1, len(self.gpus))
            # never clamp below what the config already uses
            try:
                vmax = max(vmax, int(float(m["cfg"].get(key, vmin))))
            except (TypeError, ValueError):
                pass
            out.append((key, label, vmin, vmax, step, is_int, default))
        return out

    async def _build_load_form(self, m: dict) -> None:
        form = self.query_one("#load-form", ScrollableContainer)
        await form.remove_children()
        cfg = m["cfg"]
        # parse extra args up-front: field aliases (e.g. --ubatch-size) are
        # absorbed into the sliders instead of sitting in "Other (advanced)"
        catalog = LLAMA_EXTRA_CATALOG if m["backend"] == "llama.cpp" else VLLM_EXTRA_CATALOG
        enabled, leftover = parse_extra_args(cfg.get("extra_args", []), catalog)
        # Some boolean flags are "promoted" by the parser into top-level cfg
        # fields (e.g. `enforce-eager` → cfg["enforce_eager"]=True) instead of
        # living in extra_args. They'd otherwise be invisible in the flag
        # checklist — surface them as enabled so the toggle reflects reality.
        if str(cfg.get("enforce_eager", "")).lower() in ("true", "1", "yes"):
            enabled.setdefault("--enforce-eager", [])
        absorbed: dict[str, str] = {}
        if m["backend"] == "llama.cpp":
            absorbed, leftover = absorb_field_aliases(leftover, LLAMA_FIELD_ALIASES)
        widgets = [Static(f"[{ACC_BLUE}]▮[/] Base config" + ("" if m["configured"] else " — new"),
                          classes="section-hdr"),
                   Static("", id="vram-line", classes="vram-line")]
        COUNTER_KEYS = {"tensor_parallel", "max_num_seqs"}
        for key, label, vmin, vmax, step, is_int, default in self._slider_specs(m):
            cls = ParamCounter if key in COUNTER_KEYS else ParamSlider
            value = cfg.get(key, absorbed.get(key, default))
            widgets.append(cls(key, label, vmin, vmax, step, is_int,
                               value, id=f"ld-{key}"))
        # GPU pin — radio: ● auto  ○ GPU 0  ○ GPU 1
        gpu_opts = [("auto", "")] + [(f"GPU {g['index']}", str(g["index"])) for g in self.gpus]
        widgets.append(Static(f"[{ACC_GREEN}]▮[/] GPU Pin", classes="section-hdr"))
        widgets.append(RadioRow(gpu_opts, value=str(cfg.get("gpu_id", "")), id="ld-gpu_id"))
        # mmproj (llama.cpp)
        if m["backend"] == "llama.cpp":
            widgets.append(Static(f"[{ACC_RED}]▮[/] mmproj (vision)", classes="section-hdr"))
            widgets.append(Horizontal(
                Input(value=str(cfg.get("mmproj", "")), id="ld-mmproj"),
                Button("📂", id="btn-browse-mmproj", classes="mini"),
                classes="mmproj-row",
            ))
        # extra args — checklist with editable values and per-flag (i) help
        widgets.append(Static(f"[{ACC_ORANGE}]▮[/] Features & tuning", classes="section-hdr"))
        for flag, defaults, desc in catalog:
            widgets.append(FlagRow(flag, defaults, desc,
                                   on=flag in enabled, vals=enabled.get(flag)))
        widgets.append(Collapsible(
            Static("Flags outside the list, original spelling (--flag value). "
                   "Enter moves known flags to the list above.",
                   classes="field-hint"),
            Input(value=" ".join(leftover), id="ld-x-custom"),
            title="Other (advanced)", collapsed=not leftover,
        ))
        # profile to load + actions
        profs = [p["name"] for p in self.profiles if p["base"] == m["key"]]
        if profs:
            widgets.append(Static("Load via profile", classes="field-hint"))
            widgets.append(Select([(p, p) for p in profs], value=profs[0],
                                  allow_blank=False, id="ld-profile"))
        btns = []
        if m["configured"] and profs:
            btns.append(Button("▶ Load once", id="btn-load-once"))
        if m["key"] in self.loaded_status:
            btns.append(Button("⏏ Unload", id="btn-unload", classes="warn"))
        btns.append(Button("💾 Save base", id="btn-save-base"))
        widgets.append(Horizontal(*btns, classes="btn-row"))
        if m["configured"] and not profs:
            widgets.append(Static("[#7a1818]no profiles — create one in the PROFILES tab[/]",
                                  classes="field-hint"))
        await form.mount_all(widgets)
        if m["backend"] == "vllm":
            # KV budget for the len ↔ seqs coupling (from the current config)
            len0 = float(cfg.get("max_model_len", 131072) or 131072)
            seqs0 = float(cfg.get("max_num_seqs", 8) or 8)
            self._kv_budget = max(1024.0, len0) * max(1.0, seqs0)
        self._update_vram_line(m)

    # — VRAM estimate + couplings —
    def _ps_value(self, wid: str, default=None):
        """Value of a ParamSlider OR ParamCounter by id."""
        try:
            return self.query_one(f"#{wid}").value
        except Exception:
            return default

    def _breakdown_cfg(self, m: dict) -> dict:
        """Assemble a cfg dict from disk config + live form state, in the shape
        core.gpu.get_vram_breakdown expects."""
        cfg = dict(m.get("cfg") or {})
        cfg.update(self._load_form_values(m))
        cfg["backend"] = m["backend"]
        if m["backend"] == "llama.cpp":
            cfg.setdefault("model", m["path"])
        else:
            cfg.setdefault("path", m["path"])
        return cfg

    def _vram_breakdown(self, m: dict) -> dict:
        from core.gpu import get_vram_breakdown
        try:
            return get_vram_breakdown(self._breakdown_cfg(m), m.get("key", ""))
        except Exception:
            return {"weights_gb": m["size_gb"] * 1.15 if m["size_gb"] else 4.0,
                    "kv_cache_gb": 0.5, "mmproj_gb": 0.0, "vision_gb": 0.0,
                    "cudagraph_gb": 0.0, "mtp_gb": 0.0, "overhead_gb": 0.5,
                    "total_gb": (m["size_gb"] * 1.15 if m["size_gb"] else 4.0) + 1.0}

    def _target_gpus(self, m: dict) -> list[dict]:
        """GPUs the current form targets: the pinned one, or the top-TP by free."""
        try:
            pin = int(self.query_one("#ld-gpu_id", RadioRow).value)
        except (Exception, ValueError):
            pin = None
        if pin is not None:
            return [g for g in self.gpus if g["index"] == pin]
        tp = self._ps_value("ld-tensor_parallel", 1) or 1
        ranked = sorted(self.gpus, key=lambda g: g["free_gb"], reverse=True)
        return ranked[:max(1, int(tp))] or self.gpus[:1]

    def _vram_estimate(self, m: dict) -> tuple[float, float]:
        """(needed_gb, available_gb) for the current form state."""
        need = self._vram_breakdown(m)["total_gb"]
        avail = sum(g["free_gb"] for g in self._target_gpus(m))
        return need, avail

    def _update_vram_line(self, m: dict) -> None:
        try:
            line = self.query_one("#vram-line", Static)
        except Exception:
            return
        bd = self._vram_breakdown(m)
        targets = self._target_gpus(m)
        total_gb = sum(g["total_gb"] for g in targets) or 24.0
        free_gb = sum(g["free_gb"] for g in targets)
        sys_gb = max(0.0, total_gb - free_gb)

        extras_gb = (bd["mmproj_gb"] + bd["vision_gb"] + bd["cudagraph_gb"]
                     + bd["mtp_gb"] + bd["overhead_gb"])
        need = bd["total_gb"]
        fits = need <= free_gb

        # Segmented bar over the target GPUs' TOTAL capacity:
        #   system-in-use │ weights │ kv cache │ extras │ free
        width = 30
        segs = [
            (sys_gb,           "#a89878"),   # already used by system/other models
            (bd["weights_gb"], ACC_BLUE),    # model weights
            (bd["kv_cache_gb"], ACC_ORANGE), # kv cache
            (extras_gb,        ACC_YELLOW),  # mtp / cudagraph / vision / overhead
        ]
        cells = []
        used_cells = 0
        for gb, color in segs:
            n = round(gb / total_gb * width)
            n = min(n, width - used_cells)
            if gb > 0 and n == 0 and used_cells < width:
                n = 1  # never hide a non-zero component entirely
            used_cells += n
            cells.append((n, color))
        over = used_cells > width or (sys_gb + need) > total_gb
        bar = "".join(f"[{c}]{'▰' * n}[/]" for n, c in cells if n)
        bar += f"[#d8cfae]{'▱' * max(0, width - used_cells)}[/]"

        status_color = ACC_GREEN if fits else ACC_RED
        verdict = "fits" if fits else "DOESN'T FIT"

        # Legend: colored swatch + label + GB, laid out in two aligned columns.
        items = [("#a89878", "sys", sys_gb),
                 (ACC_BLUE, "weights", bd["weights_gb"]),
                 (ACC_ORANGE, "kv", bd["kv_cache_gb"])]
        if bd["mtp_gb"]:
            items.append((ACC_YELLOW, "mtp", bd["mtp_gb"]))
        if bd["cudagraph_gb"]:
            items.append((ACC_YELLOW, "cudagraph", bd["cudagraph_gb"]))
        if bd["vision_gb"] or bd["mmproj_gb"]:
            items.append((ACC_YELLOW, "vision", bd["vision_gb"] + bd["mmproj_gb"]))
        items.append((ACC_YELLOW, "ovh", bd["overhead_gb"]))

        col_w = 18  # visible chars per column cell
        cells = []
        for color, label, gb in items:
            plain = f"{label} {gb:.1f}"
            pad = " " * max(1, col_w - 2 - len(plain))
            cells.append(f"[{color}]▰[/] {plain}{pad}")
        legend_rows = ["".join(cells[i:i + 2]).rstrip() for i in range(0, len(cells), 2)]
        legend = "\n".join(legend_rows)

        line.update(
            f"{bar} [bold {status_color}]{verdict}[/] "
            f"[bold]{need:.1f}[/]/{free_gb:.1f} GB free of {total_gb:.0f}\n{legend}"
            + (f"\n[{ACC_RED}]⚠ over capacity — reduce context or offload[/]" if over and not fits else "")
        )

    def on_param_slider_changed(self, event: ParamSlider.Changed) -> None:
        m = self.selected
        if m is None:
            return
        pid = event.param.id or ""
        key = event.key
        if pid.startswith("ld-"):
            # the user touched the coupling target manually → respect it
            if key == "n_gpu_layers":
                self._auto_layers = False
            if key == "max_num_seqs":
                self._auto_seqs = False
            # coupling: context ↑ → shrink whatever competes for the same VRAM
            if key == "n_ctx" and self._auto_layers and m["size_gb"]:
                weights = m["size_gb"] * 1.15
                kv = event.value * m["size_gb"] / 500_000
                _, avail = self._vram_estimate(m)
                fit = (avail - kv) / weights if weights else 1.0
                new_layers = -1 if fit >= 1.0 else max(0, int(99 * max(0.0, fit)))
                ps = self._ps_value_widget("ld-n_gpu_layers")
                if ps and ps.value != new_layers:
                    ps.set_value(new_layers)
                    self._status(f"auto: GPU Offload → {ps._display()}")
            if key == "max_model_len" and self._auto_seqs and self._kv_budget:
                new_seqs = max(1, min(64, int(self._kv_budget / max(1024, event.value))))
                ps = self._ps_value_widget("ld-max_num_seqs")
                if ps and ps.value != new_seqs:
                    ps.set_value(new_seqs)
                    self._status(f"auto: Max Sequences → {new_seqs}")
            self._update_vram_line(m)

    def _ps_value_widget(self, wid: str):
        """ParamSlider or ParamCounter by id (shared interface: value/value_str/set_value)."""
        try:
            w = self.query_one(f"#{wid}")
            return w if isinstance(w, (ParamSlider, ParamCounter)) else None
        except Exception:
            return None

    def on_radio_row_changed(self, event: RadioRow.Changed) -> None:
        if (event.radio.id or "") == "ld-gpu_id" and self.selected:
            self._update_vram_line(self.selected)

    def on_param_counter_changed(self, event: ParamCounter.Changed) -> None:
        if not self.selected or not (event.param.id or "").startswith("ld-"):
            return
        if event.key == "max_num_seqs":
            self._auto_seqs = False   # the user took over
        self._update_vram_line(self.selected)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Enter in "Other (advanced)": known flags move up to the list
        if (event.input.id or "") != "ld-x-custom" or not self.selected:
            return
        catalog = (LLAMA_EXTRA_CATALOG if self.selected["backend"] == "llama.cpp"
                   else VLLM_EXTRA_CATALOG)
        try:
            tokens = shlex.split(event.value)
        except ValueError:
            return
        enabled, leftover = parse_extra_args(tokens, catalog)
        if not enabled:
            return
        for row in self.query_one("#load-form").query(FlagRow):
            if row.flag in enabled:
                row.set_state(True, enabled[row.flag] or row.defaults)
        event.input.value = " ".join(leftover)
        self._status(f"→ moved to the list: {', '.join(enabled)}")

    # ── form values ───────────────────────────────────────────────────────
    def _load_form_values(self, m: dict) -> dict:
        out = {}
        for key, *_ in self._slider_specs(m):
            ps = self._ps_value_widget(f"ld-{key}")
            if ps is not None:
                out[key] = ps.value_str
        try:
            pin = self.query_one("#ld-gpu_id", RadioRow).value
            if pin:
                out["gpu_id"] = str(pin)
        except Exception:
            pass
        if m["backend"] == "llama.cpp":
            try:
                mm = self.query_one("#ld-mmproj", Input).value.strip()
                if mm:
                    out["mmproj"] = mm
            except Exception:
                pass
        # extra args — read each FlagRow's state/value + advanced leftovers
        enabled = {}
        try:
            for row in self.query_one("#load-form").query(FlagRow):
                if row.enabled:
                    enabled[row.flag] = row.args()
        except Exception:
            pass
        try:
            custom = self.query_one("#ld-x-custom", Input).value
        except Exception:
            custom = ""
        out["extra_args"] = serialize_extra_args(enabled, custom)
        return out

    @staticmethod
    def _values_equal(a, b) -> bool:
        if isinstance(a, list) or isinstance(b, list):
            return [str(x) for x in (a or [])] == [str(x) for x in (b or [])]
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return str(a) == str(b)

    def _load_overrides(self, m: dict) -> dict:
        return {k: v for k, v in self._load_form_values(m).items()
                if not self._values_equal(m["cfg"].get(k, ""), v)}

    # ── PROFILES form ─────────────────────────────────────────────────────
    async def _build_profiles_form(self, m: dict) -> None:
        form = self.query_one("#profiles-form", ScrollableContainer)
        await form.remove_children()
        widgets = []
        for i, p in enumerate([p for p in self.profiles if p["base"] == m["key"]]):
            widgets.append(self._profile_collapsible(p, i, collapsed=i > 0))
        widgets.append(Horizontal(Button("＋ New profile", id="btn-new-profile"),
                                  classes="btn-row", id="profiles-actions"))
        await form.mount_all(widgets)

    def _profile_collapsible(self, p: dict, idx: int, collapsed: bool = True,
                             new: bool = False) -> Collapsible:
        sampling = p["cfg"].get("sampling", {})
        kids = [Static("Profile name", classes="field-hint"),
                Input(value=p["name"], id=f"pf{idx}-name"),
                Toggle("enable_thinking",
                       value=_as_bool(p["cfg"].get("enable_thinking", False)),
                       id=f"pf{idx}-thinking")]
        for key, label, vmin, vmax, step, is_int, default in SAMPLING_SLIDERS:
            kids.append(ParamSlider(key, label, vmin, vmax, step, is_int,
                                    sampling.get(key, default), id=f"pf{idx}-{key}"))
        kids.append(Horizontal(
            Button("▶ Load once", id=f"pf{idx}-load"),
            Button("💾 Save", id=f"pf{idx}-save"),
            classes="btn-row",
        ))
        c = Collapsible(*kids, title=p["name"], collapsed=collapsed, id=f"pf{idx}")
        c._allma_profile = p
        c._allma_new = new
        return c

    def _profile_form_values(self, idx: int) -> dict:
        out = {}
        for key, *_ in SAMPLING_SLIDERS:
            ps = self._ps_value_widget(f"pf{idx}-{key}")
            if ps is not None:
                out[key] = int(ps.value) if ps.is_int else round(float(ps.value), 4)
        return out

    # ── actions ───────────────────────────────────────────────────────────
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        m = self.selected
        if bid == "btn-new-profile":
            self._new_profile()
            return
        if bid == "btn-dl":
            self._start_download()
            return
        if bid == "btn-top-reload":
            self._status("⟳ restarting the server…")
            self._reload_server_async()
            return
        if bid == "btn-top-load":
            if m is None:
                self.notify("Select a model in the list", severity="warning")
            elif not self.server_online:
                self.notify("Server offline — use the ⟳ Server button", severity="error")
            else:
                profile = self._selected_profile_name(m)
                if not profile:
                    self.notify("Create a profile first (PROFILES tab)", severity="warning")
                else:
                    # play = "official" load: clears ephemeral overrides, uses the .allm
                    self._status(f"⏳ loading {profile} (disk config)…")
                    self._load_async(profile, {}, {})
            return
        if m is None:
            return
        if bid == "btn-save-base":
            await self._save_base(m)
        elif bid == "btn-load-once":
            self._do_load_once(m, profile=self._selected_profile_name(m))
        elif bid == "btn-unload":
            self._do_unload(m)
        elif bid == "btn-browse-mmproj":
            self._pick_mmproj(m)
        elif bid.startswith("pf") and bid.endswith("-load"):
            idx = int(bid[2:].split("-")[0])
            c = self.query_one(f"#pf{idx}", Collapsible)
            self._do_load_once(m, profile=c._allma_profile["name"],
                               sampling=self._profile_form_values(idx))
        elif bid.startswith("pf") and bid.endswith("-save"):
            await self._save_profile(m, int(bid[2:].split("-")[0]))

    def _pick_mmproj(self, m: dict) -> None:
        root = Path(self._detect_path(m))

        def _done(path: str | None) -> None:
            if path:
                try:
                    self.query_one("#ld-mmproj", Input).value = path
                except Exception:
                    pass
        self.push_screen(PickerScreen(root, "gguf", "Select mmproj (.gguf)"), _done)

    def _selected_profile_name(self, m: dict) -> str | None:
        try:
            sel = self.query_one("#ld-profile", Select)
            if sel.value not in ("", Select.BLANK, None):
                return str(sel.value)
        except Exception:
            pass
        profs = [p["name"] for p in self.profiles if p["base"] == m["key"]]
        return profs[0] if profs else None

    # — save base —
    # Promoted-field → v2 flag-name map (mirror of the parser's promotions).
    _V2_VLLM = {
        "tensor_parallel": "tensor-parallel-size", "max_model_len": "max-model-len",
        "max_num_seqs": "max-num-seqs", "max_num_batched_tokens": "max-num-batched-tokens",
        "gpu_memory_utilization": "gpu-memory-utilization",
    }
    _V2_LLAMA = {
        "n_ctx": "-c", "n_batch": "-b", "n_gpu_layers": "-ngl",
        "n_threads": "-t", "n_ubatch": "-ub",
    }

    def _render_base_allm(self, m: dict, values: dict, extra: list) -> str:
        """Build a complete v2 .allm from the form state (always full-rewrite —
        the old field-patcher emitted TOML and corrupted v2 files)."""
        is_llama = m["backend"] == "llama.cpp"
        promo = self._V2_LLAMA if is_llama else self._V2_VLLM
        lines = [f"# Base model: {m['key']} ({m['backend']} backend)"]
        lines.append("@llamacpp" if is_llama else "@vllm")
        # model path: for llama the gguf; for vllm the model dir
        if is_llama:
            model_path = m["cfg"].get("model") or m["path"]
            lines.append(f"@path {model_path}")
        else:
            lines.append(f"@path {m['path']}")
        gid = values.get("gpu_id", m["cfg"].get("gpu_id"))
        if gid not in (None, "", "-1"):
            lines.append(f"@gpu {gid}")
        if m["cfg"].get("pin_loaded"):
            lines.append("@pin")
        if is_llama:
            mm = values.get("mmproj") or m["cfg"].get("mmproj")
            lines.append("")
            for key, flag in promo.items():
                if key in values:
                    lines.append(f"{flag} {values[key]}")
            if mm:
                lines.append(f"--mmproj {mm}")
        else:
            lines.append("")
            for key, flag in promo.items():
                if key in values:
                    lines.append(f"{flag} {values[key]}")
        # extra_args: emit as bare flag lines (dashes preserved, JSON verbatim)
        i = 0
        extra = [str(a) for a in extra]
        while i < len(extra):
            tok = extra[i]
            if tok.startswith("-"):
                name = tok.lstrip("-")
                if i + 1 < len(extra) and not extra[i + 1].startswith("--"):
                    lines.append(f"{name} {extra[i + 1]}")
                    i += 2
                else:
                    lines.append(name)
                    i += 1
            else:
                lines.append(tok)
                i += 1
        return "\n".join(lines) + "\n"

    async def _save_base(self, m: dict) -> None:
        values = self._load_form_values(m)
        extra = values.pop("extra_args", [])
        if not m["configured"]:
            path = Path(m["path"])
            info = self._detect_cache.get(str(path)) or detect_model(path)
            if m["backend"] == "llama.cpp":
                ggufs = [g for g in info.get("gguf_files", []) if "mmproj" not in g.lower()]
                m["cfg"].setdefault("model", ggufs[0] if ggufs else m["path"])
                mmprojs = info.get("mmproj_files", [])
                if mmprojs and not values.get("mmproj"):
                    values["mmproj"] = mmprojs[0]
        f = m["file"] if m["configured"] else CONFIG_DIR / "base" / f"{m['key']}.allm"
        f.write_text(self._render_base_allm(m, values, extra), encoding="utf-8")
        self._status(f"💾 {f.name} {'saved' if m['configured'] else 'created'}")
        await self._after_save()

    # — save profile (with rename) —
    async def _save_profile(self, m: dict, idx: int) -> None:
        c = self.query_one(f"#pf{idx}", Collapsible)
        p = c._allma_profile
        sampling = self._profile_form_values(idx)
        thinking = self.query_one(f"#pf{idx}-thinking", Toggle).value
        new_name = self.query_one(f"#pf{idx}-name", Input).value.strip() or p["name"]
        base = p.get("base") or m["key"]
        content = self._render_profile_allm(new_name, base, thinking, sampling)
        if getattr(c, "_allma_new", False):
            fname = new_name.replace(":", "-").replace("/", "-")
            f = CONFIG_DIR / "profile" / f"{fname}.allm"
            f.write_text(content, encoding="utf-8")
            self._status(f"💾 {f.name} created")
        else:
            f = p["file"]
            f.write_text(content, encoding="utf-8")
            if new_name != p["name"]:
                new_file = f.with_name(new_name.replace(":", "-").replace("/", "-") + ".allm")
                if not new_file.exists():
                    f.rename(new_file)
                    f = new_file
                self._status(f"💾 renamed: {p['name']} → {new_name}")
            else:
                self._status(f"💾 {f.name} saved")
        await self._after_save()

    @staticmethod
    def _render_profile_allm(name: str, base: str, thinking: bool, sampling: dict) -> str:
        lines = [f"# Profile: {name}", f"@name {name}", f"@base {base}"]
        if not thinking:
            lines.append("@thinking-off")
        lines.append("")
        for k, v in sampling.items():
            lines.append(f"{k.replace('_', '-')} {v}")
        return "\n".join(lines) + "\n"

    async def _after_save(self) -> None:
        self.profiles = scan_profiles()
        self.models = scan_models(self.models_dir)
        self._refresh_table()
        if self.selected:
            key = self.selected["key"]
            self.selected = next((x for x in self.models if x["key"] == key), None)
            if self.selected:
                await self._show_model(self.selected)
        if self.server_online:
            self._reload_configs_async()

    @work(thread=True, group="http")
    def _reload_configs_async(self) -> None:
        ok, resp = _http("POST", "/v1/reload-configs")
        msg = ("⟳ configs reloaded on the server" if ok
               else f"⚠ reload failed: {resp.get('error', '?')}")
        self.app.call_from_thread(self._status, msg)

    # — load once / unload —
    def _do_load_once(self, m: dict, profile: str | None, sampling: dict | None = None) -> None:
        if not self.server_online:
            self.notify("Server offline — run `allma serve`", severity="error")
            return
        if not profile:
            self.notify("Create a profile first (PROFILES tab)", severity="warning")
            return
        overrides = self._load_overrides(m)
        sampling_diff = {}
        if sampling is not None:
            prof = next((p for p in self.profiles if p["name"] == profile), None)
            disk = (prof or {}).get("cfg", {}).get("sampling", {})
            sampling_diff = {k: v for k, v in sampling.items()
                             if not self._values_equal(disk.get(k, ""), v)}
        self._status(f"⏳ loading {profile}…")
        self._load_async(profile, overrides, sampling_diff)

    @work(thread=True, group="http")
    def _load_async(self, profile: str, overrides: dict, sampling: dict) -> None:
        body = {"model": profile, "load_overrides": overrides, "sampling": sampling}
        ok, resp = _http("POST", "/v1/load", body, timeout=600)
        if ok:
            mark = " (one-time config)" if resp.get("custom_load") else ""
            msg = f"● {profile} loaded{mark}"
        else:
            msg = f"✗ failed: {resp.get('error') or resp}"
        self.app.call_from_thread(self._status, msg)
        self.app.call_from_thread(self._poll_server)

    def _do_unload(self, m: dict) -> None:
        self._status(f"⏏ unloading {m['key']}…")
        self._unload_async(m["key"])

    @work(thread=True, exclusive=True, group="server")
    def _reload_server_async(self) -> None:
        cli = str(BASE_DIR / "allma_cli.py")
        try:
            subprocess.run([sys.executable, cli, "stop"], capture_output=True, timeout=90)
            self.app.call_from_thread(self._status, "⟳ server stopped — bringing it back up…")
            subprocess.run([sys.executable, cli, "serve"], capture_output=True, timeout=90)
            ok = False
            for _ in range(20):
                ok, _h = _http("GET", "/health", timeout=2)
                if ok:
                    break
                import time
                time.sleep(0.5)
            msg = "⟳ server restarted" if ok else "✗ server didn't come back — check `allma logs`"
        except Exception as e:
            msg = f"✗ restart failed: {e}"
        self.app.call_from_thread(self._status, msg)
        self.app.call_from_thread(self._poll_server)

    @work(thread=True, group="http")
    def _unload_async(self, base: str) -> None:
        ok, resp = _http("POST", "/v1/unload", {"model": base}, timeout=60)
        msg = f"⏏ {base} unloaded" if ok else f"✗ {resp.get('error') or resp}"
        self.app.call_from_thread(self._status, msg)
        self.app.call_from_thread(self._poll_server)

    # — new profile —
    def _new_profile(self) -> None:
        m = self.selected
        if m is None:
            return
        form = self.query_one("#profiles-form", ScrollableContainer)
        existing = [p for p in self.profiles if p["base"] == m["key"]]
        idx = 1000 + len(form.query(Collapsible))
        info = self._detect_cache.get(self._detect_path(m), {})
        preset = FAMILY_PRESETS.get(info.get("family", ""), {})
        sampling = dict(preset.get("sampling", {}))
        p = {"name": f"{m['key']}-Custom{len(existing) + 1}",
             "base": m["key"], "cfg": {"sampling": sampling}, "file": None}
        c = self._profile_collapsible(p, idx, collapsed=False, new=True)
        form.mount(c, before=self.query_one("#profiles-actions"))

    # ── DOWNLOAD (HuggingFace repo / direct .gguf URL e.g. GitHub releases) ──
    def _start_download(self) -> None:
        if self._dl_busy:
            self.notify("A download is already in progress", severity="warning")
            return
        url = self.query_one("#dl-url", Input).value.strip()
        if not url:
            self.notify("Enter a HuggingFace repo-id or a .gguf URL", severity="warning")
            return
        # direct file URL (GitHub releases, any host)
        if url.startswith("http") and url.split("?")[0].endswith(".gguf"):
            self._dl_direct_async(url)
            return
        # HF repo: first click lists, second click downloads the selection
        if self._dl_files is not None and self._dl_repo:
            try:
                choice = self.query_one("#dl-pick", Select).value
            except Exception:
                choice = None
            if choice in (None, Select.BLANK):
                self.notify("Pick a file to download", severity="warning")
                return
            self._dl_download_async(self._dl_repo, self._dl_files, str(choice))
        else:
            self._status(f"⇣ listing {url}…")
            self._dl_list_async(url)

    def on_input_changed(self, event: Input.Changed) -> None:
        # mmproj path affects the VRAM estimate (vision projector weight)
        if (event.input.id or "") == "ld-mmproj" and self.selected:
            self._update_vram_line(self.selected)
            return
        # typing a new URL invalidates the previous listing
        if (event.input.id or "") == "dl-url" and self._dl_files is not None:
            self._dl_files = None
            self._dl_repo = None
            try:
                self.query_one("#dl-pick").remove()
            except Exception:
                pass
            self.query_one("#btn-dl", Button).label = "⇣ Fetch"

    @work(thread=True, exclusive=True, group="dl")
    def _dl_list_async(self, url: str) -> None:
        try:
            from core.downloader import list_repo_files, parse_hf_url
            repo_id = parse_hf_url(url)
            files = list_repo_files(repo_id)
        except Exception as e:
            self.app.call_from_thread(self._status, f"✗ listing failed: {e}")
            return
        self.app.call_from_thread(self._dl_show_options, repo_id, files)

    def _dl_show_options(self, repo_id: str, files: dict) -> None:
        self._dl_repo = repo_id
        self._dl_files = files
        options: list[tuple[str, str]] = []
        for g in files.get("gguf", []):
            options.append((f"{g['name']} ({_fmt_size(g['size'])})", g["name"]))
        if files.get("safetensors"):
            total = sum(s.get("size") or 0 for s in files["safetensors"])
            options.append((f"full repo · safetensors ({_fmt_size(total)})", "__safetensors__"))
        if not options:
            self._status(f"✗ no model files found in {repo_id}")
            self._dl_files = None
            return
        try:
            self.query_one("#dl-pick").remove()
        except Exception:
            pass
        panel = self.query_one("#download", Vertical)
        sel = Select(options, value=options[0][1], allow_blank=False, id="dl-pick")
        panel.mount(sel, before=panel.query_one(".dl-row"))
        self.query_one("#btn-dl", Button).label = "⇣ Download"
        n = len(files.get("gguf", []))
        self._status(f"⇣ {repo_id}: {n} GGUF file(s) — pick one and hit Download")

    # — progress (polls destination dir size while the worker downloads) —
    def _dl_progress_start(self, dest: Path, expected: int, label: str) -> None:
        self._dl_dest = dest
        self._dl_expected = expected
        self._dl_label = label
        self._dl_timer = self.set_interval(1.0, self._dl_tick)

    def _dl_tick(self) -> None:
        if not self._dl_dest:
            return
        done = _dir_size(self._dl_dest)
        if self._dl_expected:
            pct = min(100, int(done * 100 / self._dl_expected))
            self._status(f"⇣ {self._dl_label}: {_fmt_size(done)} / "
                         f"{_fmt_size(self._dl_expected)} ({pct}%)")
        else:
            self._status(f"⇣ {self._dl_label}: {_fmt_size(done)}")

    def _dl_progress_stop(self) -> None:
        if self._dl_timer is not None:
            self._dl_timer.stop()
            self._dl_timer = None
        self._dl_dest = None

    @work(thread=True, exclusive=True, group="dl")
    def _dl_download_async(self, repo_id: str, files: dict, choice: str) -> None:
        self._dl_busy = True
        model_name = repo_id.split("/")[-1]
        dest = self.models_dir / model_name
        try:
            from huggingface_hub import hf_hub_download
            if choice == "__safetensors__":
                expected = sum(s.get("size") or 0 for s in files["safetensors"])
                self.app.call_from_thread(self._dl_progress_start, dest, expected, model_name)
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=repo_id, local_dir=str(dest))
            else:
                chosen = next(g for g in files["gguf"] if g["name"] == choice)
                expected = (chosen.get("size") or 0) \
                    + sum(m.get("size") or 0 for m in files.get("mmproj", []))
                self.app.call_from_thread(self._dl_progress_start, dest, expected, model_name)
                to_get = [choice] + [m["name"] for m in files.get("mmproj", [])] \
                    + list(files.get("config", []))
                for fname in to_get:
                    hf_hub_download(repo_id=repo_id, filename=fname, local_dir=str(dest))
            # auto-create configs like `allma download` does
            try:
                from core.downloader import create_configs
                create_configs(dest, model_name)
            except Exception:
                pass
            _http("POST", "/v1/reload-configs")
            msg = f"✓ {model_name} downloaded"
        except Exception as e:
            msg = f"✗ download failed: {e}"
        finally:
            self._dl_busy = False
        self.app.call_from_thread(self._dl_progress_stop)
        self.app.call_from_thread(self._dl_reset_ui)
        self.app.call_from_thread(self.action_rescan)
        self.app.call_from_thread(self._status, msg)   # last, so ✓/✗ stays visible

    @work(thread=True, exclusive=True, group="dl")
    def _dl_direct_async(self, url: str) -> None:
        self._dl_busy = True
        fname = Path(url.split("?")[0]).name
        dest = self.models_dir / Path(fname).stem
        try:
            dest.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(url, headers={"User-Agent": "AllmaTUI/2.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                expected = int(resp.headers.get("Content-Length") or 0)
                self.app.call_from_thread(self._dl_progress_start, dest, expected, fname)
                with open(dest / fname, "wb") as out:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
            try:
                from core.downloader import create_configs
                create_configs(dest, dest.name)
            except Exception:
                pass
            _http("POST", "/v1/reload-configs")
            msg = f"✓ {fname} downloaded"
        except Exception as e:
            msg = f"✗ download failed: {e}"
        finally:
            self._dl_busy = False
        self.app.call_from_thread(self._dl_progress_stop)
        self.app.call_from_thread(self.action_rescan)
        self.app.call_from_thread(self._status, msg)   # last, so ✓/✗ stays visible

    def _dl_reset_ui(self) -> None:
        self._dl_files = None
        self._dl_repo = None
        try:
            self.query_one("#dl-pick").remove()
        except Exception:
            pass
        try:
            self.query_one("#btn-dl", Button).label = "⇣ Fetch"
            self.query_one("#dl-url", Input).value = ""
        except Exception:
            pass

    # — shortcuts —
    def action_load_one_time(self) -> None:
        if self.selected:
            self._do_load_once(self.selected, self._selected_profile_name(self.selected))

    async def action_save_base(self) -> None:
        if self.selected:
            await self._save_base(self.selected)

    def _status(self, msg: str) -> None:
        d = "#6a5a48"
        keys = (f"[{ACC_ORANGE}]F5[/][{d}] rescan  ·[/] [{ACC_ORANGE}]^L[/][{d}] load once  ·[/] "
                f"[{ACC_ORANGE}]^S[/][{d}] save base  ·[/] [{ACC_ORANGE}]Q[/][{d}] quit[/]")
        self.query_one("#statusline", Static).update(f"{msg}    [{d}]·[/]  {keys}")


def main():
    AllmaTUI().run()


if __name__ == "__main__":
    main()
