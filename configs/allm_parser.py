"""New .allm parser (v2 format).

The v1 parser (TOML-like) is being replaced with a minimal, shell-command-flavoured
syntax. See README for the full spec. Summary:

- Lines starting with `@` are Allma meta-directives.
- Lines without `@` are backend flags (in a base) or sampling params (in a profile).
- No `=`, no quotes, no brackets. Hyphens on flag names are optional (the parser
  adds `--` when missing).
- Boolean flags: presence = enabled; absence = default. There is no `false` form —
  to disable, delete the line.
- `#` starts a comment. Blank lines are decorative.

Backend detection:
    @vllm         → backend = "vllm"
    @llamacpp     → backend = "llama.cpp"

Meta directives (bases):
    @path <abs-path>          # required
    @tokenizer <abs-path>     # optional; defaults to @path
    @gpu N                    # single GPU id, sets gpu_id
    @gpus N,M                 # multi-GPU list (future — currently sets gpu_id to first)
    @pin                      # pin_loaded = True
    @keep-alive N             # per-model idle timeout override (seconds)

Meta directives (profiles):
    @name <shown-name>        # required (also inferred from filename)
    @base <base-config-name>  # required
    @thinking-off             # enable_thinking = False (default is True)

Flag promotion — some backend flags become top-level cfg fields because
core/process.py wires them directly into the command. The parser promotes
these; anything else lands in cfg["extra_args"]:

    vLLM promoted:
        tensor-parallel-size    → tensor_parallel
        max-model-len           → max_model_len
        max-num-seqs            → max_num_seqs
        max-num-batched-tokens  → max_num_batched_tokens
        gpu-memory-utilization  → gpu_memory_utilization
        enforce-eager (bool)    → enforce_eager

    llama.cpp promoted:
        -c / --ctx-size         → n_ctx
        -b / --batch-size       → n_batch
        -t / --threads          → n_threads
        -ngl / --n-gpu-layers   → n_gpu_layers
        -ub / --ubatch-size     → n_ubatch
        --mmproj                → mmproj
        --chat-template-file    → chat_template_file
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("allma.allm_parser")


# vLLM flags that core/process.py treats specially and expects at top level.
_VLLM_PROMOTIONS = {
    "tensor-parallel-size":   "tensor_parallel",
    "max-model-len":          "max_model_len",
    "max-num-seqs":           "max_num_seqs",
    "max-num-batched-tokens": "max_num_batched_tokens",
    "gpu-memory-utilization": "gpu_memory_utilization",
    "enforce-eager":          "enforce_eager",  # boolean (no value)
    "tokenizer":              "tokenizer",
}

# llama.cpp/llama-server flags treated specially. Includes both short and long
# forms — canonical key is the same field on cfg.
_LLAMA_PROMOTIONS = {
    "c":                   "n_ctx",
    "ctx-size":            "n_ctx",
    "b":                   "n_batch",
    "batch-size":          "n_batch",
    "t":                   "n_threads",
    "threads":             "n_threads",
    "ngl":                 "n_gpu_layers",
    "n-gpu-layers":        "n_gpu_layers",
    "ub":                  "n_ubatch",
    "ubatch-size":         "n_ubatch",
    "mmproj":              "mmproj",
    "chat-template-file":  "chat_template_file",
}

_BOOLEAN_VLLM_FLAGS = {"enforce-eager"}


def _strip_leading_dashes(token: str) -> str:
    """Remove any leading `-` or `--` from a flag name."""
    return token.lstrip("-")


def _add_dashes(token: str, prefer_short: bool = False) -> str:
    """Ensure `token` has a `-` prefix appropriate for a CLI flag.

    Adds `--` when absent unless the raw name is a single character (then `-`)
    or the caller preserved short-form dashes.
    """
    if token.startswith("-"):
        return token
    return f"-{token}" if len(token) == 1 else f"--{token}"


def _coerce(value: str) -> Any:
    """Parse a raw string value into int, float, bool, or leave as str.

    JSON-like values stay as strings (the backend parses them). This matters for
    things like `--speculative-config {...}` where we need the JSON verbatim.
    """
    if value.startswith("{") or value.startswith("["):
        return value  # leave JSON verbatim
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _split_flag_and_value(rest: str) -> tuple[str, str | None]:
    """Split a flag line's remainder into flag name + optional value.

    The value is everything after the first whitespace. Whitespace within a
    JSON value is preserved.
    """
    stripped = rest.strip()
    if not stripped:
        return "", None
    # Find the split between name and value.
    for i, ch in enumerate(stripped):
        if ch in (" ", "\t"):
            name = stripped[:i]
            value = stripped[i + 1:].strip()
            return name, (value or None)
    return stripped, None


def parse_allm(content: str, source_hint: str = "<memory>") -> Dict[str, Any]:
    """Parse a single .allm v2 file. Returns a dict compatible with the
    consumers in core/process.py and elsewhere.

    Args:
        content: full file text.
        source_hint: filename or identifier used in log messages.

    Returns:
        cfg dict with backend, path, tokenizer, promoted flags, extra_args,
        and (for profiles) name/base/enable_thinking/sampling.
    """
    cfg: Dict[str, Any] = {}
    extra_args: list = []
    sampling: Dict[str, Any] = {}
    backend: str | None = None  # only set once we see @vllm / @llamacpp
    is_profile = False          # inferred from @name / @base
    thinking_off = False        # only set if @thinking-off encountered

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # ── meta directives ─────────────────────────────────────────────
        if line.startswith("@"):
            name, value = _split_flag_and_value(line[1:])
            if not name:
                logger.warning(f"{source_hint}: empty @-directive '{raw_line}'")
                continue

            if name == "vllm":
                backend = "vllm"
                cfg["backend"] = "vllm"
                continue
            if name == "llamacpp":
                backend = "llama.cpp"
                cfg["backend"] = "llama.cpp"
                continue

            if name == "path" and value:
                # For llama.cpp, the model path lives on cfg["model"] because
                # that's what core/process.py reads. Otherwise cfg["path"].
                if backend == "llama.cpp":
                    cfg["model"] = value
                else:
                    cfg["path"] = value
                    cfg.setdefault("tokenizer", value)
                continue
            if name == "tokenizer" and value:
                cfg["tokenizer"] = value
                continue

            if name == "gpu" and value is not None:
                try:
                    cfg["gpu_id"] = str(int(value))
                except ValueError:
                    logger.warning(f"{source_hint}: invalid @gpu value '{value}'")
                continue
            if name == "gpus" and value:
                # Multi-GPU list — currently pin to first, store rest for future use.
                parts = [p.strip() for p in value.split(",") if p.strip()]
                if parts:
                    try:
                        cfg["gpu_id"] = str(int(parts[0]))
                        cfg["gpus"] = [int(p) for p in parts]
                    except ValueError:
                        logger.warning(f"{source_hint}: invalid @gpus value '{value}'")
                continue

            if name == "pin":
                cfg["pin_loaded"] = True
                continue
            if name == "keep-alive" and value is not None:
                try:
                    cfg["keep_alive_seconds"] = int(value)
                except ValueError:
                    logger.warning(f"{source_hint}: invalid @keep-alive value '{value}'")
                continue

            # ── profile directives ──────────────────────────────────────
            if name == "name" and value:
                cfg["name"] = value
                is_profile = True
                continue
            if name == "base" and value:
                cfg["base"] = value
                is_profile = True
                continue
            if name == "thinking-off":
                thinking_off = True
                is_profile = True
                continue

            # Unknown @-directive
            logger.warning(f"{source_hint}: unknown @-directive '@{name}'")
            continue

        # ── non-@ line ──────────────────────────────────────────────────
        # In a profile (base=... or name=...) these are sampling params.
        # In a base they are backend flags. We don't know yet if we're a profile
        # until after the file is fully read — so collect ambiguous lines by
        # what's practical: sampling params never start with a dash, flags often
        # do. Use `is_profile` as the tiebreaker at end.
        # Simpler: check if the token, once stripped of dashes, matches a known
        # sampling key. This lets bases and profiles coexist even if a base uses
        # a flag with the same name.
        name, value = _split_flag_and_value(line)
        if not name:
            continue

        # Sampling detection: strip dashes, lowercase, hyphens -> underscores
        # for comparison, but preserve original for backend flag emission.
        raw_name = _strip_leading_dashes(name)
        norm = raw_name.replace("-", "_").lower()

        _SAMPLING_KEYS = {
            "temperature", "top_p", "top_k", "min_p",
            "presence_penalty", "repetition_penalty", "frequency_penalty",
            "typical_p", "top_a", "tfs", "seed",
        }
        if is_profile or norm in _SAMPLING_KEYS:
            if value is None:
                logger.warning(f"{source_hint}: sampling '{norm}' has no value")
                continue
            sampling[norm] = _coerce(value)
            continue

        # It's a backend flag. Handle promotions first.
        promotions = _VLLM_PROMOTIONS if backend != "llama.cpp" else _LLAMA_PROMOTIONS

        if raw_name in promotions:
            key = promotions[raw_name]
            if raw_name in _BOOLEAN_VLLM_FLAGS and backend != "llama.cpp":
                # Boolean flag: presence enables it.
                cfg[key] = True
            elif value is None:
                # Non-boolean promoted flag with no value — keep as flag in extra_args
                extra_args.append(_add_dashes(name))
            else:
                # Store the raw string; process.py handles str→int/float coercion.
                cfg[key] = _coerce(value)
            continue

        # Otherwise it goes into extra_args. Preserve the exact dash form given.
        emitted = _add_dashes(name)
        extra_args.append(emitted)
        if value is not None:
            extra_args.append(value)

    # ── post-processing ─────────────────────────────────────────────────
    if extra_args:
        cfg["extra_args"] = extra_args
    if sampling:
        cfg["sampling"] = sampling
    if is_profile:
        cfg["enable_thinking"] = not thinking_off
    # llama.cpp cfg needs a stable `model` field even if the user wrote @path
    # after @llamacpp; already handled above. If they wrote in reverse order the
    # early lookup above missed it — fix retroactively.
    if cfg.get("backend") == "llama.cpp" and "path" in cfg and "model" not in cfg:
        cfg["model"] = cfg.pop("path")

    return cfg


def load_models_from_configs(
    config_dir: str = "configs",
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Load every .allm file under `config_dir/base` and `config_dir/profile`.

    Returns (base_models, profile_models) — same shape as the legacy loader.
    """
    config_path = Path(config_dir)
    if not config_path.exists():
        logger.warning(f"Config directory not found: {config_dir}")
        return {}, {}

    base_models: Dict[str, Dict[str, Any]] = {}
    profile_models: Dict[str, Dict[str, Any]] = {}

    base_dir = config_path / "base"
    if base_dir.exists():
        for cfg_file in sorted(base_dir.glob("*.allm")):
            if not cfg_file.is_file():
                continue
            try:
                cfg = parse_allm(cfg_file.read_text(encoding="utf-8"), cfg_file.name)
            except Exception as e:
                logger.error(f"Failed to parse {cfg_file.name}: {e}")
                continue
            if "backend" not in cfg:
                logger.warning(f"{cfg_file.name}: no @vllm or @llamacpp declaration; skipping")
                continue
            name = cfg.get("name") or cfg_file.stem
            cfg["_source_path"] = str(cfg_file)
            base_models[name] = cfg

    profile_dir = config_path / "profile"
    if profile_dir.exists():
        for cfg_file in sorted(profile_dir.glob("*.allm")):
            if not cfg_file.is_file():
                continue
            try:
                cfg = parse_allm(cfg_file.read_text(encoding="utf-8"), cfg_file.name)
            except Exception as e:
                logger.error(f"Failed to parse {cfg_file.name}: {e}")
                continue
            if "base" not in cfg:
                logger.warning(f"{cfg_file.name}: no @base directive; skipping")
                continue
            name = cfg.get("name") or cfg_file.stem
            cfg["_source_path"] = str(cfg_file)
            if cfg["base"] not in base_models:
                logger.warning(f"Profile '{name}' references unknown base '{cfg['base']}'")
            profile_models[name] = cfg

    return base_models, profile_models
