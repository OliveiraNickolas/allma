"""
HuggingFace model downloader with Allma-style UI.
"""
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

from rich import box as _box
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import (
    BarColumn, DownloadColumn, Progress, SpinnerColumn,
    TextColumn, TimeRemainingColumn, TransferSpeedColumn,
)
from rich.table import Table
from rich.text import Text

# ── Allma palette ─────────────────────────────────────────────────────────────
C_BG     = "#e8dfc8"
C_SCREEN = "#d0c4a8"
C_FG     = "#1a1408"
C_DIM    = "#6a5a48"
C_ACCENT = "#007878"
C_BORDER = "#008888"
# Verdict colors — dark variants readable on the cream background
# (terminal-bright green/yellow wash out on #e8dfc8).
C_GOOD   = "#1e7d23"
C_WARN   = "#8a6d00"
C_ORANGE = "#b45f06"
C_BAD    = "#c11a1e"
_S       = f"on {C_BG}"

console = Console()

MODELS_DIR = Path(os.environ.get("ALLMA_MODELS_DIR", str(Path.home() / "AI" / "Models")))

QUANT_ORDER = [
    "BF16", "F16", "F32",
    "Q8_0",
    "Q6_K_L", "Q6_K",
    "Q5_K_L", "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0",
    "Q4_K_L", "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0",
    "Q3_K_L", "Q3_K_M", "Q3_K_S",
    "Q2_K_L", "Q2_K",
    "IQ4_XS", "IQ4_NL", "IQ3_XXS", "IQ3_XS", "IQ2_XXS", "IQ2_XS", "IQ1_S",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _section(name: str) -> Text:
    """C64-keycap bevel around a section title (same style as show_banner)."""
    t = Text()
    t.append("▐▉ ", style=f"bold {C_ACCENT}")
    t.append(name, style=f"bold {C_ACCENT}")
    t.append(" ▉▌", style=f"bold {C_ACCENT}")
    return t


def _row(label: str, value: str) -> Text:
    t = Text(style=_S)
    t.append(f"  {label:<10}", style=f"{C_DIM} on {C_BG}")
    t.append("▸  ", style=f"{C_DIM} on {C_BG}")
    t.append(value, style=f"bold {C_ACCENT} on {C_BG}")
    return t


def _W() -> int:
    # Full terminal width (small margin), floor of 50 for tiny windows.
    return max(shutil.get_terminal_size().columns - 2, 50)


def _file_size_str(size_bytes: Optional[int]) -> str:
    if not size_bytes:
        return "  ?  "
    gb = size_bytes / (1024 ** 3)
    if gb >= 1:
        return f"{gb:.1f} GB"
    return f"{size_bytes / (1024**2):.0f} MB"


def _quant_key(filename: str) -> int:
    upper = filename.upper()
    for i, q in enumerate(QUANT_ORDER):
        if q in upper:
            return i
    return len(QUANT_ORDER)


# ── VRAM fit preview ──────────────────────────────────────────────────────────
def _fetch_repo_config(repo_id: str, files: dict) -> Optional[dict]:
    """Grab the repo's config.json (a few KB) so context math is exact."""
    if "config.json" not in files.get("config", []):
        return None
    try:
        import json
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=repo_id, filename="config.json")
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def _kv_bytes_per_token(conf: Optional[dict]) -> tuple[int, int]:
    """(bytes_per_token_fp16, native_max_ctx). Mirrors core.gpu's KV math,
    including hybrid models where only every Nth layer holds KV."""
    if not conf:
        return 65536, 0  # conservative dense-8B-ish fallback, unknown max
    tc = conf.get("text_config", conf)
    layers = tc.get("num_hidden_layers", 32)
    kv_heads = tc.get("num_key_value_heads") or tc.get("num_attention_heads", 8)
    hidden = tc.get("hidden_size", 4096)
    heads = tc.get("num_attention_heads", 16)
    head_dim = tc.get("head_dim") or (hidden // heads)
    interval = tc.get("full_attention_interval")
    kv_layers = max(1, layers // interval) if interval else layers
    bpt = 2 * kv_heads * head_dim * kv_layers * 2  # K+V, fp16
    return bpt, int(tc.get("max_position_embeddings") or 0)


def _max_ctx_tokens(size_bytes: Optional[int], gpu: Optional[dict],
                    kv_bpt: int, native_max: int) -> int:
    """Largest context that fits on the best GPU after weights + overhead."""
    if not size_bytes or not gpu or kv_bpt <= 0:
        return 0
    usable_gb = gpu["total_gb"] * 0.93 - size_bytes / (1024 ** 3) * 1.01 - 1.5
    if usable_gb <= 0:
        return 0
    tokens = int(usable_gb * (1024 ** 3) / kv_bpt)
    return min(tokens, native_max) if native_max else tokens


def _fmt_ctx(tokens: int) -> str:
    if tokens < 4096:
        return "✗"
    if tokens >= 1024 ** 2:
        return f"{tokens // 1024 // 1024}M"
    return f"{tokens // 1024}k"


def _gpu_stats() -> Optional[dict]:
    """Best single GPU (total/free GB) — None when nvidia-smi is unavailable."""
    try:
        from core.gpu import get_all_gpus
        gpus = get_all_gpus()
        if not gpus:
            return None
        best = max(gpus, key=lambda g: g.get("total_mb", 0))
        return {
            "total_gb": best.get("total_mb", 0) / 1024,
            "free_gb": best.get("free_gb", 0.0),
            "sum_total_gb": sum(g.get("total_mb", 0) for g in gpus) / 1024,
            "sum_free_gb": sum(g.get("free_gb", 0.0) for g in gpus),
        }
    except Exception:
        return None


def _fit_verdict(size_bytes: Optional[int], gpu: Optional[dict]) -> tuple[str, str]:
    """(label, rich_style) — does this file fit, sized against the best GPU?

    Reserve ~2 GB beyond the weights for KV cache + runtime overhead — the
    floor for a usable context. Multi-GPU spillover counts as a tight fit.
    """
    if not size_bytes or not gpu:
        return "?", C_DIM
    need = size_bytes / (1024 ** 3) * 1.01 + 2.0
    if need <= gpu["free_gb"]:
        return "✓ fits", f"bold {C_GOOD}"
    if need <= gpu["total_gb"]:
        return "◐ after unload", C_WARN
    if need <= gpu["sum_total_gb"]:
        return "◑ multi-gpu", C_WARN
    return "✗ too big", f"bold {C_BAD}"


def _recommendation_bars(gguf_files: list[dict], gpu: Optional[dict],
                         kv_bpt: int = 65536, native_max: int = 0) -> dict:
    """filename → 0-5 score. The highest-quality quant that fits comfortably
    (≤80% of the best GPU) gets 5; other fitting quants rank below it; files
    that only fit tightly or not at all get 1/0. Context capacity caps the
    score: a quant that leaves no room for a usable context can't rank high
    no matter its quality."""
    scores: dict = {}
    if not gpu:
        return {f["name"]: 0 for f in gguf_files}
    fitting = []
    for f in gguf_files:
        if not f.get("size"):
            scores[f["name"]] = 0
            continue
        need = f["size"] / (1024 ** 3) * 1.01 + 2.0
        if need <= gpu["total_gb"] * 0.80:
            fitting.append(f)
        elif need <= gpu["total_gb"]:
            scores[f["name"]] = 1   # fits, but no headroom for real context
        else:
            scores[f["name"]] = 0
    # fitting files: rank by quant quality (QUANT_ORDER: best first)
    fitting.sort(key=lambda f: _quant_key(f["name"]))
    for rank, f in enumerate(fitting):
        scores[f["name"]] = max(2, 5 - rank)
    # context cap: little KV headroom = low recommendation, quality regardless
    for f in gguf_files:
        if not f.get("size"):
            continue
        ctx = _max_ctx_tokens(f["size"], gpu, kv_bpt, native_max)
        if ctx < 8192:
            scores[f["name"]] = min(scores.get(f["name"], 0), 1)
        elif ctx < 32768:
            scores[f["name"]] = min(scores.get(f["name"], 0), 3)
    return scores


# ── URL parsing ───────────────────────────────────────────────────────────────
def parse_hf_url(url_or_id: str) -> str:
    url_or_id = url_or_id.strip().rstrip("/")
    match = re.search(r"huggingface\.co/([^/?#]+/[^/?#]+)", url_or_id)
    if match:
        return match.group(1)
    if re.match(r"^[\w.-]+/[\w.-]+$", url_or_id):
        return url_or_id
    raise ValueError(f"Cannot parse HuggingFace repo from: {url_or_id!r}")


# ── Repo listing ──────────────────────────────────────────────────────────────
def list_repo_files(repo_id: str) -> dict:
    from huggingface_hub import list_repo_files as hf_list, get_paths_info

    all_files = list(hf_list(repo_id))
    sizes = {}
    try:
        for info in get_paths_info(repo_id, all_files):
            if hasattr(info, "lfs") and info.lfs:
                sizes[info.path] = info.lfs.size
            elif hasattr(info, "size"):
                sizes[info.path] = info.size
    except Exception:
        pass

    gguf, safetensors, mmproj, config = [], [], [], []
    for f in all_files:
        name = f.lower()
        if name.endswith(".gguf"):
            entry = {"name": f, "size": sizes.get(f)}
            (mmproj if "mmproj" in name else gguf).append(entry)
        elif name.endswith(".safetensors"):
            safetensors.append({"name": f, "size": sizes.get(f)})
        elif f in ("config.json", "tokenizer.json", "tokenizer_config.json",
                   "tokenizer.model", "special_tokens_map.json", "generation_config.json"):
            config.append(f)

    gguf.sort(key=lambda x: _quant_key(x["name"]))
    return {"gguf": gguf, "safetensors": safetensors, "mmproj": mmproj, "config": config}


# ── Header banner ─────────────────────────────────────────────────────────────
def _print_header(repo_id: str, dest_dir: Path):
    W = _W()
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column()
    grid.add_row(_row("repo", repo_id))
    grid.add_row(_row("dest", str(dest_dir)))

    inner = Panel(
        grid,
        title=_section("Download"),
        title_align="left",
        box=_box.SQUARE,
        border_style=C_DIM,
        style=_S,
        padding=(0, 1),
    )
    outer = Panel(
        inner,
        box=_box.DOUBLE,
        border_style=C_BORDER,
        style=f"on {C_SCREEN}",
        padding=(0, 0),
        width=W,
    )
    console.print()
    console.print(outer)
    console.print()


# ── GGUF selection UI ─────────────────────────────────────────────────────────
def select_gguf_interactive(files: dict, repo_id: str) -> list[str]:
    gguf_files = files["gguf"]
    mmproj_files = files["mmproj"]
    all_numbered = gguf_files + mmproj_files

    if not gguf_files and not mmproj_files:
        console.print(Text("  No GGUF files found.", style=C_DIM))
        return []

    gpu = _gpu_stats()
    conf = _fetch_repo_config(repo_id, files) if gpu else None
    kv_bpt, native_max = _kv_bytes_per_token(conf)
    rec = _recommendation_bars(gguf_files, gpu, kv_bpt, native_max)

    def _build_panel() -> Panel:
        """Panel built at the CURRENT terminal width — called again on resize."""
        tbl = Table(
            show_header=True,
            header_style=f"bold {C_DIM}",
            box=None,
            padding=(0, 2),
            style=_S,
            expand=True,
            # zebra stripes: every other row slightly darker for scanability
            row_styles=[f"on {C_BG}", "on #ddd2b4"],
        )
        tbl.add_column("#",    style=f"bold {C_ACCENT}", width=4, justify="right")
        tbl.add_column("File", style=f"{C_FG}", overflow="fold")
        tbl.add_column("Size", style=f"bold {C_ACCENT}", justify="right", width=10)
        if gpu:
            tbl.add_column("Rec",  justify="left", width=7)
            tbl.add_column("Max ctx", justify="right", width=8)
            tbl.add_column("Fits", justify="left", width=15)

        for i, f in enumerate(gguf_files, 1):
            row = [str(i), f["name"], _file_size_str(f["size"])]
            if gpu:
                score = rec.get(f["name"], 0)
                # gauge coloring: short fill = light load (green), mid =
                # yellow, long fill = heavy (orange)
                if score == 0:
                    bar_style = C_DIM
                elif score <= 2:
                    bar_style = f"bold {C_GOOD}"
                elif score == 3:
                    bar_style = C_WARN
                else:
                    bar_style = f"bold {C_ORANGE}"
                row.append(Text("▰" * score + "▱" * (5 - score), style=bar_style))
                ctx = _max_ctx_tokens(f["size"], gpu, kv_bpt, native_max)
                ctx_style = (f"bold {C_GOOD}" if ctx >= 32768
                             else (C_WARN if ctx >= 8192 else f"bold {C_BAD}"))
                row.append(Text(_fmt_ctx(ctx), style=ctx_style))
                label, style = _fit_verdict(f["size"], gpu)
                row.append(Text(label, style=style))
            tbl.add_row(*row)

        if mmproj_files:
            tbl.add_row(*([""] * (6 if gpu else 3)))
            for i, f in enumerate(mmproj_files, len(gguf_files) + 1):
                row = [
                    Text(str(i), style=f"bold {C_DIM}"),
                    Text(f["name"], style=C_DIM, overflow="fold"),
                    Text(_file_size_str(f["size"]), style=C_DIM),
                ]
                if gpu:
                    row += [Text("", style=C_DIM), Text("", style=C_DIM),
                            Text("(vision)", style=C_DIM)]
                tbl.add_row(*row)

        hint = Text(style=_S)
        hint.append("  Enter numbers to download  ", style=f"{C_DIM} on {C_BG}")
        hint.append("1", style=f"bold {C_ACCENT} on {C_BG}")
        hint.append("  or  ", style=f"{C_DIM} on {C_BG}")
        hint.append("1 3", style=f"bold {C_ACCENT} on {C_BG}")
        hint.append("  or  ", style=f"{C_DIM} on {C_BG}")
        hint.append("1,3", style=f"bold {C_ACCENT} on {C_BG}")
        hint.append("  ·  Enter to cancel", style=f"{C_DIM} on {C_BG}")

        hint_tbl = Table.grid(expand=True, padding=(0, 1))
        hint_tbl.add_column()
        hint_tbl.add_row(hint)

        inner = Panel(
            Group(tbl, hint_tbl),
            title=_section("Files"),
            title_align="left",
            box=_box.SQUARE,
            border_style=C_DIM,
            style=_S,
            padding=(0, 1),
        )
        return Panel(
            inner,
            box=_box.DOUBLE,
            border_style=C_BORDER,
            style=f"on {C_SCREEN}",
            padding=(0, 0),
            width=_W(),
        )

    prompt = Text()
    prompt.append("  ▸ ", style=f"bold {C_ACCENT}")

    def _draw(redraw: bool = False) -> None:
        if redraw:
            # Resize: printed output can't reflow — clear and repaint at the
            # new width, prompt included.
            console.clear()
        console.print(_build_panel())
        console.print()
        console.print(prompt, end="")
        sys.stdout.flush()

    _draw()

    # Repaint on terminal resize while we wait at the prompt (SIGWINCH is
    # Unix-only; input() resumes after the handler thanks to PEP 475).
    import signal as _signal
    _old_winch = None
    if hasattr(_signal, "SIGWINCH"):
        try:
            _old_winch = _signal.signal(
                _signal.SIGWINCH, lambda *_: _draw(redraw=True))
        except (ValueError, OSError):
            _old_winch = None  # not the main thread — skip live resize

    try:
        raw = input("").strip()
    except (KeyboardInterrupt, EOFError):
        return []
    finally:
        if _old_winch is not None:
            try:
                _signal.signal(_signal.SIGWINCH, _old_winch)
            except (ValueError, OSError):
                pass

    if not raw:
        return []

    selected = []
    for token in re.split(r"[,\s]+", raw):
        token = token.strip()
        if not token:
            continue
        try:
            idx = int(token) - 1
            if 0 <= idx < len(all_numbered):
                selected.append(all_numbered[idx]["name"])
            else:
                console.print(Text(f"  {token} out of range, skipped", style=C_DIM))
        except ValueError:
            console.print(Text(f"  '{token}' is not a number, skipped", style=C_DIM))

    return selected


# ── Safetensors confirmation ───────────────────────────────────────────────────
def _confirm_safetensors(files: dict) -> bool:
    total = sum(f["size"] or 0 for f in files["safetensors"])
    total_str = _file_size_str(total) if total else "unknown size"
    W = _W()

    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column()
    grid.add_row(_row("files", f"{len(files['safetensors'])} safetensors"))
    grid.add_row(_row("size",  total_str))
    grid.add_row(Text(""))
    hint = Text(style=_S)
    hint.append("  Download full repo?  ", style=f"{C_DIM} on {C_BG}")
    hint.append("y", style=f"bold {C_ACCENT} on {C_BG}")
    hint.append(" / ", style=f"{C_DIM} on {C_BG}")
    hint.append("N", style=f"bold {C_FG} on {C_BG}")
    grid.add_row(hint)

    inner = Panel(
        grid,
        title=_section("Safetensors repo"),
        title_align="left",
        box=_box.SQUARE,
        border_style=C_DIM,
        style=_S,
        padding=(0, 1),
    )
    outer = Panel(
        inner,
        box=_box.DOUBLE,
        border_style=C_BORDER,
        style=f"on {C_SCREEN}",
        padding=(0, 0),
        width=W,
    )
    console.print(outer)
    console.print()

    prompt = Text()
    prompt.append("  ▸ ", style=f"bold {C_ACCENT}")
    console.print(prompt, end="")
    try:
        return input("").strip().lower() == "y"
    except (KeyboardInterrupt, EOFError):
        return False


# ── Download ──────────────────────────────────────────────────────────────────
def download_files(repo_id: str, filenames: list[str], dest_dir: Path):
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    progress = Progress(
        SpinnerColumn(style=f"bold {C_ACCENT}"),
        TextColumn("[progress.description]{task.description}", style=C_FG),
        BarColumn(bar_width=30, style=C_DIM, complete_style=C_ACCENT),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with progress:
        for filename in filenames:
            task = progress.add_task(f"  {filename}", total=None)
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(dest_dir),
            )
            progress.update(task, completed=1, total=1)


def download_safetensors_repo(repo_id: str, dest_dir: Path):
    from huggingface_hub import snapshot_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    progress = Progress(
        SpinnerColumn(style=f"bold {C_ACCENT}"),
        TextColumn("[progress.description]{task.description}", style=C_FG),
        console=console,
    )
    with progress:
        task = progress.add_task("  Downloading repository...", total=None)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest_dir),
            ignore_patterns=["*.gguf", "*.bin", "*.pt", "original/*", "gguf/*"],
        )
        progress.update(task, completed=1, total=1)


# ── Config generation ─────────────────────────────────────────────────────────
def create_configs(model_dir: Path, model_name: str,
                   gguf_override: Path | None = None) -> tuple[Path, list[Path]]:
    from core.detect import (
        detect_model, FAMILY_PRESETS, suggest_tp, suggest_max_len, get_gpus,
    )
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from create_config import generate_base_allm, generate_profile_allm

    allma_root = Path(__file__).parent.parent
    base_dir    = allma_root / "configs" / "base"
    profile_dir = allma_root / "configs" / "profile"
    base_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    info   = detect_model(model_dir)
    # CPU-only hosts can't run vLLM; steer the auto-config to llama.cpp so the
    # generated .allm actually works when the user runs it.
    from core.detect import detect_platform, suggest_backend
    _platform = detect_platform()
    info["backend"] = suggest_backend(info, _platform)
    preset = FAMILY_PRESETS.get(info["family"], FAMILY_PRESETS["generic"])
    gpus   = get_gpus()
    tp     = suggest_tp(info["size_gb"], gpus)
    _kv = "fp8" if "--kv-cache-dtype" in preset.get("vllm_extra_args", []) else "auto"
    max_len = suggest_max_len(info["max_ctx"], info["size_gb"], tp, gpus,
                              model_path=info["path"], kv_dtype=_kv)

    if gguf_override:
        gguf_path = str(gguf_override)
    else:
        gguf_path = info["gguf_files"][0] if info["gguf_files"] else None
    mmproj_path = info["mmproj_files"][0] if info["mmproj_files"] else None

    base_content = generate_base_allm(
        name=model_name, info=info, preset=preset,
        tp=tp, max_len=max_len,
        gguf_path=gguf_path, mmproj_path=mmproj_path,
    )
    base_path = base_dir / f"{model_name}.allm"
    base_path.write_text(base_content)

    profile_paths = []
    base_sampling = preset.get("sampling", {})
    for variant, overrides in preset.get("profile_variants", {}).items():
        profile_name = f"{model_name}-{variant}" if variant != "default" else model_name
        sampling = {**base_sampling, **overrides}
        content = generate_profile_allm(
            profile_name=profile_name,
            base_name=model_name,
            sampling=sampling,
        )
        p = profile_dir / f"{profile_name}.allm"
        p.write_text(content)
        profile_paths.append(p)

    return base_path, profile_paths


def ensure_local_configs(target: Path) -> Optional[str]:
    """Resolve a local model path (dir or .gguf file) to a runnable profile,
    generating base + profile configs when none exist yet.

    Returns the profile name to load, or None when the path holds no model.
    """
    target = target.expanduser().resolve()
    model_dir = target.parent if target.is_file() else target

    allma_root  = Path(__file__).parent.parent
    base_dir    = allma_root / "configs" / "base"
    profile_dir = allma_root / "configs" / "profile"

    sys.path.insert(0, str(allma_root))
    from configs.allm_parser import parse_allm

    def _profiles_of(base_name: str) -> list[str]:
        """Profile names for a base — the default variant (file named after
        the base) first, the rest in file order."""
        names = []
        for f in sorted(profile_dir.glob("*.allm")):
            try:
                pc = parse_allm(f.read_text(), f.name)
            except Exception:
                continue
            if pc.get("base") == base_name:
                entry = pc.get("name") or f.stem
                if f.stem == base_name:
                    names.insert(0, entry)
                else:
                    names.append(entry)
        return names

    # 1. Reuse: a base that already points at this model keeps user tuning.
    for f in sorted(base_dir.glob("*.allm")):
        try:
            cfg = parse_allm(f.read_text(), f.name)
        except Exception:
            continue
        raw = cfg.get("path") or cfg.get("model") or ""
        if not raw:
            continue
        p = Path(raw).expanduser()
        if p in (target, model_dir) or p.parent == model_dir:
            profiles = _profiles_of(f.stem)
            if profiles:
                return profiles[0]
            break  # base exists but has no profile — regenerate below under a new name

    # 2. Generate: needs actual model files in place.
    has_gguf = target.suffix == ".gguf" or bool(list(model_dir.glob("*.gguf")))
    has_sf   = bool(list(model_dir.rglob("*.safetensors")))
    if not (has_gguf or has_sf):
        return None

    model_name = re.sub(r"[^A-Za-z0-9._-]", "-", model_dir.name)
    # Never clobber an unrelated config that happens to share the dir name.
    candidate, n = model_name, 2
    while (base_dir / f"{candidate}.allm").exists():
        candidate = f"{model_name}-{n}"
        n += 1
    gguf_override = target if target.suffix == ".gguf" else None
    base_path, profile_paths = create_configs(model_dir, candidate,
                                              gguf_override=gguf_override)
    _print_result(base_path, profile_paths)
    return profile_paths[0].stem if profile_paths else None


def _print_result(base_path: Path, profile_paths: list[Path]):
    W = _W()
    allma_root = Path(__file__).parent.parent

    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column()
    grid.add_row(_row("base", str(base_path.relative_to(allma_root))))
    for p in profile_paths:
        grid.add_row(_row("profile", str(p.relative_to(allma_root))))

    first_profile = profile_paths[0].stem if profile_paths else base_path.stem
    grid.add_row(Text(""))
    tip = Text(style=_S)
    tip.append("  allma run ", style=f"{C_DIM} on {C_BG}")
    tip.append(first_profile, style=f"bold {C_ACCENT} on {C_BG}")
    grid.add_row(tip)

    inner = Panel(
        grid,
        title=_section("Ready"),
        title_align="left",
        box=_box.SQUARE,
        border_style=C_ACCENT,
        style=_S,
        padding=(0, 1),
    )
    outer = Panel(
        inner,
        box=_box.DOUBLE,
        border_style=C_BORDER,
        style=f"on {C_SCREEN}",
        padding=(0, 0),
        width=W,
    )
    console.print()
    console.print(outer)
    console.print()


# ── Entry point ───────────────────────────────────────────────────────────────
def run_download(url_or_id: str) -> Optional[str]:
    """Full download flow. Returns the first generated profile name (for
    chaining into `allma run`), or None when configs couldn't be created."""
    try:
        repo_id = parse_hf_url(url_or_id)
    except ValueError as e:
        console.print(Text(f"✕ {e}", style=f"bold red"))
        sys.exit(1)

    model_name = repo_id.split("/")[-1]
    dest_dir   = MODELS_DIR / model_name

    _print_header(repo_id, dest_dir)

    with console.status(Text("  Listing repo files...", style=C_DIM), spinner_style=f"bold {C_ACCENT}"):
        try:
            files = list_repo_files(repo_id)
        except Exception as e:
            console.print(Text(f"  ✕ Failed to list repo: {e}", style="bold red"))
            sys.exit(1)

    is_gguf_repo = bool(files["gguf"] or files["mmproj"])

    if is_gguf_repo:
        to_download = select_gguf_interactive(files, repo_id)
        if not to_download:
            console.print(Text("  Cancelled.", style=C_DIM))
            sys.exit(0)
        try:
            download_files(repo_id, to_download, dest_dir)
            for cfg_file in files["config"]:
                try:
                    from huggingface_hub import hf_hub_download
                    hf_hub_download(repo_id=repo_id, filename=cfg_file, local_dir=str(dest_dir))
                except Exception:
                    pass
        except Exception as e:
            console.print(Text(f"  ✕ Download failed: {e}", style="bold red"))
            sys.exit(1)
    else:
        if not _confirm_safetensors(files):
            console.print(Text("  Cancelled.", style=C_DIM))
            sys.exit(0)
        try:
            download_safetensors_repo(repo_id, dest_dir)
        except Exception as e:
            console.print(Text(f"  ✕ Download failed: {e}", style="bold red"))
            sys.exit(1)

    with console.status(Text("  Generating configs...", style=C_DIM), spinner_style=f"bold {C_ACCENT}"):
        try:
            base_path, profile_paths = create_configs(dest_dir, model_name)
        except Exception as e:
            console.print(Text(f"  ⚠ Config generation failed: {e}", style=f"bold {C_DIM}"))
            console.print(Text(f"  Model downloaded to {dest_dir}", style=C_DIM))
            return None

    _print_result(base_path, profile_paths)
    return profile_paths[0].stem if profile_paths else None
