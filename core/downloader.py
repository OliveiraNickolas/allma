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
    t = Text()
    t.append("[ ", style=C_DIM)
    t.append(name, style=f"bold {C_ACCENT}")
    t.append(" ]", style=C_DIM)
    return t


def _row(label: str, value: str) -> Text:
    t = Text(style=_S)
    t.append(f"  {label:<10}", style=f"{C_DIM} on {C_BG}")
    t.append("▸  ", style=f"{C_DIM} on {C_BG}")
    t.append(value, style=f"bold {C_ACCENT} on {C_BG}")
    return t


def _W() -> int:
    return min(max(shutil.get_terminal_size().columns - 4, 50), 90)


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

    W = _W()

    # Build files table
    tbl = Table(
        show_header=True,
        header_style=f"bold {C_DIM}",
        box=None,
        padding=(0, 2),
        style=_S,
        expand=True,
    )
    tbl.add_column("#",    style=f"bold {C_ACCENT}", width=4, justify="right")
    tbl.add_column("File", style=f"{C_FG}")
    tbl.add_column("Size", style=f"bold {C_ACCENT}", justify="right", width=10)

    for i, f in enumerate(gguf_files, 1):
        tbl.add_row(str(i), f["name"], _file_size_str(f["size"]))

    if mmproj_files:
        tbl.add_row("", "", "")
        for i, f in enumerate(mmproj_files, len(gguf_files) + 1):
            tbl.add_row(
                Text(str(i), style=f"bold {C_DIM}"),
                Text(f["name"], style=C_DIM),
                Text(_file_size_str(f["size"]), style=C_DIM),
            )

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
        raw = input("").strip()
    except (KeyboardInterrupt, EOFError):
        return []

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
def create_configs(model_dir: Path, model_name: str) -> tuple[Path, list[Path]]:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from create_config import (
        detect_model, FAMILY_PRESETS, suggest_tp, suggest_max_len,
        generate_base_allm, generate_profile_allm, get_gpus,
    )

    allma_root = Path(__file__).parent.parent
    base_dir    = allma_root / "configs" / "base"
    profile_dir = allma_root / "configs" / "profile"
    base_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    info   = detect_model(model_dir)
    preset = FAMILY_PRESETS.get(info["family"], FAMILY_PRESETS["generic"])
    gpus   = get_gpus()
    tp     = suggest_tp(info["size_gb"], gpus)
    max_len = suggest_max_len(info["max_ctx"], info["size_gb"], tp, gpus)

    gguf_path   = info["gguf_files"][0]   if info["gguf_files"]   else None
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
def run_download(url_or_id: str):
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
            sys.exit(0)

    _print_result(base_path, profile_paths)
