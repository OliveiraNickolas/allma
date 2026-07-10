"""First-run guided flow: pick a goal, see curated models that fit your
hardware, download and chat — one command, zero config knowledge needed.

The heavy lifting (quant picker with fit/ctx preview, download, config
generation, chat) is all delegated to the existing `allma run <hf-repo>`
flow; quickstart only narrows the choice down to a sensible starting point.
"""
import sys
from typing import Optional

from rich import box as _box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.downloader import (
    C_ACCENT, C_BG, C_BORDER, C_DIM, C_FG, C_GOOD, C_SCREEN, C_WARN,
    _S, _W, _gpu_stats, _row, _section, console,
)

# ── Curated catalog ───────────────────────────────────────────────────────────
# Hand-picked starting points per goal. `weights_gb` is the LIGHTEST viable
# quant of the repo (what the fit check uses); the in-repo picker then shows
# every quant with exact numbers. Edit freely — entries are just suggestions.
CATALOG = [
    {
        "repo": "Qwen/Qwen3.5-9B",
        "label": "Qwen3.5 9B",
        "tags": {"chat", "code", "vision", "general"},
        "weights_gb": 5.0,          # ~Q4 GGUF / FP8
        "blurb": "Fast all-rounder, native vision, 262k ctx",
    },
    {
        "repo": "Qwen/Qwen3.5-27B",
        "label": "Qwen3.5 27B",
        "tags": {"chat", "code", "vision", "general"},
        "weights_gb": 15.0,
        "blurb": "Best dense quality on 24GB-class GPUs",
    },
    {
        "repo": "Qwen/Qwen3.5-35B-A3B",
        "label": "Qwen3.5 35B-A3B (MoE)",
        "tags": {"chat", "code", "general"},
        "weights_gb": 18.0,
        "blurb": "MoE: 27B-class quality at ~9B speed",
    },
    {
        "repo": "empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF",
        "label": "Qwythos 9B (Claude-Mythos)",
        "tags": {"chat", "general"},
        "weights_gb": 5.2,
        "blurb": "Creative writing voice, up to 1M ctx",
    },
]

GOALS = [
    ("chat",    "Chat & writing"),
    ("code",    "Coding & agents"),
    ("vision",  "Vision (images)"),
    ("general", "A bit of everything"),
]


def _panel(inner_renderable, title: str, accent: bool = False) -> Panel:
    inner = Panel(
        inner_renderable,
        title=_section(title),
        title_align="left",
        box=_box.SQUARE,
        border_style=C_ACCENT if accent else C_DIM,
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


def _ask_index(prompt: str, n: int) -> Optional[int]:
    """1-based selection; None on cancel/invalid-empty."""
    p = Text()
    p.append(f"  {prompt} ", style=C_DIM)
    p.append("▸ ", style=f"bold {C_ACCENT}")
    console.print(p, end="")
    try:
        raw = input("").strip()
    except (KeyboardInterrupt, EOFError):
        return None
    if not raw:
        return None
    try:
        idx = int(raw)
        return idx if 1 <= idx <= n else None
    except ValueError:
        return None


def _fits(entry: dict, gpu: Optional[dict]) -> tuple[bool, str]:
    """(usable, note) — sized against the lightest viable quant + 2GB reserve."""
    if not gpu:
        return True, "?"
    need = entry["weights_gb"] + 2.0
    if need <= gpu["total_gb"]:
        return True, "single GPU"
    if need <= gpu["sum_total_gb"]:
        return True, "multi-GPU"
    return False, "too big"


def run_quickstart() -> Optional[str]:
    """Interactive flow. Returns the chosen HF repo id, or None on cancel."""
    gpu = _gpu_stats()

    # ── hardware summary ─────────────────────────────────────────────────
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column()
    if gpu:
        grid.add_row(_row("vram", f"best GPU {gpu['total_gb']:.0f} GB · "
                                  f"total {gpu['sum_total_gb']:.0f} GB · "
                                  f"free now {gpu['sum_free_gb']:.1f} GB"))
    else:
        grid.add_row(_row("vram", "no NVIDIA GPU detected — CPU/llama.cpp only"))
    grid.add_row(Text(""))
    for i, (_, label) in enumerate(GOALS, 1):
        line = Text(style=_S)
        line.append(f"  {i}  ", style=f"bold {C_ACCENT} on {C_BG}")
        line.append(label, style=f"{C_FG} on {C_BG}")
        grid.add_row(line)
    console.print()
    console.print(_panel(grid, "Quickstart — what will you use it for?"))
    console.print()

    sel = _ask_index("goal", len(GOALS))
    if sel is None:
        console.print(Text("  Cancelled.", style=C_DIM))
        return None
    goal = GOALS[sel - 1][0]

    # ── curated suggestions, filtered by goal + hardware ─────────────────
    entries = []
    for e in CATALOG:
        if goal not in e["tags"]:
            continue
        ok, note = _fits(e, gpu)
        if ok:
            entries.append((e, note))
    if not entries:
        console.print(Text("  Nothing in the catalog fits this hardware — "
                           "try `allma run <hf-repo>` with a small GGUF.",
                           style=C_WARN))
        return None

    tbl = Table(
        show_header=True, header_style=f"bold {C_DIM}", box=None,
        padding=(0, 2), style=_S, expand=True,
        row_styles=[f"on {C_BG}", "on #ddd2b4"],
    )
    tbl.add_column("#", style=f"bold {C_ACCENT}", width=4, justify="right")
    tbl.add_column("Model", style=C_FG)
    tbl.add_column("Why", style=C_DIM)
    tbl.add_column("Fit", width=12)
    for i, (e, note) in enumerate(entries, 1):
        tbl.add_row(str(i), e["label"], e["blurb"],
                    Text(note, style=f"bold {C_GOOD}"))
    console.print(_panel(tbl, "Suggested starting points", accent=True))
    console.print()

    sel = _ask_index("model", len(entries))
    if sel is None:
        console.print(Text("  Cancelled.", style=C_DIM))
        return None
    repo = entries[sel - 1][0]["repo"]
    console.print(Text(f"  → {repo}", style=f"bold {C_ACCENT}"))
    console.print()
    return repo
