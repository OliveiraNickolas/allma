#!/usr/bin/env python3
"""
ALLAMA · FILE & CONFIG UTILITY
DOS-style TUI for Allama LLM Proxy — Lisa-16 Edition
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    ProgressBar,
    Select,
    Static,
)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
CONFIG_DIR  = BASE_DIR / "configs"
ALLAMA_URL  = "http://127.0.0.1:9000"

# ──────────────────────────────────────────────────────────────────────────────
# Logo — same pixel art as the rich banner in core/server.py
# ──────────────────────────────────────────────────────────────────────────────
_LOGO_COLORS = ["#e52529", "#f7941d", "#f7d000", "#43b047", "#009ddc"]

def _build_logo_rows() -> list[str]:
    CHARS = {
        'A': [" ███ ", "█   █", "█████", "█   █", "█   █"],
        'L': ["█    ", "█    ", "█    ", "█    ", "█████"],
        'M': ["█   █", "██ ██", "█ █ █", "█   █", "█   █"],
    }
    word, starts = list("ALLAMA"), [0, 6, 12, 18, 24, 30]
    canvas = [[0] * 36 for _ in range(5)]
    for ch, col in zip(word, starts):
        for r, row in enumerate(CHARS[ch]):
            for c, px in enumerate(row):
                if px == "█":
                    canvas[r][col + c] = 1
        if ch == "L":
            for r, row in enumerate(CHARS[ch]):
                for c, px in enumerate(row):
                    if px == "█" and col + c + 1 < 36 and canvas[r][col + c + 1] == 0:
                        canvas[r][col + c + 1] = 2
    return ["".join("█" if v == 1 else "▒" if v == 2 else " " for v in row)
            for row in canvas]

def _build_logo_markup() -> str:
    """Returns Rich markup string for the ALLAMA pixel logo."""
    from rich.text import Text
    t = Text(justify="center")
    rows = _build_logo_rows()
    for i, row in enumerate(rows):
        color = _LOGO_COLORS[i % len(_LOGO_COLORS)]
        for ch in row:
            if ch == "█":
                t.append(ch, style=f"bold {color}")
            elif ch == "▒":
                t.append(ch, style=f"dim {color}")
            else:
                t.append(ch)
        if i < len(rows) - 1:
            t.append("\n")
    return t

# ──────────────────────────────────────────────────────────────────────────────
# CSS — Mario / Allama palette
# ──────────────────────────────────────────────────────────────────────────────
CSS = """
/* ─────────────────────────────────────────────────────────────────
   DOS Wizard palette (warm cream + teal)
     BG_SCREEN  #d0c4a8   outer desktop (darker cream)
     BG_PANEL   #e8dfc8   panel / dialog background (warm cream)
     BG_SEL     #7a1818   selected row / active highlight (dark red)
     FG         #1a1408   main text (near-black warm)
     FG_DIM     #6a5a48   secondary / dim (warm gray-brown)
     ACCENT     #007878   borders, titles, F-keys (teal)
     BORDER     #008888   panel outline (teal)
     OK         #007878   online / success (teal)
     ERR        #802828   error
   ─────────────────────────────────────────────────────────────────*/

/* ─── BASE ─────────────────────────────────────────────────────── */
Screen {
    /* Dark desktop/shadow color — peeks through on right+bottom */
    background: #2a2018;
    color: #1a1408;
    /* Creates the "floating window" shadow: 3 cols right, 1 row bottom */
    padding: 0 3 1 0;
}

/* ─── BOOT ──────────────────────────────────────────────────────── */
#boot-container {
    background: #d0c4a8;
    align: center middle;
    width: 100%;
    height: 100%;
}

#boot-logo {
    content-align: center middle;
    width: 100%;
    padding: 1 0;
}

#boot-text {
    width: 80%;
    color: #007878;
    padding: 0 2;
}

#boot-bar {
    width: 80%;
    padding: 0 2 1 2;
    color: #007878;
}

/* ─── MAIN WINDOW (floating DOS dialog) ─────────────────────── */
#main-window {
    border: double #008888;
    border-title-color: #f0e8d0;
    border-title-style: bold;
    border-title-align: center;
    background: #d0c4a8;
    width: 100%;
    height: 100%;
}

/* ─── SUBTITLE BAR ──────────────────────────────────────────── */
#subtitle-bar {
    background: #d0c4a8;
    color: #007878;
    text-style: bold;
    text-align: center;
    content-align: center middle;
    height: 1;
    width: 100%;
}

/* ─── FKEY BAR ──────────────────────────────────────────────── */
#fkey-bar {
    background: #c8b898;
    color: #1a1408;
    height: 1;
    padding: 0 1;
    border-top: solid #008888;
}

.screen-title {
    background: #007878;
    color: #f0e8d0;
    text-style: bold;
    text-align: center;
    padding: 0 1;
    height: 3;
    content-align: center middle;
}

.screen-subtitle {
    color: #6a5a48;
    text-align: center;
    height: 1;
}

/* ─── MAIN LAYOUT ───────────────────────────────────────────────── */
#main-layout {
    background: #d0c4a8;
    height: 1fr;
}

#left-panel {
    width: 3fr;
    background: #e8dfc8;
    border: double #008888;
    padding: 0 1;
    border-title-color: #007878;
    border-title-style: bold;
}

#right-panel {
    width: 2fr;
    background: #e8dfc8;
    border: double #008888;
    padding: 0;
    overflow-y: auto;
}

/* ─── PANEL BORDER TITLES ───────────────────────────────────────── */
#props-box, #commands-box, #notes-box {
    border-title-color: #007878;
    border-title-style: bold;
}

/* ─── SUB-BOXES (right panel sections) ──────────────────────────── */
#props-box {
    border: solid #008888;
    padding: 0 1;
    height: auto;
    background: #e8dfc8;
}

#commands-box {
    border: solid #008888;
    padding: 0 1;
    height: auto;
    background: #e8dfc8;
}

#notes-box {
    border: solid #008888;
    padding: 0 1;
    height: auto;
    background: #e8dfc8;
}

.section-title {
    color: #007878;
    text-style: bold;
    padding: 0 1;
    width: 100%;
    background: #e8dfc8;
}

.separator {
    color: #008888;
    height: 1;
}

/* ─── CATALOG TREE ──────────────────────────────────────────────── */
#catalog-tree {
    color: #1a1408;
    padding: 0 1;
    background: #e8dfc8;
}

/* ─── PROPERTIES / NOTES ────────────────────────────────────────── */
#properties-text {
    color: #1a1408;
    padding: 0 1;
    background: #e8dfc8;
}

#commands-text {
    color: #1a1408;
    padding: 0 1;
    background: #e8dfc8;
}

#notes-text {
    color: #6a5a48;
    padding: 0 1;
    background: #e8dfc8;
}

/* ─── MENU ITEMS ────────────────────────────────────────────────── */
.menu-item {
    color: #1a1408;
    padding: 0 1;
    height: 1;
}

.menu-item:hover {
    background: #7a1818;
    color: #f0e8d0;
}

.menu-item.-active {
    background: #7a1818;
    color: #f0e8d0;
    text-style: bold;
}

.menu-key {
    color: #007878;
    text-style: bold;
}

.menu-label {
    color: #6a5a48;
}

/* ─── STATUS ─────────────────────────────────────────────────────── */
.status-label {
    color: #6a5a48;
}

.status-value {
    color: #1a1408;
}

.status-ok {
    color: #007878;
}

.status-warn {
    color: #7a5020;
}

.status-err {
    color: #802828;
}

/* ─── SCROLLABLE SCREENS ────────────────────────────────────────── */
#assistant-scroll {
    height: 1fr;
    background: #e8dfc8;
    border: double #008888;
    padding: 0 1;
}

.section-header {
    color: #007878;
    text-style: bold;
    padding: 1 0 0 0;
    border-bottom: solid #008888;
}

.profile-card {
    border: solid #008888;
    background: #f0e8d0;
    padding: 0 1;
    margin: 1 0;
    height: auto;
}

.profile-name {
    color: #007878;
    text-style: bold;
}

.profile-desc {
    color: #6a5a48;
}

.profile-stat {
    color: #7a5020;
}

.profile-args {
    color: #007878;
}

/* ─── DATA TABLE ────────────────────────────────────────────────── */
DataTable {
    background: #e8dfc8;
    color: #1a1408;
    border: double #008888;
}

DataTable > .datatable--header {
    background: #d0c4a8;
    color: #007878;
    text-style: bold;
}

DataTable > .datatable--cursor {
    background: #7a1818;
    color: #f0e8d0;
}

DataTable > .datatable--even-row {
    background: #ddd4b8;
}

/* ─── CONFIG WIZARD ─────────────────────────────────────────────── */
#wizard-container {
    height: 100%;
    background: #e8dfc8;
    border: double #008888;
    padding: 0 2;
}

.wizard-step {
    color: #6a5a48;
    padding: 1 0 0 0;
}

.wizard-field {
    padding: 0 0 1 0;
}

.wizard-field Label {
    color: #1a1408;
    padding: 0;
}

.wizard-field Input {
    background: #f0e8d0;
    color: #007878;
    border: solid #008888;
}

.wizard-field Input:focus {
    border: solid #007878;
}

/* ─── BUTTONS ───────────────────────────────────────────────────── */
Button {
    background: #e8dfc8;
    color: #007878;
    border: solid #008888;
    min-width: 12;
}

Button:hover {
    background: #7a1818;
    color: #f0e8d0;
    border: solid #7a1818;
}

Button.-primary {
    background: #e8dfc8;
    color: #007878;
    border: solid #007878;
    text-style: bold;
}

Button.-primary:hover {
    background: #7a1818;
    color: #f0e8d0;
    border: solid #7a1818;
}

/* ─── FOOTER ────────────────────────────────────────────────────── */
Footer {
    background: #c8b898;
    color: #1a1408;
}

Footer > .footer--key {
    background: #007878;
    color: #f0e8d0;
    text-style: bold;
}

/* ─── STATUS BAR ────────────────────────────────────────────────── */
#status-bar {
    height: 1;
    background: #c8b898;
    color: #1a1408;
    padding: 0 2;
    border-top: solid #008888;
}

#status-bar.online {
    color: #007878;
}

#status-bar.offline {
    color: #802828;
}

/* ─── BACK BAR ──────────────────────────────────────────────────── */
#back-bar {
    height: 3;
    align: left middle;
    padding: 0 1;
    border-top: solid #008888;
    background: #e8dfc8;
}
"""

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

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
                gpus.append({
                    "index":    int(parts[0]),
                    "name":     parts[1],
                    "total_gb": int(parts[2]) / 1024,
                    "free_gb":  int(parts[3]) / 1024,
                })
        return gpus
    except Exception:
        return []


def check_allama_server() -> tuple[bool, dict]:
    try:
        import urllib.request
        req = urllib.request.Request(f"{ALLAMA_URL}/health", method="GET")
        req.add_header("User-Agent", "AllamaTUI/1.0")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return True, data
    except Exception:
        return False, {}


def scan_physical_models(config_dir: Path) -> list[dict]:
    models = []
    phys_dir = config_dir / "physical"
    if not phys_dir.exists():
        return models
    for f in sorted(phys_dir.glob("*.allm")):
        try:
            from configs.loader import parse_all_file
            cfg = parse_all_file(f.read_text())
            if "backend" not in cfg:
                continue
            backend = cfg.get("backend", "vllm")
            path = cfg.get("path") or cfg.get("model", "")
            size_gb = _estimate_size(path, backend)
            models.append({
                "name":    f.stem,
                "backend": backend,
                "path":    path,
                "size_gb": size_gb,
                "cfg":     cfg,
            })
        except Exception:
            pass
    return models


def scan_logical_models(config_dir: Path) -> list[dict]:
    models = []
    log_dir = config_dir / "logical"
    if not log_dir.exists():
        return models
    for f in sorted(log_dir.glob("*.allm")):
        try:
            from configs.loader import parse_all_file
            cfg = parse_all_file(f.read_text())
            if "physical" not in cfg:
                continue
            models.append({"name": f.stem, "physical": cfg["physical"], "cfg": cfg})
        except Exception:
            pass
    return models


def _estimate_size(path: str, backend: str) -> float:
    if not path or not os.path.exists(path):
        return 0.0
    p = Path(path)
    if backend == "llama.cpp":
        return p.stat().st_size / (1024 ** 3) if p.is_file() else 0.0
    return sum(f.stat().st_size for f in p.rglob("*.safetensors")) / (1024 ** 3)


def vram_need(model: dict) -> float:
    """Quick VRAM estimate: size * 1.15 overhead."""
    return model["size_gb"] * 1.15 if model["size_gb"] > 0 else 4.0


def build_profiles(gpus: list, models: list) -> list[dict]:
    """Generate configuration profiles based on available hardware and models."""
    if not gpus or not models:
        return []

    total_free   = sum(g["free_gb"] for g in gpus)
    max_single   = max(g["free_gb"] for g in gpus)
    gpu_count    = len(gpus)
    profiles     = []

    by_vram = sorted(models, key=lambda m: vram_need(m))

    # ── Profile: single best model ──────────────────────────────────────────
    for m in reversed(by_vram):
        need = vram_need(m)
        if need <= total_free * 0.92:
            tp = 1 if max_single >= need else min(gpu_count, 2)
            profiles.append({
                "id":       "single-heavy",
                "name":     "SINGLE · MAXIMUM POWER",
                "emoji":    "▶▶",
                "models":   [m["name"]],
                "total_gb": need,
                "tp":       tp,
                "desc":     f"Run {m['name']} solo — all VRAM dedicated to one model",
                "ctx_hint": "Maximum context window, high quality outputs",
                "tags":     ["quality", "single-model"],
            })
            break

    # ── Profile: dual light models ───────────────────────────────────────────
    light = [m for m in by_vram if vram_need(m) <= max_single * 0.92]
    seen_bases = set()
    light_dedup = []
    for m in light:
        base = m["name"].split("-caption")[0].split("-Caption")[0]
        if base not in seen_bases:
            seen_bases.add(base)
            light_dedup.append(m)
    if len(light_dedup) >= 2 and gpu_count >= 2:
        a, b = light_dedup[-1], light_dedup[-2]
        combined = vram_need(a) + vram_need(b)
        if combined <= total_free * 0.92:
            profiles.append({
                "id":       "dual-light",
                "name":     "DUAL · ALWAYS READY",
                "emoji":    "▶▶▶",
                "models":   [a["name"], b["name"]],
                "total_gb": combined,
                "tp":       1,
                "desc":     "Two models loaded simultaneously — instant switching",
                "ctx_hint": "Split GPU 0 and GPU 1, no context sharing",
                "tags":     ["speed", "multi-model"],
            })

    # ── Profile: vision + text ───────────────────────────────────────────────
    vision = [m for m in models if m["backend"] == "vllm"
              and "vl" in m["name"].lower()]
    text   = [m for m in models if "vl" not in m["name"].lower()
              and vram_need(m) <= max_single * 0.92]
    if vision and text:
        v = vision[0]
        t = max(text, key=lambda m: vram_need(m))
        if v["name"] != t["name"]:
            combined = vram_need(v) + vram_need(t)
            if combined <= total_free * 0.92:
                profiles.append({
                    "id":       "vision-chat",
                    "name":     "VISION + CHAT",
                    "emoji":    ">>",
                    "models":   [v["name"], t["name"]],
                    "total_gb": combined,
                    "tp":       1,
                    "desc":     "Vision model + text model — image analysis + conversation",
                    "ctx_hint": "Ideal for multimodal workflows",
                    "tags":     ["vision", "multi-model"],
                })

    # ── Profile: speed / low latency ─────────────────────────────────────────
    already_in_profiles = {n for p in profiles for n in p["models"]}
    solo_candidates = [m for m in by_vram
                       if vram_need(m) <= max_single * 0.92
                       and m["name"] not in already_in_profiles]
    if not solo_candidates:
        solo_candidates = [by_vram[0]] if by_vram else []
    if solo_candidates:
        m = solo_candidates[0]
        profiles.append({
            "id":       "speed",
            "name":     "SPEED · LOW LATENCY",
            "emoji":    ">",
            "models":   [m["name"]],
            "total_gb": vram_need(m),
            "tp":       1,
            "desc":     f"{m['name']} — smallest available, fastest first-token",
            "ctx_hint": "Increase max_num_seqs for high concurrency workloads",
            "tags":     ["speed", "concurrent"],
        })

    return profiles


def bar(value: float, total: float, width: int = 20) -> str:
    """Render an ASCII progress bar."""
    if total == 0:
        return "░" * width
    filled = round(value / total * width)
    return "█" * filled + "░" * (width - filled)


def gb_bar(used: float, total: float, width: int = 12) -> str:
    return f"[{bar(used, total, width)}] {used:.0f}/{total:.0f}GB"


# ──────────────────────────────────────────────────────────────────────────────
# Boot Screen
# ──────────────────────────────────────────────────────────────────────────────
class BootScreen(Screen):
    BINDINGS = [Binding("enter", "proceed", "Continue"),
                Binding("space", "proceed", "Continue")]

    _lines:     reactive[list] = reactive([])
    _done:      reactive[bool] = reactive(False)
    _cursor:    reactive[bool] = reactive(True)
    _progress:  reactive[int]  = reactive(0)

    def compose(self) -> ComposeResult:
        with Container(id="boot-container"):
            yield Static(_build_logo_markup(), id="boot-logo")
            yield Static("", id="boot-text")
            yield Static("", id="boot-bar")

    def on_mount(self) -> None:
        self._gpus    = get_gpus()
        self._online, self._srv = check_allama_server()
        self._phys    = scan_physical_models(CONFIG_DIR)
        self._logical = scan_logical_models(CONFIG_DIR)
        self._step    = 0
        self._blink_on = True
        self.set_interval(0.08, self._tick)
        self.set_interval(0.5, self._blink)

    def _blink(self) -> None:
        self._blink_on = not self._blink_on
        if self._done:
            widget = self.query_one("#boot-text")
            lines  = list(self._display_lines)
            if lines:
                last = lines[-1]
                if "▌" in last or "─" not in last:
                    cursor = "▌" if self._blink_on else " "
                    lines[-1] = f"{cursor} PRESS ANY KEY TO CONTINUE"
                    widget.update("\n".join(lines))

    _display_lines: list = []

    def _tick(self) -> None:
        if self._done:
            return

        step = self._step
        messages = self._build_messages()

        if step < len(messages):
            self._display_lines.append(messages[step])
            widget = self.query_one("#boot-text")
            widget.update("\n".join(self._display_lines))
            self._step += 1
            self._progress = int(step / max(len(messages) - 1, 1) * 100)
            pct = self._progress
            bar_w = 48
            filled = round(pct / 100 * bar_w)
            bar_str = "█" * filled + "░" * (bar_w - filled)
            self.query_one("#boot-bar").update(f"[{bar_str}] {pct:3d}%")
        else:
            self._done = True
            self._display_lines.append("▌ PRESS ANY KEY TO CONTINUE")
            self.query_one("#boot-text").update("\n".join(self._display_lines))
            self.query_one("#boot-bar").update("")

    def _build_messages(self) -> list[str]:
        msgs = [
            "ALLAMA · FILE & CONFIG UTILITY  v1.0",
            "(c) 2025",
            "",
            "─" * 50,
            "INITIALIZING HARDWARE SUBSYSTEMS...",
        ]
        if self._gpus:
            for g in self._gpus:
                used = g["total_gb"] - g["free_gb"]
                msgs.append(
                    f"  GPU {g['index']}: {g['name'][:24]}"
                    f"  [{g['total_gb']:.0f}GB]  ... DETECTED"
                )
        else:
            msgs.append("  NO CUDA DEVICES FOUND  [CPU MODE]")
        msgs += [
            "SCANNING CONFIGURATION FILES...",
            f"  PHYSICAL MODELS : {len(self._phys):3d}  LOADED",
            f"  LOGICAL MODELS  : {len(self._logical):3d}  LOADED",
            "CONNECTING TO ALLAMA SERVER...",
        ]
        if self._online:
            active = self._srv.get("active_servers", 0)
            msgs.append(f"  {ALLAMA_URL}  ...  ONLINE  [{active} active]")
        else:
            msgs.append(f"  {ALLAMA_URL}  ...  OFFLINE")
        msgs += ["─" * 50, "SYSTEM READY.", ""]
        return msgs

    def action_proceed(self) -> None:
        if self._done:
            self.app.push_screen(
                MainMenuScreen(self._gpus, self._online, self._srv,
                               self._phys, self._logical)
            )

    def on_key(self, event) -> None:
        if self._done:
            self.action_proceed()


# ──────────────────────────────────────────────────────────────────────────────
# Main Menu Screen  —  DOS file manager layout
# ──────────────────────────────────────────────────────────────────────────────
class MainMenuScreen(Screen):
    BINDINGS = [
        Binding("1",   "menu_1", "Config Asst",   show=False),
        Binding("2",   "menu_2", "Model Lib",      show=False),
        Binding("3",   "menu_3", "Create Config",  show=False),
        Binding("4",   "menu_4", "Profiles",       show=False),
        Binding("5",   "menu_5", "System Info",    show=False),
        Binding("q",   "quit",   "Quit",           show=False),
        Binding("f1",  "menu_1", "Config Asst"),
        Binding("f2",  "menu_2", "Model Lib"),
        Binding("f3",  "menu_3", "Create Config"),
        Binding("f4",  "menu_4", "Profiles"),
        Binding("f5",  "menu_5", "System Info"),
        Binding("f10", "quit",   "Quit"),
    ]

    def __init__(self, gpus, online, srv, phys, logical, **kwargs):
        super().__init__(**kwargs)
        self._gpus    = gpus
        self._online  = online
        self._srv     = srv
        self._phys    = phys
        self._logical = logical

    def compose(self) -> ComposeResult:
        with Container(id="main-window"):
            yield Static(
                "  SYSTEM READY" if self._online else "  SERVER OFFLINE",
                id="subtitle-bar",
            )
            with Horizontal(id="main-layout"):
                # ── Left: catalog tree ───────────────────────────────────────────
                with Vertical(id="left-panel"):
                    yield Static(self._catalog_tree(), id="catalog-tree")

                # ── Right: three bordered sub-boxes ──────────────────────────────
                with Vertical(id="right-panel"):
                    with Vertical(id="props-box"):
                        yield Static(self._properties(), id="properties-text")
                    with Vertical(id="commands-box"):
                        yield Static(self._commands(), id="commands-text")
                    with Vertical(id="notes-box"):
                        yield Static(self._notes(), id="notes-text")

            yield Static(self._status_line(), id="status-bar",
                         classes="online" if self._online else "offline")
            yield Static(self._fkey_bar(), id="fkey-bar")

    def on_mount(self) -> None:
        self.query_one("#main-window").border_title = "ALLAMA · FILE & CONFIG UTILITY  v1.0"
        self.query_one("#left-panel").border_title = "[ Model Catalog ]"
        self.query_one("#props-box").border_title = "[ Server Properties ]"
        self.query_one("#commands-box").border_title = ":: COMMANDS ::"
        self.query_one("#notes-box").border_title = "[ Technical Notes ]"

    # ── Catalog tree ─────────────────────────────────────────────────────────
    def _catalog_tree(self) -> str:
        lines = ["", "  ■ ALLAMA.CONFIG", ""]

        # Logical models grouped by physical
        lines.append("  ├─ [LOGICAL\\]")
        for i, m in enumerate(self._logical):
            is_last = (i == len(self._logical) - 1)
            conn = "│   └─" if is_last else "│   ├─"
            lines.append(f"  {conn}  {m['name']}")
        lines.append("  │")

        # Physical models
        lines.append("  └─ [PHYSICAL\\]")
        for i, m in enumerate(self._phys):
            is_last = (i == len(self._phys) - 1)
            conn = "      └─" if is_last else "      ├─"
            tag = "[gguf]" if m["backend"] == "llama.cpp" else "[vllm]"
            size = f"{m['size_gb']:.0f}GB" if m["size_gb"] > 0 else "?GB"
            lines.append(f"  {conn}  {m['name']}  {tag}  {size}")

        lines.append("")
        return "\n".join(lines)

    # ── Server properties ────────────────────────────────────────────────────
    def _properties(self) -> str:
        lines = []
        if self._online:
            active = self._srv.get("active_servers", 0)
            lines += [
                "  Status  :  ● ONLINE",
                f"  Port    :  9000",
                f"  Active  :  {active} model(s)",
            ]
        else:
            lines += [
                "  Status  :  ○ OFFLINE",
                "  Port    :  9000",
            ]
        lines.append("")
        if self._gpus:
            for g in self._gpus:
                used = g["total_gb"] - g["free_gb"]
                b    = bar(used, g["total_gb"], 12)
                lines.append(f"  GPU {g['index']}    :  [{b}]")
                lines.append(f"            {used:.0f}/{g['total_gb']:.0f} GB  free: {g['free_gb']:.0f} GB")
        else:
            lines.append("  GPU     :  NO CUDA DEVICES")
        lines.append("")
        return "\n".join(lines)

    # ── Commands ─────────────────────────────────────────────────────────────
    def _commands(self) -> str:
        lines = [""]
        for key, label, _ in self._menu_items():
            lines.append(f"  ({key})  {label}")
        lines += ["  (Q)  QUIT", ""]
        return "\n".join(lines)

    # ── Technical notes ───────────────────────────────────────────────────────
    def _notes(self) -> str:
        lines = [
            "",
            f"  {len(self._phys)} physical  ·  {len(self._logical)} logical",
            "  Config integrity:  OK",
            "  Press 1-5 to navigate",
            "",
        ]
        return "\n".join(lines)

    def _status_line(self) -> str:
        if self._online:
            active = self._srv.get("active_servers", 0)
            servers = self._srv.get("servers", [])
            if servers:
                names = "  ·  ".join(s.get("model", "?") for s in servers[:3])
                return f"  ● ONLINE  ·  {active} active  ·  {names}"
            return f"  ● ONLINE  ·  {active} model server(s) active  ·  {ALLAMA_URL}"
        return f"  ○ OFFLINE  ·  Allama server not reachable  ·  {ALLAMA_URL}"

    def _menu_items(self):
        return [
            ("1", "Config Assistant",   "assistant"),
            ("2", "Model Library",      "models"),
            ("3", "Create Config",      "creator"),
            ("4", "Profiles",           "profiles"),
            ("5", "System Info",        "sysinfo"),
        ]

    def _fkey_bar(self) -> str:
        keys = [
            ("F1", "Config"),
            ("F2", "Library"),
            ("F3", "Create"),
            ("F4", "Profiles"),
            ("F5", "SysInfo"),
            ("F10", "Quit"),
        ]
        return "  " + "   ".join(f"<{k}: {label}>" for k, label in keys)

    # ── Menu actions ─────────────────────────────────────────────────────────
    def action_menu_1(self) -> None:
        self.app.push_screen(AssistantScreen(self._gpus, self._phys, self._logical))

    def action_menu_2(self) -> None:
        self.app.push_screen(ModelLibraryScreen(self._phys, self._logical))

    def action_menu_3(self) -> None:
        self.app.push_screen(ConfigCreatorScreen())

    def action_menu_4(self) -> None:
        self.app.push_screen(ProfilesScreen(self._gpus, self._phys))

    def action_menu_5(self) -> None:
        self.app.push_screen(SysInfoScreen(self._gpus, self._online, self._srv))

    def action_quit(self) -> None:
        self.app.exit()

    def on_key(self, event) -> None:
        key = event.key
        if   key == "1": self.action_menu_1()
        elif key == "2": self.action_menu_2()
        elif key == "3": self.action_menu_3()
        elif key == "4": self.action_menu_4()
        elif key == "5": self.action_menu_5()
        elif key in ("q", "Q"): self.action_quit()


# ──────────────────────────────────────────────────────────────────────────────
# Config Assistant Screen
# ──────────────────────────────────────────────────────────────────────────────
class AssistantScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back", show=False),
        Binding("q",      "back", "Back", show=False),
        Binding("f10",    "back", "Back"),
    ]

    def __init__(self, gpus, phys_models, logical_models, **kwargs):
        super().__init__(**kwargs)
        self._gpus    = gpus
        self._phys    = phys_models
        self._logical = logical_models

    def compose(self) -> ComposeResult:
        with Container(id="main-window"):
            with ScrollableContainer(id="assistant-scroll"):
                yield Static(self._hardware_section(), id="hw-section")
                yield Static(self._models_section(),   id="models-section")
                yield Static(self._profiles_section(), id="profiles-section")
                yield Static(self._tips_section(),     id="tips-section")
            with Horizontal(id="back-bar"):
                yield Button("◄ BACK  [ESC]", id="back-btn", variant="default")
            yield Static("<F10: Back>", id="fkey-bar")

    def on_mount(self) -> None:
        self.query_one("#main-window").border_title = "[ CONFIG ASSISTANT ]"

    def _hardware_section(self) -> str:
        lines = ["", "  HARDWARE ANALYSIS", "  ══════════════════", ""]
        if not self._gpus:
            lines.append("  ⚠  NO CUDA DEVICES DETECTED — running in CPU mode")
        else:
            total_free  = sum(g["free_gb"] for g in self._gpus)
            total_vram  = sum(g["total_gb"] for g in self._gpus)
            lines.append(f"  Detected {len(self._gpus)} GPU(s)  ·  "
                         f"Total VRAM: {total_vram:.0f}GB  ·  Free: {total_free:.0f}GB")
            lines.append("")
            for g in self._gpus:
                used  = g["total_gb"] - g["free_gb"]
                b     = bar(used, g["total_gb"], 20)
                lines.append(f"  GPU {g['index']}  {g['name']}")
                lines.append(f"         [{b}]  {used:.1f} / {g['total_gb']:.1f} GB")
                lines.append(f"         FREE: {g['free_gb']:.1f} GB")
                lines.append("")
        return "\n".join(lines)

    def _models_section(self) -> str:
        lines = ["  LOADED CONFIGURATION FILES", "  ════════════════════════════", ""]
        if not self._phys:
            lines.append("  No physical models configured yet.")
            lines.append("  → Use [3] CREATE CONFIG to add models.")
            return "\n".join(lines)

        lines.append(f"  {'MODEL':<32} {'BACKEND':<12} {'SIZE':>7}  {'VRAM NEED':>9}")
        lines.append(f"  {'─'*32} {'─'*12} {'─'*7}  {'─'*9}")
        for m in self._phys:
            size  = m["size_gb"]
            need  = vram_need(m)
            runnable = "✓" if self._gpus and need <= sum(g["free_gb"] for g in self._gpus) * 0.92 else "✗"
            lines.append(
                f"  {runnable} {m['name']:<30} {m['backend']:<12} "
                f"{size:>5.1f}GB  {need:>7.1f}GB"
            )
        lines.append("")
        lines.append(f"  Total models: {len(self._phys)} physical  ·  "
                     f"{len(self._logical)} logical")
        lines.append("")
        return "\n".join(lines)

    def _profiles_section(self) -> str:
        profiles = build_profiles(self._gpus, self._phys)
        lines = ["  RECOMMENDED PROFILES", "  ════════════════════", ""]
        if not profiles:
            if not self._gpus:
                lines.append("  Configure GPU access to receive recommendations.")
            elif not self._phys:
                lines.append("  Add models via [3] CREATE CONFIG to receive recommendations.")
            else:
                lines.append("  No viable profiles found for current hardware.")
            return "\n".join(lines)

        for p in profiles:
            lines += [
                f"  ┌─ {p['emoji']}  {p['name']} {'─' * max(0, 42 - len(p['name']))}┐",
                f"  │  {p['desc']}",
                f"  │  Context: {p['ctx_hint']}",
                f"  │  Models : " + ", ".join(p["models"]),
            ]
            total_vram = sum(g["total_gb"] for g in self._gpus) if self._gpus else 0
            vram_pct   = p["total_gb"] / total_vram * 100 if total_vram > 0 else 0
            b          = bar(p["total_gb"], total_vram, 20)
            lines += [
                f"  │  VRAM   : [{b}]  {p['total_gb']:.1f}GB  ({vram_pct:.0f}%)",
                f"  │  TP     : {p['tp']}",
                f"  └{'─' * 52}┘",
                "",
            ]
        return "\n".join(lines)

    def _tips_section(self) -> str:
        gpus = self._gpus
        lines = ["  OPTIMIZATION TIPS", "  ══════════════════", ""]
        if gpus:
            total_free = sum(g["free_gb"] for g in gpus)
            max_single = max(g["free_gb"] for g in gpus)
            gpu_count  = len(gpus)

            if gpu_count >= 2:
                lines.append("  ▸ With multiple GPUs, use tensor_parallel=2 for models")
                lines.append("    larger than a single GPU's VRAM.")
                lines.append("  ▸ Models ≤ single GPU VRAM benefit from separate pinning")
                lines.append("    (TP=1 per GPU) for parallel serving.")
            if total_free > 40:
                lines.append("  ▸ You have plenty of VRAM — consider increasing")
                lines.append("    max_model_len for larger context windows.")
            if max_single < 16:
                lines.append("  ▸ Limited single-GPU VRAM — prefer Q4/Q5 quantized")
                lines.append("    GGUF models with llama.cpp backend.")
            lines.append("  ▸ For Qwen3.5 models: --reasoning-parser qwen3 enables")
            lines.append("    the thinking mode; adjust temperature 0.6-0.7.")
            lines.append("  ▸ Use presence_penalty=1.5 to reduce repetition on")
            lines.append("    chat models without degrading quality.")
        else:
            lines.append("  ▸ Install CUDA drivers and nvidia-smi for GPU detection.")
            lines.append("  ▸ llama.cpp supports CPU-only inference without CUDA.")
        lines.append("")
        return "\n".join(lines)

    def action_back(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.action_back()


# ──────────────────────────────────────────────────────────────────────────────
# Model Library Screen
# ──────────────────────────────────────────────────────────────────────────────
class ModelLibraryScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back", show=False),
        Binding("q",      "back", "Back", show=False),
        Binding("f10",    "back", "Back"),
    ]

    def __init__(self, phys, logical, **kwargs):
        super().__init__(**kwargs)
        self._phys    = phys
        self._logical = logical

    def compose(self) -> ComposeResult:
        with Container(id="main-window"):
            with Vertical(id="wizard-container"):
                yield Static("  PHYSICAL MODELS (execution configs)", classes="section-header")
                yield Static("")
                tbl = DataTable(id="phys-table")
                yield tbl
                yield Static("")
                yield Static("  LOGICAL MODELS (sampling presets)", classes="section-header")
                yield Static("")
                tbl2 = DataTable(id="log-table")
                yield tbl2
            with Horizontal(id="back-bar"):
                yield Button("◄ BACK  [ESC]", id="back-btn")
            yield Static("<F10: Back>", id="fkey-bar")

    def on_mount(self) -> None:
        self.query_one("#main-window").border_title = "[ MODEL LIBRARY ]"
        pt = self.query_one("#phys-table", DataTable)
        pt.add_columns("NAME", "BACKEND", "SIZE", "VRAM EST", "TP", "CTX")
        for m in self._phys:
            cfg  = m["cfg"]
            tp   = cfg.get("tensor_parallel", "1")
            ctx  = cfg.get("max_model_len") or cfg.get("n_ctx", "—")
            need = vram_need(m)
            size_s = f"{m['size_gb']:.1f}GB" if m["size_gb"] > 0 else "—"
            need_s = f"{need:.1f}GB"
            if isinstance(ctx, str) and len(ctx) > 6:
                ctx = ctx[:6]
            pt.add_row(m["name"], m["backend"], size_s, need_s, str(tp), str(ctx))

        lt = self.query_one("#log-table", DataTable)
        lt.add_columns("NAME", "PHYSICAL", "TEMP", "TOP_P", "TOP_K")
        for m in self._logical:
            cfg      = m["cfg"]
            sampling = cfg.get("sampling", {})
            temp  = str(sampling.get("temperature", "—"))
            top_p = str(sampling.get("top_p",       "—"))
            top_k = str(sampling.get("top_k",       "—"))
            lt.add_row(m["name"], m["physical"], temp, top_p, top_k)

    def action_back(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.action_back()


# ──────────────────────────────────────────────────────────────────────────────
# Profiles Screen
# ──────────────────────────────────────────────────────────────────────────────
class ProfilesScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back", show=False),
        Binding("q",      "back", "Back", show=False),
        Binding("f10",    "back", "Back"),
    ]

    def __init__(self, gpus, phys, **kwargs):
        super().__init__(**kwargs)
        self._gpus  = gpus
        self._phys  = phys

    def compose(self) -> ComposeResult:
        profiles = build_profiles(self._gpus, self._phys)
        with Container(id="main-window"):
            with ScrollableContainer(id="assistant-scroll"):
                if not profiles:
                    yield Static(
                        "\n  No profiles available.\n"
                        "  Add physical models and ensure GPUs are detected.\n",
                        classes="profile-desc"
                    )
                else:
                    yield Static(
                        "\n  Profiles are generated from your hardware + loaded model configs.\n"
                        "  Each profile represents an optimal load-out for a specific use case.\n",
                        classes="profile-desc"
                    )
                    for p in profiles:
                        yield Static(self._render_profile(p), classes="profile-card")

                yield Static(self._preset_catalog(), id="preset-catalog")

            with Horizontal(id="back-bar"):
                yield Button("◄ BACK  [ESC]", id="back-btn")
            yield Static("<F10: Back>", id="fkey-bar")

    def on_mount(self) -> None:
        self.query_one("#main-window").border_title = "[ CONFIGURATION PROFILES ]"

    def _render_profile(self, p: dict) -> str:
        gpus = self._gpus
        total_vram = sum(g["total_gb"] for g in gpus) if gpus else 48
        b = bar(p["total_gb"], total_vram, 24)
        models_str = "\n".join(f"    · {m}" for m in p["models"])
        tags = "  ".join(f"[{t.upper()}]" for t in p.get("tags", []))
        return (
            f"\n  {p['emoji']}  {p['name']}\n"
            f"  {'═' * 54}\n"
            f"  {p['desc']}\n"
            f"  {p['ctx_hint']}\n\n"
            f"  MODELS:\n{models_str}\n\n"
            f"  VRAM  [{b}]  {p['total_gb']:.1f}GB\n"
            f"  TP    {p['tp']}\n"
            f"  {tags}\n"
        )

    def _preset_catalog(self) -> str:
        lines = [
            "",
            "  SAMPLING PRESET CATALOG",
            "  ════════════════════════",
            "",
            "  These are the recommended sampling parameters by model family:",
            "",
            f"  {'FAMILY':<20} {'TEMP':>5} {'TOP_P':>6} {'TOP_K':>6} {'NOTES'}",
            f"  {'─'*20} {'─'*5} {'─'*6} {'─'*6} {'─'*30}",
            f"  {'Qwen3.5 (thinking)':<20} {'0.7':>5} {'0.8':>6} {'20':>6}  reasoning-parser qwen3",
            f"  {'Qwen3.5 (non-thinking)':<20} {'0.6':>5} {'0.95':>6} {'20':>6}  max_tokens_thinking=0",
            f"  {'Qwen3-VL (vision)':<20} {'0.7':>5} {'0.8':>6} {'20':>6}  + vision encoder args",
            f"  {'DeepSeek R1':<20} {'0.6':>5} {'0.95':>6} {'40':>6}  reasoning-parser deepseek_r1",
            f"  {'Llama 3.x':<20} {'0.7':>5} {'0.9':>6} {'40':>6}  tool-call-parser llama3_json",
            f"  {'Mistral/Mixtral':<20} {'0.7':>5} {'0.9':>6} {'40':>6}  tool-call-parser mistral",
            f"  {'Gemma':<20} {'1.0':>5} {'0.95':>6} {'64':>6}  default Google config",
            "",
            "  Source: Official HuggingFace model cards + vLLM Recipes documentation",
            "",
        ]
        return "\n".join(lines)

    def action_back(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.action_back()


# ──────────────────────────────────────────────────────────────────────────────
# Config Creator Screen
# ──────────────────────────────────────────────────────────────────────────────
class ConfigCreatorScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back", show=False),
        Binding("q",      "back", "Back", show=False),
        Binding("f10",    "back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        with Container(id="main-window"):
            with ScrollableContainer(id="wizard-container"):
                yield Static(
                    "\n  This wizard creates .allm config files for a downloaded model.\n"
                    "  For a fully interactive wizard, run the CLI tool instead:\n\n"
                    "      python3 create_config.py /path/to/model\n\n"
                    "  The CLI wizard auto-detects the model architecture, suggests\n"
                    "  tensor_parallel settings based on your GPUs, and generates\n"
                    "  both physical and logical .allm files ready to use.\n",
                    classes="profile-desc"
                )
                yield Static("  QUICK REFERENCE — PHYSICAL MODEL FIELDS", classes="section-header")
                yield Static(self._phys_reference())
                yield Static("  QUICK REFERENCE — LOGICAL MODEL FIELDS", classes="section-header")
                yield Static(self._logical_reference())
                yield Static("  FILE LOCATIONS", classes="section-header")
                yield Static(self._file_locations())
            with Horizontal(id="back-bar"):
                yield Button("◄ BACK  [ESC]", id="back-btn")
            yield Static("<F10: Back>", id="fkey-bar")

    def on_mount(self) -> None:
        self.query_one("#main-window").border_title = "[ CREATE CONFIG ]"

    def _phys_reference(self) -> str:
        return """
  VLLM BACKEND:
  ─────────────
  backend = "vllm"
  path    = "/path/to/model/directory"
  tokenizer = "/path/to/model/directory"    ← same as path
  tensor_parallel = "1"                     ← 2 or 4 for large models
  gpu_memory_utilization = "0.90"
  max_model_len = "131072"                  ← from model's config.json
  max_num_seqs = "8"                        ← concurrent requests
  extra_args = ["--reasoning-parser", "qwen3", ...]

  LLAMA.CPP BACKEND:
  ──────────────────
  backend = "llama.cpp"
  model   = "/path/to/model.gguf"
  mmproj  = "/path/to/mmproj.gguf"         ← only for vision models
  n_ctx   = "40960"                         ← context window
  n_batch = "1024"
  n_gpu_layers = "-1"                       ← -1 = all layers to GPU
  n_threads = "16"
  extra_args = ["--chat-template", "chatml", "--jinja"]
"""

    def _logical_reference(self) -> str:
        return """
  physical = "physical-model-name"          ← must match physical .allm filename

  [sampling]
  temperature = 0.7
  top_p = 0.8
  top_k = 20
  min_p = 0.0
  presence_penalty = 1.5                    ← optional
  repetition_penalty = 1.0                  ← optional
"""

    def _file_locations(self) -> str:
        phys_dir = CONFIG_DIR / "physical"
        log_dir  = CONFIG_DIR / "logical"
        return f"""
  Physical configs:  {phys_dir}
  Logical configs:   {log_dir}

  Filename = model name in Allama
  Example:  Qwen3.5-9b.allm  →  physical model "Qwen3.5-9b"
            Qwen3.5:9b.allm  →  logical model "Qwen3.5:9b"
"""

    def action_back(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.action_back()


# ──────────────────────────────────────────────────────────────────────────────
# System Info Screen
# ──────────────────────────────────────────────────────────────────────────────
class SysInfoScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back", show=False),
        Binding("q",      "back", "Back", show=False),
        Binding("f10",    "back", "Back"),
    ]

    def __init__(self, gpus, online, srv, **kwargs):
        super().__init__(**kwargs)
        self._gpus   = gpus
        self._online = online
        self._srv    = srv

    def compose(self) -> ComposeResult:
        with Container(id="main-window"):
            with ScrollableContainer(id="assistant-scroll"):
                yield Static(self._render_all(), id="sysinfo-content")
            with Horizontal(id="back-bar"):
                yield Button("◄ BACK  [ESC]", id="back-btn")
                yield Button("⟳ REFRESH", id="refresh-btn")
            yield Static("<F10: Back>", id="fkey-bar")

    def on_mount(self) -> None:
        self.query_one("#main-window").border_title = "[ SYSTEM INFORMATION ]"

    def _render_all(self) -> str:
        lines = [""]

        lines += ["  GRAPHICS PROCESSING UNITS", "  ═════════════════════════", ""]
        if self._gpus:
            for g in self._gpus:
                used = g["total_gb"] - g["free_gb"]
                b    = bar(used, g["total_gb"], 30)
                lines += [
                    f"  GPU {g['index']} ─ {g['name']}",
                    f"    VRAM   [{b}]",
                    f"    TOTAL  {g['total_gb']:.1f} GB",
                    f"    USED   {used:.1f} GB",
                    f"    FREE   {g['free_gb']:.1f} GB",
                    "",
                ]
        else:
            lines += ["  No GPUs detected.", ""]

        lines += ["  ALLAMA SERVER", "  ════════════", ""]
        if self._online:
            lines += [
                f"  Endpoint  : {ALLAMA_URL}",
                f"  Status    : ONLINE",
                f"  Active    : {self._srv.get('active_servers', 0)} model server(s)",
                "",
            ]
        else:
            lines += [f"  Endpoint  : {ALLAMA_URL}", "  Status    : OFFLINE", ""]

        lines += ["  ENVIRONMENT", "  ═══════════", ""]
        lines.append(f"  Python    : {sys.version.split()[0]}")
        lines.append(f"  Config Dir: {CONFIG_DIR}")
        vllm_path = os.environ.get("PATH", "").split(":")
        vllm_found = any(Path(p, "vllm").exists() for p in vllm_path)
        lines.append(f"  vLLM      : {'found in PATH' if vllm_found else 'not in PATH'}")
        llama_candidates = [
            "/home/nick/AI/vllm/allama/llama-server",
            "/usr/local/bin/llama-server",
        ]
        llama_found = next((p for p in llama_candidates if Path(p).exists()), None)
        lines.append(f"  llama-srv : {llama_found or 'not found'}")
        lines.append("")

        lines += ["  CONFIGURATION SUMMARY", "  ════════════════════", ""]
        phys_dir = CONFIG_DIR / "physical"
        log_dir  = CONFIG_DIR / "logical"
        if phys_dir.exists():
            phys_files = list(phys_dir.glob("*.allm"))
            lines.append(f"  Physical  : {len(phys_files)} file(s) in {phys_dir}")
        if log_dir.exists():
            log_files = list(log_dir.glob("*.allm"))
            lines.append(f"  Logical   : {len(log_files)} file(s) in {log_dir}")
        lines.append("")

        return "\n".join(lines)

    def action_back(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.action_back()
        elif event.button.id == "refresh-btn":
            gpus    = get_gpus()
            online, srv = check_allama_server()
            self._gpus   = gpus
            self._online = online
            self._srv    = srv
            self.query_one("#sysinfo-content").update(self._render_all())


# ──────────────────────────────────────────────────────────────────────────────
# Application
# ──────────────────────────────────────────────────────────────────────────────
class AllamaTUI(App):
    TITLE        = "ALLAMA · FILE & CONFIG UTILITY"
    SUB_TITLE    = "v1.0"
    CSS          = CSS
    ENABLE_COMMAND_PALETTE = False

    def on_mount(self) -> None:
        self.push_screen(BootScreen())


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = AllamaTUI()
    app.run()
