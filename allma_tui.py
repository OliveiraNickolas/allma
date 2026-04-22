#!/usr/bin/env python3
"""
ALLMA · FILE & CONFIG UTILITY
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from wizard import WizardStep1Screen
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Button, DataTable, Static

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "configs"
ALLMA_URL = "http://127.0.0.1:9000"

# ──────────────────────────────────────────────────────────────────────────────
# Logo
# ──────────────────────────────────────────────────────────────────────────────
_LOGO_COLORS = ["#e52529", "#f7941d", "#f7d000", "#43b047", "#009ddc"]

def _build_logo_rows() -> list[str]:
    CHARS = {
        'A': [" ███ ", "█   █", "█████", "█   █", "█   █"],
        'L': ["█    ", "█    ", "█    ", "█    ", "█████"],
        'M': ["█   █", "██ ██", "█ █ █", "█   █", "█   █"],
    }
    word, starts = list("ALLMA"), [0, 6, 12, 18, 24]
    canvas = [[0] * 30 for _ in range(5)]
    for ch, col in zip(word, starts):
        for r, row in enumerate(CHARS[ch]):
            for c, px in enumerate(row):
                if px == "█":
                    canvas[r][col + c] = 1
        if ch == "L":
            for r, row in enumerate(CHARS[ch]):
                for c, px in enumerate(row):
                    if px == "█" and col + c + 1 < 30 and canvas[r][col + c + 1] == 0:
                        canvas[r][col + c + 1] = 2
    return ["".join("█" if v == 1 else "▒" if v == 2 else " " for v in row)
            for row in canvas]

def _build_logo_markup():
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
# Editable column specs for profile models
# (col_idx, col_key, header, param_key, min, max, big_step, small_step, is_int, default)
# ──────────────────────────────────────────────────────────────────────────────
_LOG_EDITABLE = {
    2: ("temperature",         0.0,  2.0,   0.05, 0.01, False, 0.7,  "Temperature"),
    3: ("top_p",               0.0,  1.0,   0.05, 0.01, False, 0.95, "Top-P"),
    4: ("top_k",               0,    200,   5,    1,    True,  20,   "Top-K"),
    5: ("min_p",               0.0,  1.0,   0.05, 0.01, False, 0.0,  "Min-P"),
    6: ("presence_penalty",   -2.0,  2.0,   0.1,  0.05, False, 0.0,  "Presence Penalty"),
    7: ("repetition_penalty",  0.8,  1.5,   0.05, 0.01, False, 1.0,  "Repetition Penalty"),
}

# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────
CSS = """
Screen {
    background: #0a0a08;
    align: center middle;
}

/* ─── floating window shell ─────────────────────────────────── */
.shadow-wrap {
    background: #3a2c1a;
    width: 133;
    height: 100%;
    align: center middle;
}
#main-window {
    border: double #008888;
    border-title-color: #f0e8d0;
    border-title-style: bold;
    border-title-align: center;
    background: #d0c4a8;
    width: 100%;
    height: 100%;
    margin: 0 3 0 0;
}

/* ─── boot ──────────────────────────────────────────────────── */
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

/* ─── main menu ─────────────────────────────────────────────── */
#menu-logo {
    content-align: center middle;
    width: 100%;
    padding: 1 0 1 0;
}
#menu-dashboard {
    color: #007878;
    padding: 0 4;
}
#menu-items {
    color: #1a1408;
    padding: 0 4;
}
#menu-fkeys {
    height: 1;
    background: #c8b898;
    color: #007878;
    text-style: bold;
    padding: 0 2;
    content-align: left middle;
}

/* ─── sub-screens ───────────────────────────────────────────── */
#sub-content {
    height: 1fr;
    background: #e8dfc8;
    color: #1a1408;
    padding: 1 2;
    scrollbar-color: #007878;
    scrollbar-background: #c8b898;
    scrollbar-size: 1 1;
}
#sub-bottom {
    height: 3;
    background: #c8b898;
    align: left middle;
    padding: 0 2;
}

/* ─── buttons ───────────────────────────────────────────────── */
Button {
    background: #007878;
    color: #f0e8d0;
    border: tall #009898 #004848;
    text-style: bold;
    min-width: 16;
}
Button:hover {
    background: #7a1818;
    color: #f0e8d0;
    border: tall #9a2828 #4a0808;
}

/* ─── data table ─────────────────────────────────────────────── */
DataTable {
    background: #e8dfc8;
    color: #1a1408;
    border: solid #008888;
    height: auto;
    max-height: 20;
}
DataTable > .datatable--header {
    background: #c8b898;
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

/* ─── inline editor panel ──────────────────────────────────── */
#editor-panel {
    display: none;
    height: auto;
    background: #d0c4a8;
    border-top: solid #008888;
    padding: 0 2;
}
#editor-panel.editing {
    display: block;
}
#ed-title {
    color: #007878;
    text-style: bold;
    padding: 1 0 0 0;
}
#ed-bar {
    color: #007878;
    text-style: bold;
}
#ed-bar:focus {
    background: #c8b898;
}
#ed-value {
    color: #1a1408;
    text-style: bold;
}
#ed-vram {
    color: #007878;
}
#ed-hint {
    color: #6a5a48;
    padding: 0 0 1 0;
}

/* ─── wizard passthrough ─────────────────────────────────────── */
#wizard-layout { height: 1fr; }
#step-nav {
    width: 24;
    background: #c8b898;
    border-right: solid #008888;
    padding: 0 1;
    height: 100%;
}
#step-content { width: 1fr; height: 100%; }
.wizard-step      { color: #6a5a48; padding: 1 0 0 0; }
.wizard-step-bar  {
    color: #007878; text-style: bold;
    text-align: center; content-align: center middle;
    height: 1; padding: 1 0 0 0;
}
.wizard-step-title { color: #1a1408; text-style: bold; }
.wizard-section {
    color: #007878; text-style: bold;
    padding: 1 0 0 0; border-bottom: solid #008888;
}
.wizard-hint    { color: #6a5a48; padding: 0 0 1 0; }
.wizard-preview {
    color: #1a1408; background: #f0e8d0;
    padding: 1; border: solid #008888; margin: 1 0;
}
.wizard-field         { padding: 0 0 1 0; }
.wizard-field Label   { color: #1a1408; }
.wizard-field Input   { background: #f0e8d0; color: #007878; border: solid #008888; }
.wizard-field Input:focus { border: solid #007878; }
.wizard-sampling-row  { height: auto; }
.sampling-col         { width: 1fr; padding: 0 1 0 0; }
.wizard-row-2col      { height: auto; margin: 0 0 1 0; }
.col-half             { width: 1fr; padding: 0 1 0 0; }
.sampling-row-mini    { height: auto; }
.sampling-mini        { width: 1fr; padding: 0 1 0 0; }
#wizard-nav     { height: 3; align: left middle; padding: 0 1; background: #e8dfc8; }
#wizard-container { height: 1fr; background: #e8dfc8; padding: 0 2; }
#tip-box {
    height: auto; border: solid #008888;
    border-title-color: #007878; border-title-style: bold;
    background: #d0c4a8; padding: 0 1;
}
#tip-text { color: #6a5a48; }
Select          { background: #e8dfc8; border: solid #008888; color: #1a1408; }
Select:focus    { border: solid #007878; }
SelectOverlay   { background: #f0e8d0; border: solid #008888; }
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


def check_allma_server() -> tuple[bool, dict]:
    try:
        import urllib.request
        req = urllib.request.Request(f"{ALLMA_URL}/health", method="GET")
        req.add_header("User-Agent", "AllmaTUI/1.0")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return True, data
    except Exception:
        return False, {}


def scan_base_models(config_dir: Path) -> list[dict]:
    models = []
    phys_dir = config_dir / "base"
    if not phys_dir.exists():
        return models
    for f in sorted(phys_dir.glob("*.allm")):
        try:
            from configs.loader import parse_all_file
            cfg = parse_all_file(f.read_text())
            if "backend" not in cfg:
                continue
            backend = cfg.get("backend", "vllm")
            path    = cfg.get("path") or cfg.get("model", "")
            size_gb = _estimate_size(path, backend)
            models.append({
                "name":    cfg.get("name") or f.stem,
                "backend": backend,
                "path":    path,
                "size_gb": size_gb,
                "cfg":     cfg,
                "file":    f,
            })
        except Exception:
            pass
    return models


def scan_profile_models(config_dir: Path) -> list[dict]:
    models = []
    log_dir = config_dir / "profile"
    if not log_dir.exists():
        return models
    for f in sorted(log_dir.glob("*.allm")):
        try:
            from configs.loader import parse_all_file
            cfg = parse_all_file(f.read_text())
            if "base" not in cfg:
                continue
            models.append({
                "name":     cfg.get("name") or f.stem,
                "base": cfg["base"],
                "cfg":      cfg,
                "file":     f,
            })
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


def vram_need(model: dict, ctx: int | None = None) -> float:
    """Estimate VRAM: weights + KV cache for given context length."""
    base = model["size_gb"] * 1.15 if model["size_gb"] > 0 else 4.0
    if ctx and model["size_gb"] > 0:
        kv = ctx * model["size_gb"] / 500_000
        return base + kv
    return base


def bar(value: float, total: float, width: int = 20) -> str:
    if total == 0:
        return "░" * width
    filled = round(value / total * width)
    return "█" * filled + "░" * (width - filled)


def update_allm_param(path: Path, section: str | None, key: str, value) -> None:
    """Update or add key=value in an .allm file, respecting sections."""
    lines = path.read_text(encoding="utf-8").splitlines()
    current_section: str | None = None
    updated = False
    result = []

    for line in lines:
        s = line.strip()
        if s.startswith("[") and s.endswith("]") and "=" not in s:
            current_section = s[1:-1].strip()
        in_scope = (current_section == section)
        if in_scope and "=" in s and not s.startswith("#"):
            k = s.split("=", 1)[0].strip()
            if k == key:
                result.append(f"{key} = {value}")
                updated = True
                continue
        result.append(line)

    if not updated:
        if section:
            # Insert after the section header, before any next section
            new_result, in_sec, added = [], False, False
            for line in result:
                s = line.strip()
                if s == f"[{section}]":
                    in_sec = True
                    new_result.append(line)
                    continue
                if in_sec and s.startswith("[") and s.endswith("]") and "=" not in s:
                    new_result.append(f"{key} = {value}")
                    added = True
                    in_sec = False
                new_result.append(line)
            if in_sec:  # section was at end of file
                new_result.append(f"{key} = {value}")
                added = True
            if not added:  # section didn't exist
                new_result += ["", f"[{section}]", f"{key} = {value}"]
            result = new_result
        else:
            result.append(f"{key} = {value}")

    path.write_text("\n".join(result) + "\n", encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Boot Screen
# ──────────────────────────────────────────────────────────────────────────────
class BootScreen(Screen):
    BINDINGS = [Binding("enter", "proceed", "Continue"),
                Binding("space", "proceed", "Continue")]

    _done:          reactive[bool] = reactive(False)
    _step:          int  = 0
    _blink_on:      bool = True
    _display_lines: list = []

    def compose(self) -> ComposeResult:
        with Container(classes="shadow-wrap"):
            with Container(id="boot-container"):
                yield Static(_build_logo_markup(), id="boot-logo")
                yield Static("", id="boot-text")
                yield Static("", id="boot-bar")

    def on_mount(self) -> None:
        self._gpus    = get_gpus()
        self._online, self._srv = check_allma_server()
        self._base    = scan_base_models(CONFIG_DIR)
        self._profiles = scan_profile_models(CONFIG_DIR)
        self._step, self._blink_on, self._display_lines, self._done = 0, True, [], False
        self.set_interval(0.08, self._tick)
        self.set_interval(0.5,  self._blink)

    def _blink(self) -> None:
        self._blink_on = not self._blink_on
        if self._done:
            lines = list(self._display_lines)
            cursor = "▌" if self._blink_on else " "
            lines[-1] = f"{cursor} PRESS ANY KEY TO CONTINUE"
            self.query_one("#boot-text").update("\n".join(lines))

    def _tick(self) -> None:
        if self._done:
            return
        messages = self._build_messages()
        if self._step < len(messages):
            self._display_lines.append(messages[self._step])
            self.query_one("#boot-text").update("\n".join(self._display_lines))
            pct    = int(self._step / max(len(messages) - 1, 1) * 100)
            filled = round(pct / 100 * 48)
            self.query_one("#boot-bar").update(
                f"[{'█' * filled}{'░' * (48 - filled)}] {pct:3d}%"
            )
            self._step += 1
        else:
            self._done = True
            self._display_lines.append("▌ PRESS ANY KEY TO CONTINUE")
            self.query_one("#boot-text").update("\n".join(self._display_lines))
            self.query_one("#boot-bar").update("")

    def _build_messages(self) -> list[str]:
        msgs = ["ALLMA · FILE & CONFIG UTILITY  v1.0", "(c) 2025", "",
                "─" * 50, "INITIALIZING HARDWARE SUBSYSTEMS..."]
        if self._gpus:
            for g in self._gpus:
                msgs.append(f"  GPU {g['index']}: {g['name'][:24]}  [{g['total_gb']:.0f}GB]  ... DETECTED")
        else:
            msgs.append("  NO CUDA DEVICES FOUND  [CPU MODE]")
        msgs += ["SCANNING CONFIGURATION FILES...",
                 f"  BASE MODELS : {len(self._base):3d}  LOADED",
                 f"  PROFILE MODELS  : {len(self._profiles):3d}  LOADED",
                 "CONNECTING TO ALLMA SERVER..."]
        if self._online:
            msgs.append(f"  {ALLMA_URL}  ...  ONLINE  [{self._srv.get('active_servers', 0)} active]")
        else:
            msgs.append(f"  {ALLMA_URL}  ...  OFFLINE")
        msgs += ["─" * 50, "SYSTEM READY.", ""]
        return msgs

    def action_proceed(self) -> None:
        if self._done:
            self.app.push_screen(
                MainMenuScreen(self._gpus, self._online, self._srv, self._base, self._profiles)
            )

    def on_key(self, event) -> None:
        if self._done:
            self.action_proceed()


# ──────────────────────────────────────────────────────────────────────────────
# Main Menu Screen
# ──────────────────────────────────────────────────────────────────────────────
class MainMenuScreen(Screen):
    BINDINGS = [
        Binding("1",   "go_1", show=False),
        Binding("2",   "go_2", show=False),
        Binding("q",   "quit", show=False),
        Binding("f1",  "go_1", "Add Model"),
        Binding("f2",  "go_2", "Library"),
        Binding("f10", "quit", "Quit"),
    ]

    def __init__(self, gpus, online, srv, base, profiles, **kwargs):
        super().__init__(**kwargs)
        self._gpus, self._online, self._srv = gpus, online, srv
        self._base, self._profiles = base, profiles

    def _dashboard(self) -> str:
        lines = []
        if self._online:
            active  = self._srv.get("active_servers", 0)
            servers = self._srv.get("servers", [])
            lines.append(f"  SERVER  ●  ONLINE  ·  {active} active  ·  {ALLMA_URL}")
            if servers:
                lines.append("          ↳  " + "  ·  ".join(s.get("model", "?") for s in servers[:3]))
        else:
            lines.append(f"  SERVER  ○  OFFLINE  ·  {ALLMA_URL}")
        lines.append("")
        if self._gpus:
            for g in self._gpus:
                used = g["total_gb"] - g["free_gb"]
                lines.append(
                    f"  GPU {g['index']}   [{bar(used, g['total_gb'], 20)}]  "
                    f"{used:.0f}/{g['total_gb']:.0f}GB  ·  free: {g['free_gb']:.0f}GB  ·  {g['name']}"
                )
        else:
            lines.append("  GPU     NO CUDA DEVICES  [CPU MODE]")
        lines.append("")
        lines.append(f"  CONFIGS  {len(self._base)} base  ·  {len(self._profiles)} profile")
        return "\n".join(lines)

    def _menu_text(self) -> str:
        sep = "─" * 62
        def row(key, name, hint):
            return f"  {'  [' + key + ']  ' + name:<26}  {hint}"
        return "\n".join([
            "", f"  {sep}",
            row("F1", "Add Model",     "5-step guided config wizard"),
            row("F2", "Model Library", f"{len(self._base)} base  ·  {len(self._profiles)} profile"),
            f"  {sep}",
            row("Q",  "Quit", ""),
            f"  {sep}", "",
        ])

    def compose(self) -> ComposeResult:
        with Container(classes="shadow-wrap"):
            with Container(id="main-window"):
                yield Static(_build_logo_markup(), id="menu-logo")
                yield Static(self._dashboard(), id="menu-dashboard")
                yield Static(self._menu_text(), id="menu-items")
                yield Static("  F1: Add Model   F2: Model Library   F10 / Q: Quit", id="menu-fkeys")

    def on_mount(self) -> None:
        self.query_one("#main-window").border_title = " ALLMA · FILE & CONFIG UTILITY  v1.0 "

    def on_key(self, event) -> None:
        k = event.key
        if   k == "1": self.action_go_1()
        elif k == "2": self.action_go_2()
        elif k in ("q", "Q"): self.action_quit()

    def action_go_1(self): self.app.push_screen(WizardStep1Screen())
    def action_go_2(self): self.app.push_screen(ModelLibraryScreen(self._base, self._profiles))
    def action_quit(self):  self.app.exit()


# ──────────────────────────────────────────────────────────────────────────────
# Sub-screen base
# ──────────────────────────────────────────────────────────────────────────────
class _SubScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", show=False),
        Binding("q",      "back", show=False),
        Binding("f10",    "back", "Back"),
    ]
    _screen_title: str = "SCREEN"

    def compose(self) -> ComposeResult:
        with Container(classes="shadow-wrap"):
            with Container(id="main-window"):
                yield ScrollableContainer(
                    Static(self._build_content(), id="sub-text"),
                    id="sub-content",
                )
                with Horizontal(id="sub-bottom"):
                    yield Button("◄ Back  [Esc]", id="back-btn")

    def on_mount(self) -> None:
        self.query_one("#main-window").border_title = f" {self._screen_title} "

    def _build_content(self) -> str:
        return ""

    def action_back(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.action_back()


# ──────────────────────────────────────────────────────────────────────────────
# Focusable widget for inline editor
# ──────────────────────────────────────────────────────────────────────────────
class _EditorBar(Static, can_focus=True):
    """Focusable static that captures keys for the inline slider."""
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Model Library Screen
# ──────────────────────────────────────────────────────────────────────────────
class ModelLibraryScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", show=False),
        Binding("q",      "back", show=False),
        Binding("f10",    "back", "Back"),
    ]

    def __init__(self, base, profiles, **kwargs):
        super().__init__(**kwargs)
        self._base    = base
        self._profiles = profiles
        # inline editor state
        self._editing       = False
        self._ed_value      = 0.0
        self._ed_min        = 0.0
        self._ed_max        = 1.0
        self._ed_big        = 0.1
        self._ed_small      = 0.01
        self._ed_is_int     = False
        self._ed_callback   = None   # fn(new_value)
        self._ed_vram_fn    = None   # fn(value) -> str

    def compose(self) -> ComposeResult:
        with Container(classes="shadow-wrap"):
            with Container(id="main-window"):
                yield ScrollableContainer(
                    Static(f"\n  Base Models  ({len(self._base)})\n  {'─'*50}\n"),
                    DataTable(id="base-tbl"),
                    Static(
                        f"\n  Profile Models  ({len(self._profiles)})\n"
                        f"  {'─'*50}\n"
                        "  ← → arrows to move  ·  Enter to edit highlighted value\n"
                    ),
                    DataTable(id="log-tbl"),
                    Static(""),
                    id="sub-content",
                )
                with Container(id="editor-panel"):
                    yield Static("", id="ed-title")
                    yield _EditorBar("", id="ed-bar")
                    yield Static("", id="ed-value")
                    yield Static("", id="ed-vram")
                    yield Static("", id="ed-hint")
                with Horizontal(id="sub-bottom"):
                    yield Button("◄ Back  [Esc]", id="back-btn")

    def on_mount(self) -> None:
        self.query_one("#main-window").border_title = " MODEL LIBRARY "
        self._build_phys_table()
        self._build_log_table()

    # ── table builders ───────────────────────────────────────────────────────
    def _build_phys_table(self) -> None:
        pt = self.query_one("#base-tbl", DataTable)
        pt.cursor_type = "row"
        pt.add_column("NAME",    key="name")
        pt.add_column("BACKEND", key="backend")
        pt.add_column("SIZE",    key="size")
        pt.add_column("VRAM",    key="vram")
        pt.add_column("TP",      key="tp")
        pt.add_column("CTX",     key="ctx")
        for m in self._base:
            cfg  = m["cfg"]
            tp   = cfg.get("tensor_parallel", "1")
            ctx  = cfg.get("max_model_len") or cfg.get("n_ctx", "—")
            need = vram_need(m, int(ctx) if str(ctx).isdigit() else None)
            pt.add_row(
                m["name"], m["backend"],
                f"{m['size_gb']:.1f}GB" if m["size_gb"] > 0 else "—",
                f"{need:.1f}GB", str(tp), str(ctx),
                key=m["name"],
            )
        if not self._base:
            pt.add_row("(none)", "", "", "", "", "", key="_none")

    def _build_log_table(self) -> None:
        lt = self.query_one("#log-tbl", DataTable)
        lt.cursor_type = "cell"
        lt.add_column("NAME",    key="name")
        lt.add_column("PHYSICAL",key="base")
        lt.add_column("TEMP",    key="temperature")
        lt.add_column("TOP_P",   key="top_p")
        lt.add_column("TOP_K",   key="top_k")
        lt.add_column("MIN_P",   key="min_p")
        lt.add_column("PRES_P",  key="presence_penalty")
        lt.add_column("REP_P",   key="repetition_penalty")
        for m in self._profiles:
            s = m["cfg"].get("sampling", {})
            lt.add_row(
                m["name"], m["base"],
                str(s.get("temperature",         "—")),
                str(s.get("top_p",               "—")),
                str(s.get("top_k",               "—")),
                str(s.get("min_p",               "—")),
                str(s.get("presence_penalty",    "—")),
                str(s.get("repetition_penalty",  "—")),
                key=m["name"],
            )
        if not self._profiles:
            lt.add_row("(none)", "", "", "", "", "", "", "", key="_none")

    # ── inline editor ────────────────────────────────────────────────────────
    def _open_editor(self, title, model_name, current,
                     min_v, max_v, big_step, small_step, is_int, callback, vram_fn=None):
        self._editing     = True
        self._ed_min      = float(min_v)
        self._ed_max      = float(max_v)
        self._ed_big      = float(big_step)
        self._ed_small    = float(small_step)
        self._ed_is_int   = is_int
        self._ed_callback = callback
        self._ed_vram_fn  = vram_fn
        try:
            self._ed_value = float(current) if str(current) not in ("—", "", "None") else float((min_v + max_v) / 2)
        except (ValueError, TypeError):
            self._ed_value = float((min_v + max_v) / 2)
        self._ed_value = max(self._ed_min, min(self._ed_max, self._ed_value))

        self.query_one("#ed-title").update(f"  {title}  ·  {model_name}")
        hint_big   = str(int(big_step)) if is_int else str(big_step)
        hint_small = str(int(small_step)) if is_int else str(small_step)
        self.query_one("#ed-hint").update(
            f"  ←  →  ±{hint_big}    Shift+←→  ±{hint_small}"
            f"    Range: {min_v}–{max_v}    Enter: save    Esc: cancel"
        )
        self._refresh_editor()
        self.query_one("#editor-panel").add_class("editing")
        self.query_one("#ed-bar").focus()

    def _refresh_editor(self):
        frac   = (self._ed_value - self._ed_min) / max(self._ed_max - self._ed_min, 1e-9)
        w      = 44
        filled = round(frac * w)
        self.query_one("#ed-bar").update(
            f"  [{'█' * filled}{'░' * (w - filled)}]"
        )
        disp = str(int(self._ed_value)) if self._ed_is_int else f"{self._ed_value:.2f}"
        self.query_one("#ed-value").update(f"  {disp}")
        if self._ed_vram_fn:
            self.query_one("#ed-vram").update(f"  {self._ed_vram_fn(self._ed_value)}")
        else:
            self.query_one("#ed-vram").update("")

    def _close_editor(self):
        self._editing     = False
        self._ed_callback = None
        self._ed_vram_fn  = None
        self.query_one("#editor-panel").remove_class("editing")

    # ── base table: Enter on row → edit CTX ──────────────────────────────
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "base-tbl":
            return
        row_idx = event.cursor_row
        if row_idx >= len(self._base):
            return
        m   = self._base[row_idx]
        cfg = m["cfg"]
        ctx_key = "max_model_len" if "max_model_len" in cfg else "n_ctx"
        cur_ctx = cfg.get(ctx_key, 4096)

        def vram_label(ctx_val):
            need = vram_need(m, int(ctx_val))
            return (f"VRAM estimate: {need:.1f}GB  "
                    f"(weights {m['size_gb'] * 1.15:.1f}GB + KV {need - m['size_gb'] * 1.15:.1f}GB)")

        def on_save(new_ctx):
            update_allm_param(m["file"], None, ctx_key, int(new_ctx))
            cfg[ctx_key] = int(new_ctx)
            new_vram = f"{vram_need(m, int(new_ctx)):.1f}GB"
            tbl = self.query_one("#base-tbl", DataTable)
            tbl.update_cell(row_key=m["name"], column_key="ctx",  value=str(int(new_ctx)))
            tbl.update_cell(row_key=m["name"], column_key="vram", value=new_vram)

        self._open_editor(
            "Context Length", m["name"], cur_ctx,
            1024, 262144, 8192, 1024, True, on_save, vram_label,
        )

    # ── profile table: Enter on editable cell → edit sampling param ──────────
    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        if event.data_table.id != "log-tbl":
            return
        col_idx = event.coordinate.column
        row_idx = event.coordinate.row
        if col_idx not in _LOG_EDITABLE or row_idx >= len(self._profiles):
            return

        param_key, min_v, max_v, big_s, small_s, is_int, default, display = _LOG_EDITABLE[col_idx]
        m        = self._profiles[row_idx]
        sampling = m["cfg"].setdefault("sampling", {})
        cur_val  = sampling.get(param_key, default)

        def on_save(new_val):
            update_allm_param(m["file"], "sampling", param_key, new_val)
            sampling[param_key] = new_val
            self.query_one("#log-tbl", DataTable).update_cell(
                row_key=m["name"], column_key=param_key, value=str(new_val),
            )

        self._open_editor(
            display, m["name"], cur_val,
            min_v, max_v, big_s, small_s, is_int, on_save,
        )

    # ── key handling: editor takes priority ──────────────────────────────────
    def on_key(self, event) -> None:
        if not self._editing:
            return
        step = self._ed_small if "shift" in event.key else self._ed_big
        if event.key in ("left", "shift+left"):
            self._ed_value = max(self._ed_min, self._ed_value - step)
            self._refresh_editor()
            event.stop()
            event.prevent_default()
        elif event.key in ("right", "shift+right"):
            self._ed_value = min(self._ed_max, self._ed_value + step)
            self._refresh_editor()
            event.stop()
            event.prevent_default()
        elif event.key == "enter":
            v = int(self._ed_value) if self._ed_is_int else round(self._ed_value, 4)
            cb = self._ed_callback
            self._close_editor()
            if cb:
                cb(v)
            event.stop()
            event.prevent_default()
        elif event.key == "escape":
            self._close_editor()
            event.stop()
            event.prevent_default()

    def action_back(self) -> None:
        if self._editing:
            self._close_editor()
        else:
            self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.action_back()


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────
class AllmaTUI(App):
    TITLE                  = "ALLMA"
    CSS                    = CSS
    ENABLE_COMMAND_PALETTE = False

    def on_mount(self) -> None:
        self.push_screen(BootScreen())


if __name__ == "__main__":
    AllmaTUI().run()
