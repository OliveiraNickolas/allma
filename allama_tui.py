#!/usr/bin/env python3
"""
╔═╗╦  ╦  ╔═╗╔╦╗╔═╗  ╔╦╗╔═╗╦═╗╔╦╗╦╔╗╔╔═╗╦
╠═╣║  ║  ╠═╣║║║╠═╣   ║ ║╣ ╠╦╝║║║║║║║╠═╣║
╩ ╩╩═╝╩═╝╩ ╩╩ ╩╩ ╩   ╩ ╚═╝╩╚═╩ ╩╩╝╚╝╩ ╩╩═╝

ALLAMA TERMINAL INTERFACE  ·  PHOSPHOR EDITION
Retro TUI for Allama LLM Proxy
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
# ASCII art
# ──────────────────────────────────────────────────────────────────────────────
LOGO_LARGE = """\
 ▄▄   ▄      ▄      ▄▄   ▄  ▄   ▄▄▄  ▄
█  █  █      █     █  █  ██ █  █   █ █
█▀▀█  █      █     █▀▀█  █▀██  █▄▄▄█ █
█  █  █▄▄▄▄  █▄▄▄▄ █  █  █  █  █   █ █▄▄▄▄"""

LOGO_SMALL = "╔═╗╦  ╦  ╔═╗╔╦╗╔═╗\n╠═╣║  ║  ╠═╣║║║╠═╣\n╩ ╩╩═╝╩═╝╩ ╩╩ ╩╩ ╩"

BOOT_SEQUENCE = [
    ("dim",    "ALLAMA TERMINAL INTERFACE v1.0"),
    ("dim",    "(c) 2024  ·  PHOSPHOR GREEN EDITION"),
    ("dim",    ""),
    ("dim",    "─" * 50),
    ("normal", "INITIALIZING HARDWARE SUBSYSTEMS..."),
    ("gpu",    None),          # replaced with real GPU info
    ("normal", "SCANNING CONFIGURATION FILES..."),
    ("cfg",    None),          # replaced with config counts
    ("normal", "CONNECTING TO ALLAMA SERVER..."),
    ("server", None),          # replaced with server status
    ("dim",    "─" * 50),
    ("bright", "SYSTEM READY."),
    ("dim",    ""),
    ("blink",  "▌ PRESS ANY KEY TO CONTINUE"),
]

# ──────────────────────────────────────────────────────────────────────────────
# CSS – Phosphor green theme
# ──────────────────────────────────────────────────────────────────────────────
CSS = """
/* ─── BASE ─────────────────────────────────────────────────────── */
Screen {
    background: #000000;
    color: #00bb00;
}

/* ─── BOOT ──────────────────────────────────────────────────────── */
#boot-container {
    align: center middle;
    width: 100%;
    height: 100%;
}

#boot-logo {
    color: #00ff44;
    text-style: bold;
    content-align: center middle;
    width: 100%;
    padding: 1 0;
}

#boot-text {
    width: 60;
    color: #009900;
    padding: 0 2;
}

#boot-bar {
    width: 60;
    padding: 0 2 1 2;
    color: #006600;
}

/* ─── MAIN MENU ─────────────────────────────────────────────────── */
#main-layout {
    height: 100%;
}

#left-panel {
    width: 1fr;
    border: double #004400;
    padding: 0 1;
}

#right-panel {
    width: 28;
    border: double #004400;
    padding: 0 1;
}

.panel-title {
    color: #00ff44;
    text-style: bold;
    padding: 0 0 1 0;
    border-bottom: solid #003300;
}

.separator {
    color: #003300;
    height: 1;
}

/* ─── MENU ITEMS ────────────────────────────────────────────────── */
.menu-item {
    color: #00bb00;
    padding: 0 1;
    height: 2;
}

.menu-item:hover {
    background: #002200;
    color: #00ff44;
}

.menu-item.-active {
    background: #003300;
    color: #00ff44;
    text-style: bold;
}

.menu-key {
    color: #00ff44;
    text-style: bold;
}

.menu-label {
    color: #009900;
}

/* ─── STATUS PANEL ──────────────────────────────────────────────── */
#status-panel {
    padding: 0 1;
}

.status-label {
    color: #009900;
}

.status-value {
    color: #00bb00;
}

.status-ok {
    color: #00ff44;
}

.status-warn {
    color: #ddaa00;
}

.status-err {
    color: #cc2200;
}

/* ─── ASSISTANT ─────────────────────────────────────────────────── */
#assistant-scroll {
    height: 1fr;
    border: double #004400;
    padding: 0 1;
}

.section-header {
    color: #00ff44;
    text-style: bold;
    padding: 1 0 0 0;
    border-bottom: solid #003300;
}

.profile-card {
    border: solid #003300;
    padding: 0 1;
    margin: 1 0;
    height: auto;
}

.profile-name {
    color: #00ff44;
    text-style: bold;
}

.profile-desc {
    color: #009900;
}

.profile-stat {
    color: #ddaa00;
}

.profile-args {
    color: #006600;
}

/* ─── MODEL LIBRARY ─────────────────────────────────────────────── */
DataTable {
    background: #000000;
    color: #009900;
    border: double #004400;
}

DataTable > .datatable--header {
    background: #001100;
    color: #00ff44;
    text-style: bold;
}

DataTable > .datatable--cursor {
    background: #003300;
    color: #00ff44;
}

DataTable > .datatable--even-row {
    background: #000800;
}

/* ─── CONFIG WIZARD ─────────────────────────────────────────────── */
#wizard-container {
    height: 100%;
    border: double #004400;
    padding: 0 2;
}

.wizard-step {
    color: #009900;
    padding: 1 0 0 0;
}

.wizard-field {
    padding: 0 0 1 0;
}

.wizard-field Label {
    color: #00bb00;
    padding: 0;
}

.wizard-field Input {
    background: #001100;
    color: #00ff44;
    border: solid #004400;
}

.wizard-field Input:focus {
    border: solid #009900;
}

/* ─── BUTTONS ───────────────────────────────────────────────────── */
Button {
    background: #001100;
    color: #00bb00;
    border: solid #004400;
    min-width: 12;
}

Button:hover {
    background: #003300;
    color: #00ff44;
    border: solid #00aa00;
}

Button.-primary {
    background: #002200;
    color: #00ff44;
    border: solid #009900;
    text-style: bold;
}

Button.-primary:hover {
    background: #003300;
    border: solid #00ff44;
}

/* ─── FOOTER ────────────────────────────────────────────────────── */
Footer {
    background: #001100;
    color: #005500;
}

/* ─── SCREEN HEADERS ────────────────────────────────────────────── */
.screen-title {
    background: #001100;
    color: #00ff44;
    text-style: bold;
    text-align: center;
    padding: 0 1;
    height: 3;
    content-align: center middle;
    border-bottom: solid #004400;
}

.screen-subtitle {
    color: #006600;
    text-align: center;
    height: 1;
}

/* ─── BACK BUTTON ───────────────────────────────────────────────── */
#back-bar {
    height: 3;
    align: left middle;
    padding: 0 1;
    border-top: solid #003300;
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

    # Sort models by VRAM need
    by_vram = sorted(models, key=lambda m: vram_need(m))

    # ── Profile: single best model ──────────────────────────────────────────
    # Pick largest model that fits in total VRAM
    for m in reversed(by_vram):
        need = vram_need(m)
        if need <= total_free * 0.92:
            tp = 1 if max_single >= need else min(gpu_count, 2)
            profiles.append({
                "id":       "single-heavy",
                "name":     "SINGLE · MAXIMUM POWER",
                "emoji":    "⚡",
                "models":   [m["name"]],
                "total_gb": need,
                "tp":       tp,
                "desc":     f"Run {m['name']} solo — all VRAM dedicated to one model",
                "ctx_hint": "Maximum context window, high quality outputs",
                "tags":     ["quality", "single-model"],
            })
            break

    # ── Profile: dual light models ───────────────────────────────────────────
    # Each model must fit on a single GPU (TP=1), and together fit in total VRAM
    light = [m for m in by_vram if vram_need(m) <= max_single * 0.92]
    # Deduplicate by similar name (avoid pairing Qwen3vl-8b + Qwen3vl-8b-caption)
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
                "emoji":    "⚡⚡",
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
                    "emoji":    "👁",
                    "models":   [v["name"], t["name"]],
                    "total_gb": combined,
                    "tp":       1,
                    "desc":     "Vision model + text model — image analysis + conversation",
                    "ctx_hint": "Ideal for multimodal workflows",
                    "tags":     ["vision", "multi-model"],
                })

    # ── Profile: speed / low latency — smallest model that fits single GPU ──
    already_in_profiles = {n for p in profiles for n in p["models"]}
    solo_candidates = [m for m in by_vram
                       if vram_need(m) <= max_single * 0.92
                       and m["name"] not in already_in_profiles]
    if not solo_candidates:
        # Fall back to the smallest model overall
        solo_candidates = [by_vram[0]] if by_vram else []
    if solo_candidates:
        m = solo_candidates[0]   # smallest
        profiles.append({
            "id":       "speed",
            "name":     "SPEED · LOW LATENCY",
            "emoji":    "▶",
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
            yield Static(LOGO_LARGE, id="boot-logo")
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
        total_steps = 18  # approximate

        # Build messages progressively
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
            "ALLAMA TERMINAL INTERFACE  v1.0",
            "(c) 2024  ·  PHOSPHOR GREEN EDITION",
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
# Main Menu Screen
# ──────────────────────────────────────────────────────────────────────────────
class MainMenuScreen(Screen):
    BINDINGS = [
        Binding("1", "menu_1", ""),
        Binding("2", "menu_2", ""),
        Binding("3", "menu_3", ""),
        Binding("4", "menu_4", ""),
        Binding("5", "menu_5", ""),
        Binding("q", "quit",   "Quit"),
    ]

    def __init__(self, gpus, online, srv, phys, logical, **kwargs):
        super().__init__(**kwargs)
        self._gpus    = gpus
        self._online  = online
        self._srv     = srv
        self._phys    = phys
        self._logical = logical

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-layout"):
            # ── Left: menu ──────────────────────────────────────────────────
            with Vertical(id="left-panel"):
                yield Static(LOGO_SMALL, id="boot-logo")
                yield Static("LLM PROXY TERMINAL  ·  v1.0", classes="screen-subtitle")
                yield Static("")
                yield Static("MAIN MENU", classes="panel-title")
                yield Static("─" * 36, classes="separator")
                for key, label, _ in self._menu_items():
                    yield Static(
                        f"  [{key}]  {label}",
                        classes="menu-item",
                        id=f"menu-{key}",
                    )
                yield Static("")
                yield Static("─" * 36, classes="separator")
                yield Static("  [Q]  QUIT", classes="menu-item")

            # ── Right: status ───────────────────────────────────────────────
            with Vertical(id="right-panel"):
                yield Static("SYSTEM STATUS", classes="panel-title")
                yield Static("")
                yield Static(self._gpu_status(), id="gpu-status")
                yield Static("")
                yield Static(self._server_status(), id="server-status")
                yield Static("")
                yield Static(self._model_status(), id="model-status")

    def _menu_items(self):
        return [
            ("1", "CONFIG ASSISTANT",   "assistant"),
            ("2", "MODEL LIBRARY",      "models"),
            ("3", "CREATE CONFIG",      "creator"),
            ("4", "PROFILES",           "profiles"),
            ("5", "SYSTEM INFO",        "sysinfo"),
        ]

    def _gpu_status(self) -> str:
        if not self._gpus:
            return "GPU STATUS\n══════════\n  NO GPUs DETECTED"
        lines = ["GPU STATUS", "══════════"]
        for g in self._gpus:
            used  = g["total_gb"] - g["free_gb"]
            b     = bar(used, g["total_gb"], 10)
            lines.append(f"GPU {g['index']}  [{b}]")
            lines.append(f"      {used:.0f}/{g['total_gb']:.0f}GB  FREE:{g['free_gb']:.0f}GB")
        return "\n".join(lines)

    def _server_status(self) -> str:
        if self._online:
            active = self._srv.get("active_servers", 0)
            return f"SERVER STATUS\n═════════════\n  ● ALLAMA  :9000  ONLINE\n  ACTIVE: {active} model(s)"
        return "SERVER STATUS\n═════════════\n  ○ ALLAMA  :9000  OFFLINE"

    def _model_status(self) -> str:
        lines = ["CONFIG FILES", "════════════",
                 f"  PHYSICAL : {len(self._phys)}",
                 f"  LOGICAL  : {len(self._logical)}"]
        return "\n".join(lines)

    # ── Menu actions ────────────────────────────────────────────────────────
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
        if key == "1":   self.action_menu_1()
        elif key == "2": self.action_menu_2()
        elif key == "3": self.action_menu_3()
        elif key == "4": self.action_menu_4()
        elif key == "5": self.action_menu_5()
        elif key in ("q", "Q"): self.action_quit()


# ──────────────────────────────────────────────────────────────────────────────
# Config Assistant Screen
# ──────────────────────────────────────────────────────────────────────────────
class AssistantScreen(Screen):
    BINDINGS = [Binding("escape", "back", "Back"), Binding("q", "back", "Back")]

    def __init__(self, gpus, phys_models, logical_models, **kwargs):
        super().__init__(**kwargs)
        self._gpus    = gpus
        self._phys    = phys_models
        self._logical = logical_models

    def compose(self) -> ComposeResult:
        yield Static("[ CONFIG ASSISTANT ]", classes="screen-title")
        with ScrollableContainer(id="assistant-scroll"):
            yield Static(self._hardware_section(), id="hw-section")
            yield Static(self._models_section(),   id="models-section")
            yield Static(self._profiles_section(), id="profiles-section")
            yield Static(self._tips_section(),     id="tips-section")
        with Horizontal(id="back-bar"):
            yield Button("◄ BACK  [ESC]", id="back-btn", variant="default")

    # ── Render sections ──────────────────────────────────────────────────────
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
    BINDINGS = [Binding("escape", "back", "Back")]

    def __init__(self, phys, logical, **kwargs):
        super().__init__(**kwargs)
        self._phys    = phys
        self._logical = logical

    def compose(self) -> ComposeResult:
        yield Static("[ MODEL LIBRARY ]", classes="screen-title")
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

    def on_mount(self) -> None:
        # Physical table
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

        # Logical table
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
    BINDINGS = [Binding("escape", "back", "Back")]

    def __init__(self, gpus, phys, **kwargs):
        super().__init__(**kwargs)
        self._gpus  = gpus
        self._phys  = phys

    def compose(self) -> ComposeResult:
        yield Static("[ CONFIGURATION PROFILES ]", classes="screen-title")
        profiles = build_profiles(self._gpus, self._phys)
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
# Config Creator Screen  (TUI wizard)
# ──────────────────────────────────────────────────────────────────────────────
class ConfigCreatorScreen(Screen):
    BINDINGS = [Binding("escape", "back", "Back")]

    def compose(self) -> ComposeResult:
        yield Static("[ CREATE CONFIG ]", classes="screen-title")
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
    BINDINGS = [Binding("escape", "back", "Back")]

    def __init__(self, gpus, online, srv, **kwargs):
        super().__init__(**kwargs)
        self._gpus   = gpus
        self._online = online
        self._srv    = srv

    def compose(self) -> ComposeResult:
        yield Static("[ SYSTEM INFORMATION ]", classes="screen-title")
        with ScrollableContainer(id="assistant-scroll"):
            yield Static(self._render_all(), id="sysinfo-content")
        with Horizontal(id="back-bar"):
            yield Button("◄ BACK  [ESC]", id="back-btn")
            yield Button("⟳ REFRESH", id="refresh-btn")

    def _render_all(self) -> str:
        lines = [""]

        # GPU info
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

        # Allama server
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

        # Environment
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

        # Config files
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
    TITLE        = "ALLAMA TERMINAL INTERFACE"
    SUB_TITLE    = "PHOSPHOR GREEN EDITION  ·  v1.0"
    CSS          = CSS
    ENABLE_COMMAND_PALETTE = False

    def on_mount(self) -> None:
        self.push_screen(BootScreen())


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = AllamaTUI()
    app.run()
