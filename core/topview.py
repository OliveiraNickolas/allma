"""`allma top` — a small live dashboard: GPUs (util / VRAM / temp / power /
throttle) and loaded models (tok/s, KV-cache usage, active requests).

Deliberately NOT overwhelming: one line per GPU, two lines per model.
Data sources:
  - nvidia-smi (GPU line)
  - vLLM  :port/metrics   (Prometheus text — token counters, kv usage)
  - llama :port/slots     (per-slot context usage)
"""
import json
import subprocess
import sys
import time
import urllib.request
from typing import Optional

from rich import box as _box
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from core.downloader import (
    C_ACCENT, C_BAD, C_BG, C_BORDER, C_DIM, C_FG, C_GOOD, C_ORANGE,
    C_SCREEN, C_WARN, _S, _W, _section, console,
)

ALLMA_URL = "http://127.0.0.1:9000"

# nvidia-smi throttle bitmask: HW slowdown 0x8, SW thermal 0x20,
# HW thermal 0x40, HW power brake 0x80. (Idle 0x1 / power cap 0x4 are normal.)
_THROTTLE_MASK = 0x8 | 0x20 | 0x40 | 0x80


# ── data collection ───────────────────────────────────────────────────────────
def _http_json(url: str, timeout: float = 0.8) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers={"Authorization": "Bearer dummy"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _http_text(url: str, timeout: float = 0.8) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={"Authorization": "Bearer dummy"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def _gpus() -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
             "temperature.gpu,power.draw,power.limit,clocks_throttle_reasons.active",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 9:
            continue
        try:
            throttle = int(p[8], 16) if p[8].startswith("0x") else 0
        except ValueError:
            throttle = 0
        try:
            rows.append({
                "index": int(p[0]), "name": p[1],
                "util": float(p[2] or 0), "used": float(p[3] or 0) / 1024,
                "total": float(p[4] or 1) / 1024, "temp": float(p[5] or 0),
                "power": float(p[6] or 0), "plimit": float(p[7] or 1),
                "throttled": bool(throttle & _THROTTLE_MASK),
            })
        except ValueError:
            continue
    return rows


def _vllm_stats(port: int) -> Optional[dict]:
    """Parse the Prometheus text: token counters + kv usage + request gauges."""
    text = _http_text(f"http://127.0.0.1:{port}/metrics")
    if text is None:
        return None
    out = {"gen_tokens": 0.0, "kv_perc": 0.0, "running": 0.0, "waiting": 0.0}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        try:
            metric, value = line.rsplit(" ", 1)
            v = float(value)
        except ValueError:
            continue
        if metric.startswith("vllm:generation_tokens_total"):
            out["gen_tokens"] += v
        elif metric.startswith("vllm:gpu_cache_usage_perc"):
            out["kv_perc"] = max(out["kv_perc"], v)
        elif metric.startswith("vllm:num_requests_running"):
            out["running"] += v
        elif metric.startswith("vllm:num_requests_waiting"):
            out["waiting"] += v
    return out


def _llama_stats(port: int) -> Optional[dict]:
    """llama-server /slots: context used per slot."""
    slots = _http_json(f"http://127.0.0.1:{port}/slots")
    if not isinstance(slots, list):
        return None
    busy = 0
    ctx_used = 0
    n_ctx = 0
    for s in slots:
        n_ctx = max(n_ctx, int(s.get("n_ctx") or 0))
        past = int(s.get("n_past") or s.get("n_ctx_used") or 0)
        ctx_used = max(ctx_used, past)
        if s.get("is_processing") or s.get("state") in (1, "processing"):
            busy += 1
    return {"busy": busy, "ctx_used": ctx_used, "n_ctx": n_ctx}


def _max_ctx_of(port: int) -> int:
    data = _http_json(f"http://127.0.0.1:{port}/v1/models", timeout=1.5)
    try:
        return int((data.get("data") or [{}])[0].get("max_model_len") or 0)
    except Exception:
        return 0


# ── rendering ─────────────────────────────────────────────────────────────────
def _bar(frac: float, width: int = 10, color: Optional[str] = None) -> Text:
    frac = max(0.0, min(1.0, frac))
    filled = round(frac * width)
    if color is None:
        color = C_GOOD if frac < 0.7 else (C_WARN if frac < 0.9 else C_BAD)
    t = Text()
    t.append("▰" * filled, style=color)
    t.append("▱" * (width - filled), style="#c9bfa2")
    return t


def _temp_style(temp: float) -> str:
    return C_GOOD if temp < 70 else (C_WARN if temp < 80 else f"bold {C_BAD}")


class TopView:
    def __init__(self):
        # per-model tok/s state: name -> (last_counter, last_ts, rate)
        self._rate: dict = {}
        self._maxctx: dict = {}

    def _model_rate(self, name: str, gen_tokens: float) -> float:
        now = time.time()
        last = self._rate.get(name)
        rate = 0.0
        if last:
            dt = now - last[1]
            if dt > 0 and gen_tokens >= last[0]:
                inst = (gen_tokens - last[0]) / dt
                # light smoothing so the number doesn't jitter
                rate = inst if last[2] == 0 else (0.6 * inst + 0.4 * last[2])
        self._rate[name] = (gen_tokens, now, rate)
        return rate

    def snapshot(self) -> Panel:
        gpus = _gpus()
        ps = _http_json(f"{ALLMA_URL}/v1/ps", timeout=1.5) or {}
        servers = [s for s in ps.get("servers", []) if s.get("alive")]

        body = Table.grid(expand=True, padding=(0, 1))
        body.add_column()

        # ── GPU lines ────────────────────────────────────────────────────
        for g in gpus:
            line = Text(style=_S)
            short = (g["name"].replace("NVIDIA GeForce ", "")
                     .replace("NVIDIA ", ""))[:8]
            line.append(f" GPU{g['index']} ", style=f"bold {C_ACCENT} on {C_BG}")
            line.append(f"{short:<8} ", style=f"{C_DIM} on {C_BG}")
            line.append("util ", style=f"{C_DIM} on {C_BG}")
            line.append_text(_bar(g["util"] / 100, width=8))
            line.append(f" {g['util']:3.0f}% ", style=f"{C_FG} on {C_BG}")
            line.append("vram ", style=f"{C_DIM} on {C_BG}")
            line.append_text(_bar(g["used"] / g["total"], width=8))
            line.append(f" {g['used']:4.1f}/{g['total']:.0f}G ", style=f"{C_FG} on {C_BG}")
            line.append(f"{g['temp']:3.0f}°C ", style=f"{_temp_style(g['temp'])} on {C_BG}")
            line.append(f"{g['power']:3.0f}W", style=f"{C_DIM} on {C_BG}")
            if g["throttled"]:
                line.append("  ⚠ throttle", style=f"bold {C_BAD} on {C_BG}")
            body.add_row(line)
        if not gpus:
            body.add_row(Text(" no NVIDIA GPU detected", style=f"{C_WARN} on {C_BG}"))

        body.add_row(Text(""))

        # ── model lines ──────────────────────────────────────────────────
        if not servers:
            body.add_row(Text(" no models loaded", style=f"{C_DIM} on {C_BG}"))
        for s in servers:
            name, port, backend = s["name"], s["port"], s.get("backend", "?")
            head = Text(style=_S)
            head.append(f" {name} ", style=f"bold {C_FG} on {C_BG}")
            head.append(f"{backend} :{port}  gpu{s.get('gpu', '?')}",
                        style=f"{C_DIM} on {C_BG}")

            detail = Text(style=_S)
            if backend == "vllm":
                m = _vllm_stats(port)
                if m:
                    rate = self._model_rate(name, m["gen_tokens"])
                    head.append(f"   {rate:6.1f} tok/s",
                                style=f"bold {C_ACCENT} on {C_BG}")
                    if name not in self._maxctx:
                        self._maxctx[name] = _max_ctx_of(port)
                    maxctx = self._maxctx.get(name) or 0
                    detail.append("   kv-cache ", style=f"{C_DIM} on {C_BG}")
                    detail.append_text(_bar(m["kv_perc"]))
                    detail.append(f" {m['kv_perc'] * 100:3.0f}%", style=f"{C_FG} on {C_BG}")
                    detail.append(f"   reqs {m['running']:.0f} run · "
                                  f"{m['waiting']:.0f} wait", style=f"{C_DIM} on {C_BG}")
                    if maxctx:
                        detail.append(f"   ctx max {maxctx}", style=f"{C_DIM} on {C_BG}")
                else:
                    detail.append("   metrics unavailable", style=f"{C_DIM} on {C_BG}")
            else:  # llama.cpp
                m = _llama_stats(port)
                if m and m["n_ctx"]:
                    frac = m["ctx_used"] / m["n_ctx"] if m["n_ctx"] else 0.0
                    detail.append("   ctx ", style=f"{C_DIM} on {C_BG}")
                    detail.append_text(_bar(frac))
                    detail.append(f" {m['ctx_used']}/{m['n_ctx']}",
                                  style=f"{C_FG} on {C_BG}")
                    detail.append(f"   slots busy {m['busy']}", style=f"{C_DIM} on {C_BG}")
                else:
                    detail.append("   up (no /slots data)", style=f"{C_DIM} on {C_BG}")
            body.add_row(head)
            body.add_row(detail)

        stamp = time.strftime("%H:%M:%S")
        inner = Panel(
            body,
            title=_section(f"allma top · {stamp}"),
            title_align="left",
            subtitle=Text(" q to quit ", style=f"{C_DIM} on {C_BG}"),
            subtitle_align="right",
            box=_box.SQUARE,
            border_style=C_DIM,
            style=_S,
            padding=(0, 1),
        )
        return Panel(inner, box=_box.DOUBLE, border_style=C_BORDER,
                     style=f"on {C_SCREEN}", padding=(0, 0), width=_W())


def run_top(interval: float = 1.0, once: bool = False) -> None:
    view = TopView()

    if once:
        # Two samples so tok/s has a delta to work with.
        view.snapshot()
        time.sleep(min(1.0, interval))
        console.print(view.snapshot())
        return

    # q-to-quit without blocking the refresh loop
    import select
    import termios
    import tty
    is_tty = sys.stdin.isatty()
    old_attrs = None
    if is_tty:
        try:
            old_attrs = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except Exception:
            old_attrs = None
    try:
        with Live(view.snapshot(), console=console, screen=True,
                  auto_refresh=False) as live:
            while True:
                if is_tty:
                    r, _, _ = select.select([sys.stdin], [], [], interval)
                    if r and sys.stdin.read(1).lower() == "q":
                        break
                else:
                    time.sleep(interval)
                live.update(view.snapshot(), refresh=True)
    except KeyboardInterrupt:
        pass
    finally:
        if old_attrs is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attrs)
            except Exception:
                pass
