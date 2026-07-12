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
from rich.columns import Columns
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
             "temperature.gpu,power.draw,power.limit,clocks_throttle_reasons.active,"
             "clocks.current.sm,clocks.max.sm,fan.speed",
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
        def _f(i, d=0.0):
            try: return float(p[i])
            except (ValueError, IndexError): return d
        try:
            rows.append({
                "index": int(p[0]), "name": p[1],
                "util": float(p[2] or 0), "used": float(p[3] or 0) / 1024,
                "total": float(p[4] or 1) / 1024, "temp": float(p[5] or 0),
                "power": float(p[6] or 0), "plimit": float(p[7] or 1),
                "throttled": bool(throttle & _THROTTLE_MASK),
                "clock": _f(9), "clock_max": _f(10, 1), "fan": _f(11),
            })
        except ValueError:
            continue
    return rows


def _vllm_stats(port: int) -> Optional[dict]:
    """Parse the Prometheus text: token counters + kv usage + request gauges."""
    text = _http_text(f"http://127.0.0.1:{port}/metrics")
    if text is None:
        return None
    out = {"gen_tokens": 0.0, "prompt_tokens": 0.0, "kv_perc": 0.0,
           "running": 0.0, "waiting": 0.0}
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
        elif metric.startswith("vllm:prompt_tokens_total"):
            out["prompt_tokens"] += v
        elif metric.startswith("vllm:gpu_cache_usage_perc"):
            out["kv_perc"] = max(out["kv_perc"], v)
        elif metric.startswith("vllm:num_requests_running"):
            out["running"] += v
        elif metric.startswith("vllm:num_requests_waiting"):
            out["waiting"] += v
    return out


def _llama_stats(port: int) -> Optional[dict]:
    """llama-server /slots (context) + /metrics. llama.cpp exposes the
    instantaneous gen/prefill rates directly, so no delta math is needed.
    Both endpoints need --slots/--metrics (allma adds them)."""
    slots = _http_json(f"http://127.0.0.1:{port}/slots")
    metrics = _http_text(f"http://127.0.0.1:{port}/metrics")
    if not isinstance(slots, list) and metrics is None:
        return None
    out = {"busy": 0, "ctx_used": 0, "n_ctx": 0,
           "gen_tps": None, "prompt_tps": None, "pred_tokens": None}
    if isinstance(slots, list):
        for s in slots:
            if not isinstance(s, dict):
                continue
            out["n_ctx"] = max(out["n_ctx"], int(s.get("n_ctx") or 0))
            # no n_past field in modern llama-server; n_prompt_tokens is the
            # slot's current context fill (prompt + generated so far).
            used = int(s.get("n_prompt_tokens") or s.get("n_past") or 0)
            out["ctx_used"] = max(out["ctx_used"], used)
            if s.get("is_processing"):
                out["busy"] += 1
    if metrics is not None:
        total = 0.0
        for line in metrics.splitlines():
            if line.startswith("llamacpp:predicted_tokens_seconds"):
                try: out["gen_tps"] = float(line.rsplit(" ", 1)[1])
                except (ValueError, IndexError): pass
            elif line.startswith("llamacpp:prompt_tokens_seconds"):
                try: out["prompt_tps"] = float(line.rsplit(" ", 1)[1])
                except (ValueError, IndexError): pass
            if line.startswith("llamacpp:tokens_predicted_total"):
                try:
                    total += float(line.rsplit(" ", 1)[1])
                except (ValueError, IndexError):
                    pass
        out["pred_tokens"] = total
    return out


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


def _kfmt(n) -> str:
    """Compact token counts: 21491 → 21.5k, 1250000 → 1.2M."""
    n = float(n or 0)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return f"{n:.0f}"


class TopView:
    def __init__(self):
        # per-model tok/s state: name -> (last_counter, last_ts, rate)
        self._rate: dict = {}
        self._maxctx: dict = {}
        self._last_active: dict = {}   # name -> last rate seen while generating

    def _rate_text(self, name: str, rate: float, active: bool) -> Text:
        """Live rate while generating; the last active rate (dimmed) when idle."""
        if active and rate > 0.05:
            self._last_active[name] = rate
            return Text(f"{rate:6.1f} tok/s", style=f"bold {C_ACCENT} on {C_BG}")
        last = self._last_active.get(name)
        if last:
            return Text(f"idle · last {last:.1f} tok/s", style=f"{C_DIM} on {C_BG}")
        return Text("idle", style=f"{C_DIM} on {C_BG}")

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

    # ── card builders ─────────────────────────────────────────────────────
    def _barrow(self, label: str, frac: float, value: str,
                width: int = 14, color: Optional[str] = None) -> Text:
        """'label  ██████░░░  value' — one aligned bar row inside a card."""
        t = Text(style=_S)
        t.append(f"{label:<5}", style=f"{C_DIM} on {C_BG}")
        t.append_text(_bar(frac, width=width, color=color))
        t.append(f" {value}", style=f"{C_FG} on {C_BG}")
        return t

    def _gpu_card(self, g: dict) -> Panel:
        short = (g["name"].replace("NVIDIA GeForce ", "").replace("NVIDIA ", ""))[:10]
        grid = Table.grid(padding=(0, 0))
        grid.add_column()
        grid.add_row(self._barrow("util", g["util"] / 100, f"{g['util']:3.0f}%"))
        grid.add_row(self._barrow("vram", g["used"] / g["total"],
                                  f"{g['used']:.1f}/{g['total']:.0f}G"))
        # footer: temp (colored) · clock · power · fan
        foot = Text(style=_S)
        foot.append(f"{g['temp']:.0f}°C", style=f"bold {_temp_style(g['temp'])} on {C_BG}")
        foot.append(f"  {g['clock']:.0f}MHz", style=f"{C_DIM} on {C_BG}")
        foot.append(f"  {g['power']:.0f}W", style=f"{C_DIM} on {C_BG}")
        if g["fan"]:
            foot.append(f"  ~{g['fan']:.0f}%", style=f"{C_DIM} on {C_BG}")
        if g["throttled"]:
            foot.append("  ⚠ throttle", style=f"bold {C_BAD} on {C_BG}")
        grid.add_row(foot)
        title = Text.assemble((f" GPU {g['index']} ", f"bold {C_ACCENT}"),
                              (short, C_DIM))
        return Panel(grid, title=title, title_align="left", box=_box.ROUNDED,
                     border_style=C_DIM, style=_S, padding=(0, 1), width=34)

    def _model_card(self, s: dict) -> Panel:
        name, port, backend = s["name"], s["port"], s.get("backend", "?")
        grid = Table.grid(padding=(0, 0))
        grid.add_column()

        rate_line = Text("idle", style=f"{C_DIM} on {C_BG}")   # the big number
        detail = Table.grid(padding=(0, 0))
        detail.add_column()

        if backend == "vllm":
            m = _vllm_stats(port)
            if m:
                rate = self._model_rate(name, m["gen_tokens"])
                rate_line = self._rate_text(name, rate, m["running"] > 0)
                if name not in self._maxctx:
                    self._maxctx[name] = _max_ctx_of(port)
                maxctx = self._maxctx.get(name) or 0
                detail.add_row(self._barrow("kv", m["kv_perc"],
                               f"{m['kv_perc'] * 100:.0f}%", width=12))
                foot = Text(style=_S)
                foot.append(f"reqs {m['running']:.0f}▶ {m['waiting']:.0f}⏸",
                            style=f"{C_DIM} on {C_BG}")
                foot.append(f"   gen {_kfmt(m['gen_tokens'])}", style=f"{C_DIM} on {C_BG}")
                if maxctx:
                    foot.append(f"   ctx≤{_kfmt(maxctx)}", style=f"{C_DIM} on {C_BG}")
                detail.add_row(foot)
            else:
                detail.add_row(Text("metrics unavailable", style=f"{C_DIM} on {C_BG}"))
        else:  # llama.cpp
            m = _llama_stats(port)
            if m:
                if m["gen_tps"] is not None:
                    rate_line = self._rate_text(name, m["gen_tps"], m["busy"] > 0)
                if m["n_ctx"]:
                    detail.add_row(self._barrow(
                        "ctx", m["ctx_used"] / m["n_ctx"],
                        f"{_kfmt(m['ctx_used'])}/{_kfmt(m['n_ctx'])}", width=12))
                foot = Text(style=_S)
                if m["prompt_tps"]:
                    foot.append(f"prefill {m['prompt_tps']:.0f} t/s",
                                style=f"{C_DIM} on {C_BG}")
                foot.append(f"   slots {m['busy']}▶", style=f"{C_DIM} on {C_BG}")
                detail.add_row(foot)
                if not m["n_ctx"] and m["gen_tps"] is None:
                    detail.add_row(Text("'allma reload' to enable metrics",
                                        style=f"{C_DIM} on {C_BG}"))
            else:
                detail.add_row(Text("up — 'allma reload' to enable metrics",
                                    style=f"{C_DIM} on {C_BG}"))

        grid.add_row(rate_line)
        grid.add_row(detail)
        short = name if len(name) <= 22 else name[:21] + "…"
        title = Text.assemble(("● ", C_GOOD), (short, f"bold {C_FG}"),
                              (f"  {backend}·gpu{s.get('gpu', '?')}", C_DIM))
        return Panel(grid, title=title, title_align="left", box=_box.ROUNDED,
                     border_style=C_DIM, style=_S, padding=(0, 1), width=44)

    def snapshot(self) -> Panel:
        gpus = _gpus()
        ps = _http_json(f"{ALLMA_URL}/v1/ps", timeout=1.5) or {}
        servers = [s for s in ps.get("servers", []) if s.get("alive")]

        sections = []
        if gpus:
            sections.append(Columns([self._gpu_card(g) for g in gpus],
                                    padding=(0, 1), expand=False))
        else:
            sections.append(Text("  no NVIDIA GPU detected", style=f"{C_WARN} on {C_BG}"))
        sections.append(Text(""))
        if servers:
            sections.append(Columns([self._model_card(s) for s in servers],
                                    padding=(0, 1), expand=False))
        else:
            sections.append(Text("  no models loaded", style=f"{C_DIM} on {C_BG}"))

        stamp = time.strftime("%H:%M:%S")
        inner = Panel(
            Group(*sections),
            title=_section(f"allma top · {stamp}"),
            title_align="left",
            subtitle=Text(" q to quit ", style=f"{C_DIM} on {C_BG}"),
            subtitle_align="right",
            box=_box.SQUARE,
            border_style=C_DIM,
            style=_S,
            padding=(1, 1),
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
