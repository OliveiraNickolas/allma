#!/usr/bin/env python3
"""
Allama CLI — allama serve / run / list / ps / stop / logs / launch
"""
import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
ALLAMA_DIR  = Path(__file__).parent
ALLAMA_PORT = int(os.environ.get("ALLAMA_PORT", "9000"))
BASE_URL    = f"http://127.0.0.1:{ALLAMA_PORT}"
PID_FILE    = Path(os.environ.get("ALLAMA_PID_FILE", "/tmp/allama_watchdog.pid"))
LOG_FILE    = ALLAMA_DIR / "logs" / "allama.log"
PYTHON      = sys.executable
SERVER_SCRIPT = str(ALLAMA_DIR / "allama.py")

# ── Helpers ────────────────────────────────────────────────────────────────────
def _get(path: str, timeout: float = 3.0):
    """Simple HTTP GET, returns parsed JSON or None."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _is_running() -> bool:
    return _get("/health") is not None


def _run_spinner(stop_event: threading.Event, label_ref: list):
    """3-row parallax Andes spinner with running llama."""
    ci = si = ni = 0
    last_c = last_s = last_n = 0
    start = time.time()
    tick = 0

    sys.stdout.write("\n\n")  # reserve 3 lines

    while not stop_event.is_set():
        elapsed = time.time() - start
        cview = (_SPINNER_CLOUDS    * 2)[ci % len(_SPINNER_CLOUDS):    ci % len(_SPINNER_CLOUDS)    + _WINDOW]
        sview = (_SPINNER_SKY       * 2)[si % len(_SPINNER_SKY):       si % len(_SPINNER_SKY)       + _WINDOW]
        nview = (_SPINNER_MOUNTAINS * 2)[ni % len(_SPINNER_MOUNTAINS): ni % len(_SPINNER_MOUNTAINS) + _WINDOW]
        cview, sview, nview = _inject_llama(cview, sview, nview, tick)

        cloud_line = f"  {cview}"
        sky_line   = f"  {sview}"
        near_line  = f"  {nview}  {label_ref[0]}{elapsed:.1f}s"

        sys.stdout.write(f"\033[2A\r{' ' * last_c}\r{cloud_line}\n")
        sys.stdout.write(f"\r{' ' * last_s}\r{sky_line}\n")
        sys.stdout.write(f"\r{' ' * last_n}\r{near_line}")
        sys.stdout.flush()

        last_c = len(cloud_line)
        last_s = len(sky_line)
        last_n = len(near_line)

        time.sleep(0.06)
        tick += 1
        if tick % 5 == 0:
            ci += 1
        if tick % 3 == 0:
            si += 1
        if tick % 2 == 0:
            ni += 1

    sys.stdout.write(f"\r{' ' * last_n}\r")
    sys.stdout.write(f"\033[1A\r{' ' * last_s}\r")
    sys.stdout.write(f"\033[1A\r{' ' * last_c}\r")
    sys.stdout.flush()


def _post(path: str, body: dict, timeout: float = 300.0):
    """Simple HTTP POST, returns parsed JSON or None."""
    try:
        import urllib.request as _ur
        data = json.dumps(body).encode()
        req = _ur.Request(
            f"{BASE_URL}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with _ur.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except KeyboardInterrupt:
        raise
    except Exception:
        return None


def _wait_for_server(timeout: int = 30) -> bool:
    """Block until server responds or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _is_running():
            return True
        time.sleep(0.5)
    return False


def _start_daemon() -> bool:
    """Start the watchdog daemon in background. Returns True if newly started."""
    if _is_running():
        return False

    # Spawn watchdog detached from terminal
    proc = subprocess.Popen(
        [PYTHON, __file__, "__watchdog__"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(proc.pid))
    return True


def _read_pid() -> int | None:
    try:
        return int(PID_FILE.read_text().strip())
    except Exception:
        return None


# ── Watchdog (internal, not user-facing) ───────────────────────────────────────
def _run_watchdog(verbose: bool):
    """
    Loop forever restarting allama.py if it dies.
    Called internally when this script is run as __watchdog__.
    """
    PID_FILE.write_text(str(os.getpid()))
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _sigterm(sig, frame):
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sigterm)

    restart_count = 0
    while True:
        try:
            if verbose:
                proc = subprocess.run([PYTHON, SERVER_SCRIPT])
            else:
                with open(LOG_FILE, "a") as log:
                    proc = subprocess.run(
                        [PYTHON, SERVER_SCRIPT],
                        stdout=log,
                        stderr=log,
                    )
        except KeyboardInterrupt:
            break

        code = proc.returncode

        # Clean shutdown (SIGINT / ctrl+c forwarded) — don't restart
        if code in (0, -signal.SIGINT, 130):
            break

        restart_count += 1
        msg = f"[allama] process exited (code {code}), restarting in 3s... (#{restart_count})"
        if verbose:
            print(msg, flush=True)
        else:
            with open(LOG_FILE, "a") as log:
                log.write(msg + "\n")
        time.sleep(3)

    PID_FILE.unlink(missing_ok=True)


# ── Commands ───────────────────────────────────────────────────────────────────
def cmd_serve(args):
    """Start the Allama daemon."""
    if args.verbose:
        print("Starting Allama (verbose mode — Ctrl+C to stop)...")
        _run_watchdog(verbose=True)
    else:
        if _is_running():
            print("Allama is already running.")
            return
        label = ["Starting Allama...  "]
        stop_spinner = threading.Event()
        spinner = threading.Thread(target=_run_spinner, args=(stop_spinner, label), daemon=True)
        spinner.start()
        _start_daemon()
        ok = _wait_for_server(30)
        stop_spinner.set()
        spinner.join()
        if ok:
            print("Allama ready.")
        else:
            print("Allama timed out. Check logs: allama logs")


def _kill_leftover_backends() -> int:
    """Kill any allama-managed vllm/llama-server processes still alive.

    Uses command-line fingerprints specific to allama-spawned backends
    (--host 127.0.0.1 --api-key dummy for vllm, --host 127.0.0.1 for llama)
    to avoid hitting unrelated instances.
    """
    killed = 0
    try:
        import psutil
        uid = os.getuid()
        for proc in psutil.process_iter(["pid", "cmdline", "uids"]):
            try:
                if proc.uids().real != uid:
                    continue
                cmd = " ".join(proc.cmdline())
                is_vllm  = "vllm" in cmd and "serve" in cmd and "--api-key dummy" in cmd
                is_llama = "llama-server" in cmd and "--host 127.0.0.1" in cmd
                if is_vllm or is_llama:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        proc.kill()
                    killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        pass
    return killed


def cmd_stop(args):
    """Stop the Allama daemon and all backend processes."""
    if not _is_running() and not _read_pid():
        print("Allama is not running.")
        return

    # Step 1: ask the FastAPI server to gracefully shut down (kills backends via lifespan)
    if _is_running():
        _post("/v1/shutdown", {}, timeout=5.0)
        deadline = time.time() + 10
        while time.time() < deadline:
            if not _is_running():
                break
            time.sleep(0.4)

    # Step 2: kill watchdog so it doesn't restart the server
    pid = _read_pid()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        PID_FILE.unlink(missing_ok=True)

    # Step 3: fallback — kill any orphaned backends the graceful path missed
    time.sleep(1)
    killed = _kill_leftover_backends()
    if killed:
        print(f"Cleaned up {killed} orphaned backend process(es).")

    if _is_running():
        print("Allama did not stop cleanly. Check: allama logs")
    else:
        print("Stopped.")


def cmd_status(args):
    """Show server status and loaded models."""
    health = _get("/health")
    if not health:
        print("● Allama is not running")
        return

    active = health.get("active_servers", 0)
    print(f"● Allama is running  (port {ALLAMA_PORT})")
    print(f"  Loaded models: {active}")

    ps_data = _get("/v1/models")
    if ps_data:
        models = [m["id"] for m in ps_data.get("data", [])]
        if models:
            print(f"  Available ({len(models)}):")
            for m in models:
                print(f"    · {m}")


def cmd_list(args):
    """List available logical models."""
    data = _get("/v1/models")
    if data is None:
        print("Allama is not running. Start with: allama serve")
        return
    models = [m["id"] for m in data.get("data", [])]
    if not models:
        print("No models configured.")
        return
    for m in sorted(models):
        print(m)


def cmd_ps(args):
    """Show currently loaded (running) models."""
    health = _get("/health")
    if not health:
        print("Allama is not running.")
        return
    active = health.get("active_servers", 0)
    if active == 0:
        print("No models loaded.")
    else:
        print(f"{active} model(s) loaded.")


def cmd_logs(args):
    """Tail the Allama log file."""
    if not LOG_FILE.exists():
        print(f"Log file not found: {LOG_FILE}")
        return
    try:
        if args.follow:
            subprocess.run(["tail", "-f", str(LOG_FILE)])
        else:
            lines = args.lines
            subprocess.run(["tail", f"-{lines}", str(LOG_FILE)])
    except KeyboardInterrupt:
        pass


def cmd_ui(args):
    """Open a web UI to chat with Allama models."""
    interface = args.interface.lower()

    # Check if server is running
    if not _is_running():
        print("❌ Allama server is not running. Start with: allama serve")
        sys.exit(1)

    base_url = BASE_URL
    allama_api = f"{base_url}/v1"

    if interface == "openwebui":
        print(f"🌐 Opening OpenWebUI...")
        print(f"   Allama API: {allama_api}")
        print(f"   Configure OpenWebUI to connect to: {allama_api}")

        # Try to open OpenWebUI if installed
        try:
            import webbrowser
            webbrowser.open("http://localhost:8080")
            print("✅ Opened browser at http://localhost:8080")
            print("   (Make sure OpenWebUI is running separately)")
        except Exception as e:
            print(f"⚠️  Could not auto-open browser: {e}")
            print(f"   Visit: http://localhost:8080")

def cmd_run(args):
    """Load a model and open an interactive chat session."""
    model = args.model

    already_running = _is_running()

    label = ["Loading model...  " if already_running else "Starting Allama...  "]
    stop_spinner = threading.Event()
    spinner = threading.Thread(target=_run_spinner, args=(stop_spinner, label), daemon=True)
    spinner.start()

    if not already_running:
        _start_daemon()
        if not _wait_for_server(30):
            stop_spinner.set()
            spinner.join()
            print("Allama failed to start. Check logs: allama logs")
            sys.exit(1)
        label[0] = "Loading model...  "

    # Verify model exists
    data = _get("/v1/models")
    available = [m["id"] for m in (data or {}).get("data", [])]
    if model not in available:
        stop_spinner.set()
        spinner.join()
        print(f"Model '{model}' not found.")
        if available:
            print("Available models:")
            for m in sorted(available):
                print(f"  {m}")
        sys.exit(1)

    # Pre-load model (triggers ensure_physical_model server-side)
    try:
        _post("/v1/load", {"model": model}, timeout=300.0)
    except KeyboardInterrupt:
        stop_spinner.set()
        spinner.join()
        print("\nCancelled.")
        sys.exit(0)

    stop_spinner.set()
    spinner.join()

    _repl(model)


def _print_repl_header(console, model: str):
    from rich import box as _box
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    # Lisa-16 CLI palette (dark terminal compatible)
    C_FG     = "#b0c4d0"
    C_DIM    = "#485e6e"
    C_ACCENT = "#5aaecf"
    C_BORDER = "#1a4a8a"
    C_OK     = "#3a7850"
    W = 68

    body = Table.grid(expand=True, padding=(0, 1))
    body.add_column()

    info = Text()
    info.append("  MODEL  ▸  ", style=f"bold {C_DIM}")
    info.append(model, style=f"bold {C_ACCENT}")
    info.append("    PORT  ▸  ", style=f"bold {C_DIM}")
    info.append(str(ALLAMA_PORT), style=f"bold {C_OK}")
    body.add_row(info)

    cmds = Text()
    cmds.append("  /bye", style=f"bold {C_ACCENT}")
    cmds.append("  ·  ", style=C_DIM)
    cmds.append("/clear", style=f"bold {C_ACCENT}")
    cmds.append("  ·  ", style=C_DIM)
    cmds.append("/exit", style=f"bold {C_ACCENT}")
    cmds.append("    ", style=C_DIM)
    cmds.append("Ctrl+C", style=f"bold {C_FG}")
    cmds.append(" to quit", style=C_DIM)
    body.add_row(cmds)

    console.print()
    console.print(Panel(body, title=f"[bold {C_FG}] ALLAMA RUN [/]",
                        box=_box.DOUBLE, border_style=C_BORDER, padding=(0, 1), width=W))
    console.print()


def _repl(model: str):
    """Interactive chat REPL for a model."""
    try:
        import httpx
    except ImportError:
        print("httpx not installed. Run: pip install httpx")
        sys.exit(1)

    try:
        from rich.console import Console
        _rich_console = Console()
        _use_rich = True
    except ImportError:
        _use_rich = False

    history = []

    if _use_rich:
        _print_repl_header(_rich_console, model)
    else:
        print(f"\n{'─'*50}")
        print(f"  Model : {model}")
        print(f"  Port  : {ALLAMA_PORT}")
        print(f"  /bye or Ctrl+C to exit")
        print(f"{'─'*50}\n")

    try:
        while True:
            try:
                user_input = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("/bye", "/exit", "/quit"):
                print("Bye!")
                break
            if user_input == "/clear":
                history.clear()
                print("History cleared.")
                continue

            history.append({"role": "user", "content": user_input})

            payload = {
                "model": model,
                "messages": history,
                "stream": True,
            }

            spin_label = [""]
            full_response = ""
            stop_spinner = threading.Event()
            spinner = threading.Thread(
                target=_run_spinner,
                args=(stop_spinner, spin_label),
                daemon=True,
            )
            spinner.start()
            try:
                with httpx.Client(timeout=300.0) as client:
                    with client.stream(
                        "POST",
                        f"{BASE_URL}/v1/chat/completions",
                        json=payload,
                        headers={"Authorization": "Bearer dummy"},
                    ) as resp:
                        if resp.status_code != 200:
                            stop_spinner.set()
                            spinner.join()
                            print(f"[error {resp.status_code}]")
                            history.pop()
                            continue
                        first_token = True
                        for line in resp.iter_lines():
                            if not line or line == "data: [DONE]":
                                continue
                            if line.startswith("data: "):
                                try:
                                    chunk = json.loads(line[6:])
                                    delta = chunk["choices"][0]["delta"].get("content", "")
                                    if delta:
                                        if first_token:
                                            stop_spinner.set()
                                            spinner.join()
                                            print()
                                            first_token = False
                                        print(delta, end="", flush=True)
                                        full_response += delta
                                except (json.JSONDecodeError, KeyError):
                                    pass
            except KeyboardInterrupt:
                stop_spinner.set()
                spinner.join()
                print("\n[interrupted]")
                history.pop()
                print()
                continue
            except httpx.ConnectError:
                stop_spinner.set()
                spinner.join()
                print("[connection error — is Allama running?]")
                break
            finally:
                stop_spinner.set()
                spinner.join()

            print("\n")
            if full_response:
                history.append({"role": "assistant", "content": full_response})

    except KeyboardInterrupt:
        print("\nBye!")


_LLAMA_PHRASES_LOADING = [
    "The llama is fluffing its wool; weights are almost ready.",
    "Llama is arranging vectors into neat little pastures.",
    "Alpaca warming up the tensor before the next token.",
    "The model is sipping a byte-sized drink before starting.",
    "Attention heads lined up — all staring across the same field.",
    "Stacking embeddings with pure Andean elegance.",
    "The optimizer is stretching before the climb.",
    "Llama checking if the learning rate feels natural.",
    "Synchronizing dance steps between CPU and GPU — smooth and steady.",
    "The model is taking a deep breath before spreading wisdom.",
    "The royal llama has requested full-precision mode — no shortcuts.",
    "Warming up the GPU... no one touch the lever this time.",
    "A groovy transformation is in progress — please stand by.",
    "Alpaca intern measuring tensor fluffiness per microsecond.",
    "GPU fans spinning gloriously under the Andean sun.",
    "Model alignment supervisor insists the embeddings sparkle.",
    "Tuning the optimizer's dance to royal perfection.",
    "Calibrating attention heads — they must all bow in sync.",
    "A mysterious voice said, 'Pull the other lever!' — ignoring for safety.",
    "Llama chemistry lab mixing float32, dust, and imperial flair.",
    "Thermal sensors report: the GPU is radiating majestic confidence.",
    "Re-polishing tensor cores until they reflect true regal brilliance.",
    "Exporting llama-grade tensors with maximum grace per second.",
    "Embedding room is glowing slightly — must be enchanted gradients.",
    "The royal inspector approved the quantization — reluctantly.",
    "Andean frequency synchronization: stable, slightly dramatic.",
    "VRAM cleanup complete — the field is pristine once more.",
    "One llama muttered something about 'groove restoration.' Proceeding anyway.",
    "Attention mechanism bowing to its emperor — inference in progress.",
    "GPU hum sounds suspiciously like a victory fanfare.",
]

_LLAMA_PHRASES_READY = [
    "The field is green, the model is awake.",
    "Llama ready to spit tokens with ancient wisdom.",
    "Transformers warmed up, Andean insight engaged.",
    "All weights loaded, all llamas assembled.",
    "The Andean oracle has spoken: inference time.",
    "Alpaca clapped — model initialization achieved!",
    "The model woke from its dream; time to generate brilliance.",
    "GPU humming, CPU smiling — we've got a loaded llama!",
    "Everything calibrated — let's graze some tokens.",
    "LLM ready. Achieved full woolly elegance.",
]

_SPINNER_CLOUDS = (
    "     ▁▂▃▂▁▁         "
    "  ▁▂▃▄▃▂▁▁          "
    "       ▁▁▂▃▃▂▁▁     "
    "   ▁▂▂▃▂▂▁▁         "
    "           ▁▂▃▄▃▂▁▁ "
    "     ▁▂▃▂▁▂▃▂▁      "
    "  ▁▁▂▃▂▁▁           "
    "        ▁▂▃▃▂▁▁     "
) * 4

_SPINNER_MOUNTAINS = (
    "▁▁▁▂▄▆█▆▄▂▁▁"
    "▁▁▁▂▃▅▇█▇▅▃▂▁▁"
    "▁▁▂▄▆█▆▄▂▁▁"
    "▁▁▁▁▂▄▆█▆▄▂▁▁▁"
    "▁▁▂▃▄▆█▆▄▃▂▁▁"
    "▁▁▁▂▅▇█▇▅▂▁▁"
    "▁▁▂▃▄▅▇█▇▅▄▃▂▁▁"
    "▁▁▁▂▄▅▇█▇▅▄▂▁▁"
) * 4

_WINDOW = 36

# ── Spinner middle layer (sparse sky between clouds and mountains) ─────────────
_SPINNER_SKY = (
    "                    "
    "   ▁▁               "
    "            ▁       "
    "                ▁▁  "
    "      ▁▁▁           "
    "                    "
    "           ▁▁       "
    "    ▁               "
) * 4

# ── 3-row running llama sprite ────────────────────────────────────────────────
# Llama faces right. Spread diagonally across 3 rows (body left, head upper-right):
#
#   row 1 (clouds):    · · · · · ▒ ▒      ← head
#   row 2 (sky):       · · · · ▓           ← neck
#   row 3 (mountains): ▓ ▒ ▓               ← body + legs
#
_LLAMA_HEAD        = "▄▄"   # 2 chars, placed at P+2..P+3 in clouds row (lower-half block = small head)
_LLAMA_NECK_CH     = "▓"    # 1 char,  placed at P+4       in sky row
_LLAMA_BODY_FRAMES = [       # 3 chars, placed at P..P+2   in mountains row
    "▓▒▓",   # stride neutral
    "▓▓▒",   # right leg sweeps back
    "▓▒▓",   # stride neutral
    "▒▓▓",   # left leg sweeps back
]
_LLAMA_POS   = 4   # body anchor x in the 36-char window
_LLAMA_SPEED = 7   # ticks per animation frame


def _inject_llama(cview: str, sview: str, nview: str, tick: int):
    """Overlay the 3-row running llama onto the three terrain layers."""
    frame = _LLAMA_BODY_FRAMES[(tick // _LLAMA_SPEED) % len(_LLAMA_BODY_FRAMES)]
    cl = list(cview)
    sl = list(sview)
    nl = list(nview)
    # head in clouds row
    for k, ch in enumerate(_LLAMA_HEAD):
        p = _LLAMA_POS + 2 + k
        if p < len(cl):
            cl[p] = ch
    # neck in sky row
    p = _LLAMA_POS + 2
    if p < len(sl):
        sl[p] = _LLAMA_NECK_CH
    # body + legs in mountains row
    for k, ch in enumerate(frame):
        p = _LLAMA_POS + k
        if p < len(nl):
            nl[p] = ch
    return "".join(cl), "".join(sl), "".join(nl)


def _run_llama_spinner(stop_event: threading.Event, phase_ref: list):
    """3-row parallax spinner with running llama and rotating phrases."""
    import random
    phrases = _LLAMA_PHRASES_LOADING[:]
    random.shuffle(phrases)

    phrase_idx = 0
    phrase_ticks = 0
    phrase_interval = 42

    ci = si = ni = 0
    last_c = last_s = last_n = 0
    start = time.time()
    tick = 0

    sys.stdout.write("\n\n")

    while not stop_event.is_set():
        elapsed = time.time() - start
        phrase = phrases[phrase_idx % len(phrases)]

        cview = (_SPINNER_CLOUDS    * 2)[ci % len(_SPINNER_CLOUDS):    ci % len(_SPINNER_CLOUDS)    + _WINDOW]
        sview = (_SPINNER_SKY       * 2)[si % len(_SPINNER_SKY):       si % len(_SPINNER_SKY)       + _WINDOW]
        nview = (_SPINNER_MOUNTAINS * 2)[ni % len(_SPINNER_MOUNTAINS): ni % len(_SPINNER_MOUNTAINS) + _WINDOW]
        cview, sview, nview = _inject_llama(cview, sview, nview, tick)

        cloud_line = f"  {cview}"
        sky_line   = f"  {sview}"
        near_line  = f"  {nview}  {phrase}  {elapsed:.1f}s"

        sys.stdout.write(f"\033[2A\r{' ' * last_c}\r{cloud_line}\n")
        sys.stdout.write(f"\r{' ' * last_s}\r{sky_line}\n")
        sys.stdout.write(f"\r{' ' * last_n}\r{near_line}")
        sys.stdout.flush()

        last_c = len(cloud_line)
        last_s = len(sky_line)
        last_n = len(near_line)

        time.sleep(0.06)
        tick += 1
        if tick % 5 == 0:
            ci += 1
        if tick % 3 == 0:
            si += 1
        if tick % 2 == 0:
            ni += 1
        phrase_ticks += 1
        if phrase_ticks >= phrase_interval:
            phrase_ticks = 0
            phrase_idx += 1

    sys.stdout.write(f"\r{' ' * last_n}\r")
    sys.stdout.write(f"\033[1A\r{' ' * last_s}\r")
    sys.stdout.write(f"\033[1A\r{' ' * last_c}\r")
    sys.stdout.flush()


def _apply_claude_local_fix():
    """Patch ~/.claude/settings.json to disable attribution header (local mode)."""
    import json as _json
    file_path = os.path.expanduser("~/.claude/settings.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data = {}
    if os.path.exists(file_path):
        try:
            with open(file_path) as f:
                data = _json.load(f)
        except Exception:
            pass
    if "env" not in data:
        data["env"] = {}
    if data["env"].get("CLAUDE_CODE_ATTRIBUTION_HEADER") != "0":
        data["env"]["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"
        with open(file_path, "w") as f:
            _json.dump(data, f, indent=2)


def cmd_launch(args):
    """Start allama, load a model, and open Claude Code pointed at it."""
    model = args.model

    stop_spinner = threading.Event()
    phase_ref = ["start"]
    spinner = threading.Thread(
        target=_run_llama_spinner, args=(stop_spinner, phase_ref), daemon=True
    )
    spinner.start()

    # 1 ─ Ensure allama is running
    if not _is_running():
        _start_daemon()
        if not _wait_for_server(45):
            stop_spinner.set()
            spinner.join()
            print("Allama falhou ao iniciar. Veja os logs: allama logs")
            sys.exit(1)

    # 2 ─ Validate model exists
    data = _get("/v1/models")
    available = [m["id"] for m in (data or {}).get("data", [])]
    if model not in available:
        stop_spinner.set()
        spinner.join()
        print(f"Modelo '{model}' não encontrado.")
        if available:
            print("Modelos disponíveis:")
            for m in sorted(available):
                print(f"  · {m}")
        sys.exit(1)

    # 3 ─ Pre-load model
    phase_ref[0] = "model"
    try:
        _post("/v1/load", {"model": model}, timeout=300.0)
    except KeyboardInterrupt:
        stop_spinner.set()
        spinner.join()
        print("\nCancelado.")
        sys.exit(0)

    stop_spinner.set()
    spinner.join()

    import random as _random
    print(f"  ▲  {_random.choice(_LLAMA_PHRASES_READY)}")

    # 4 ─ Patch settings.json for local mode
    _apply_claude_local_fix()

    # 5 ─ Launch claude with model env vars
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{ALLAMA_PORT}"
    env["ANTHROPIC_AUTH_TOKEN"] = "dummy"
    env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
    # Remove any previously set conflicting vars
    for var in ["ANTHROPIC_API_KEY"]:
        env.pop(var, None)

    os.execvpe("claude", ["claude"] + args.claude_args, env)


def cmd_backend_logs(args):
    """Tail the log of the currently running backend process."""
    if not _is_running():
        print("Allama is not running.")
        return

    data = _get("/v1/ps")
    servers = (data or {}).get("servers", [])
    alive = [s for s in servers if s.get("alive")]

    if not alive:
        print("No backend is currently loaded.")
        return

    # Filter by name if provided
    if args.name:
        matches = [s for s in alive if args.name.lower() in s["name"].lower()]
        if not matches:
            print(f"No running backend matching '{args.name}'.")
            print("Running backends:")
            for s in alive:
                print(f"  · {s['name']} ({s['backend']})")
            return
        target = matches[0]
    elif len(alive) == 1:
        target = alive[0]
    else:
        print("Multiple backends running — specify one:")
        for s in alive:
            print(f"  · {s['name']} ({s['backend']})  →  allama backend logs {s['name']}")
        return

    logfile = target.get("logfile", "")
    if not logfile or not Path(logfile).exists():
        print(f"Log file not found for '{target['name']}'.")
        return

    print(f"● {target['name']} ({target['backend']})  —  {logfile}\n")
    try:
        follow = args.follow if hasattr(args, "follow") else True
        if follow:
            subprocess.run(["tail", "-f", logfile])
        else:
            subprocess.run(["tail", f"-{args.lines}", logfile])
    except KeyboardInterrupt:
        pass


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    # Internal: running as watchdog daemon
    if len(sys.argv) > 1 and sys.argv[1] == "__watchdog__":
        verbose = "--verbose" in sys.argv
        _run_watchdog(verbose=verbose)
        return

    parser = argparse.ArgumentParser(
        prog="allama",
        description="Allama — local LLM manager",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # serve
    p_serve = sub.add_parser("serve", help="Start the Allama server")
    p_serve.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Run in foreground with live logs and rich display",
    )
    p_serve.set_defaults(func=cmd_serve)

    # stop
    p_stop = sub.add_parser("stop", help="Stop the Allama server")
    p_stop.set_defaults(func=cmd_stop)

    # status
    p_status = sub.add_parser("status", help="Show server status")
    p_status.set_defaults(func=cmd_status)

    # list
    p_list = sub.add_parser("list", help="List available models")
    p_list.set_defaults(func=cmd_list)

    # ps
    p_ps = sub.add_parser("ps", help="Show loaded models")
    p_ps.set_defaults(func=cmd_ps)

    # logs
    p_logs = sub.add_parser("logs", help="Show Allama logs")
    p_logs.add_argument("-f", "--follow", action="store_true", help="Follow log output")
    p_logs.add_argument("-n", "--lines", type=int, default=50, metavar="N", help="Lines to show (default: 50)")
    p_logs.set_defaults(func=cmd_logs)

    # run
    p_run = sub.add_parser("run", help="Chat with a model interactively")
    p_run.add_argument("model", help="Model name (e.g. 'Qwen3.5:27b')")
    p_run.set_defaults(func=cmd_run)

    # ui
    p_ui = sub.add_parser("ui", help="Open web UI to chat with models")
    p_ui.add_argument("interface", nargs="?", default="openwebui",
                      choices=["openwebui"],
                      help="Web interface to open (default: openwebui)")
    p_ui.set_defaults(func=cmd_ui)

    # launch
    p_launch = sub.add_parser("launch", help="Launch an AI client with a local model")
    launch_sub = p_launch.add_subparsers(dest="launch_target", metavar="<client>")
    launch_sub.required = True

    p_lc = launch_sub.add_parser("claude", help="Open Claude Code with a local model")
    p_lc.add_argument("model", help="Logical model name (e.g. 'Qwen3.5:27b-Code')")
    p_lc.add_argument("claude_args", nargs=argparse.REMAINDER,
                      help="Extra arguments forwarded to claude")
    p_lc.set_defaults(func=cmd_launch)

    # backend
    p_backend = sub.add_parser("backend", help="Backend server commands")
    backend_sub = p_backend.add_subparsers(dest="backend_command", metavar="<subcommand>")
    backend_sub.required = True

    p_bl = backend_sub.add_parser("logs", help="Tail the log of the running backend")
    p_bl.add_argument("-f", "--follow", action="store_true", help="Follow log output (default: true)")
    p_bl.add_argument("-n", "--lines", type=int, default=50, metavar="N", help="Lines to show (default: 50)")
    p_bl.add_argument("name", nargs="?", default=None, help="Backend name (optional if only one is running)")
    p_bl.set_defaults(func=cmd_backend_logs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
