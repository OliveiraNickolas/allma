#!/usr/bin/env python3
"""
Allama CLI — allama serve / run / list / ps / stop / logs / launch
"""
import argparse
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────────
# Queue to communicate which models are being loaded (for -bv terminal opening)
_models_loading_queue = queue.Queue()

# ── Constants ──────────────────────────────────────────────────────────────────
ALLAMA_DIR  = Path(__file__).parent
ALLAMA_PORT = int(os.environ.get("ALLAMA_PORT", "9000"))
BASE_URL    = f"http://127.0.0.1:{ALLAMA_PORT}"
PID_FILE    = Path(os.environ.get("ALLAMA_PID_FILE", "/tmp/allama_watchdog.pid"))
LOG_FILE    = ALLAMA_DIR / "logs" / "allama.log"
_VENV_PYTHON = ALLAMA_DIR / "venv" / "bin" / "python"
PYTHON      = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable
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


def _limit_line_width(spinner_part: str, content: str, time_part: str, term_width: int = 80) -> str:
    """Ensure a line doesn't exceed terminal width with spinner + content + time."""
    # Calculate available space for content
    min_width = len(spinner_part) + len(time_part)
    available = max(term_width - min_width - 2, 20)  # Leave 2 char margin, min 20

    # Truncate content if too long
    if len(content) > available:
        content = content[:available-3] + "..."

    line = f"{spinner_part}{content}{time_part}"

    # Final safety check: don't exceed terminal width
    if len(line) > term_width - 1:
        line = line[:term_width-1]

    return line


def _run_simple_spinner(stop_event: threading.Event, label_ref: list):
    """Single-line braille spinner for allama run."""
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    start = time.time()
    i = 0
    last_len = 0
    while not stop_event.is_set():
        elapsed = time.time() - start
        frame = frames[i % len(frames)]
        label = label_ref[0]
        line = f"  {frame}  {label}  [{elapsed:.1f}s]"
        sys.stdout.write(f"\r{' ' * last_len}\r{line}")
        sys.stdout.flush()
        last_len = len(line)
        time.sleep(0.08)
        i += 1
    sys.stdout.write(f"\r{' ' * last_len}\r")
    sys.stdout.flush()


def _run_spinner(stop_event: threading.Event, label_ref: list):
    """3-row parallax Andes spinner with running llama."""
    import shutil

    ci = si = ni = 0
    last_c = last_s = last_n = 0
    start = time.time()
    tick = 0

    sys.stdout.write("\n\n")  # reserve 3 lines

    while not stop_event.is_set():
        # Get terminal width dynamically (detects window resize)
        try:
            term_width = shutil.get_terminal_size().columns
        except Exception:
            term_width = 80

        elapsed = time.time() - start
        cview = (_SPINNER_CLOUDS    * 2)[ci % len(_SPINNER_CLOUDS):    ci % len(_SPINNER_CLOUDS)    + _WINDOW]
        sview = (_SPINNER_SKY       * 2)[si % len(_SPINNER_SKY):       si % len(_SPINNER_SKY)       + _WINDOW]
        nview = (_SPINNER_MOUNTAINS * 2)[ni % len(_SPINNER_MOUNTAINS): ni % len(_SPINNER_MOUNTAINS) + _WINDOW]
        cview, sview, nview = _inject_llama(cview, sview, nview, tick)

        cloud_line = f"  {cview}"
        sky_line   = f"  {sview}"
        spinner_part = f"  {nview}  "
        time_part = f"{elapsed:.1f}s"
        label = label_ref[0]

        near_line = _limit_line_width(spinner_part, label, time_part, term_width)

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
    """Simple HTTP POST, returns parsed JSON or None. Raises RuntimeError on HTTP errors."""
    try:
        import urllib.request as _ur
        import urllib.error as _ue
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
    except _ue.HTTPError as e:
        try:
            body_bytes = e.read()
            err_data = json.loads(body_bytes)
            detail = err_data.get("detail") or err_data.get("error") or str(err_data)
        except Exception:
            detail = e.reason
        raise RuntimeError(f"HTTP {e.code}: {detail}") from None
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


def _kill_port_user(port: int = 9000) -> bool:
    """Kill any process occupying the given port. Returns True if killed, False otherwise."""
    import subprocess
    import os

    killed = False

    # Try lsof first (most reliable)
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        pids = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]
        for pid in pids:
            try:
                os.kill(int(pid), 9)  # SIGKILL
                logger.info(f"Killed process {pid} on port {port}")
                killed = True
            except (ValueError, ProcessLookupError, OSError) as e:
                logger.debug(f"Failed to kill {pid}: {e}")
    except FileNotFoundError:
        # lsof not available, try fuser
        try:
            subprocess.run(
                ["fuser", "-k", "-9", f"{port}/tcp"],
                capture_output=True,
                timeout=5
            )
            logger.info(f"Killed process on port {port} using fuser")
            killed = True
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"lsof failed: {e}")

    return killed


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
    fast_fail_count = 0
    FAST_FAIL_SECS = 10   # crash within this many seconds = fast fail
    MAX_FAST_FAILS = 3    # abort after this many consecutive fast fails

    while True:
        start_time = time.monotonic()
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
        elapsed = time.monotonic() - start_time

        # Clean shutdown (SIGINT / ctrl+c forwarded) — don't restart
        if code in (0, -signal.SIGINT, 130):
            break

        # Circuit breaker: too many fast crashes in a row → likely a startup error
        if elapsed < FAST_FAIL_SECS:
            fast_fail_count += 1
        else:
            fast_fail_count = 0  # ran long enough, reset counter

        if fast_fail_count >= MAX_FAST_FAILS:
            msg = (
                f"[allama] server crashed {fast_fail_count} times in under {FAST_FAIL_SECS}s "
                f"(exit code {code}) — aborting. Check logs: {LOG_FILE}"
            )
            if verbose:
                print(msg, flush=True)
            else:
                with open(LOG_FILE, "a") as log:
                    log.write(msg + "\n")
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
def _open_terminal_tail(logfile: str):
    """Open a new terminal window running tail -f on a backend log file."""
    title = Path(logfile).stem + " · backend log"
    variants = [
        ["gnome-terminal", f"--title={title}", "--", "tail", "-f", logfile],
        ["kitty", "--title", title, "tail", "-f", logfile],
        ["konsole", "--title", title, "-e", "tail", "-f", logfile],
        ["xfce4-terminal", f"--title={title}", "-e", f"tail -f {logfile}"],
        ["xterm", "-title", title, "-e", f"tail -f {logfile}"],
    ]
    for cmd in variants:
        try:
            subprocess.Popen(cmd, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Write to a marker file instead of stderr (daemon redirects stderr)
            marker_file = Path(logfile).parent / ".terminal-opened"
            with open(marker_file, "a") as m:
                m.write(f"{Path(logfile).name}\n")
            return
        except FileNotFoundError:
            continue
        except Exception as e:
            continue


def _be_verbose_watcher(log_dir: Path, stop_event: threading.Event, models_queue: queue.Queue):
    """Watch allama.log for model loading messages and open terminals for each model."""
    import re

    allama_log = log_dir / "allama.log"
    opened_models: set[str] = set()
    log_position = None  # Will be set to end-of-file on first read

    # Regex to remove ANSI color/control codes and other junk
    ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\r|[^\x20-\x7E\n]')

    while not stop_event.is_set():
        try:
            if not allama_log.exists():
                stop_event.wait(0.5)
                continue

            with open(allama_log, "r", errors="replace") as f:
                # On first read, jump to end of file (ignore past logs)
                if log_position is None:
                    f.seek(0, 2)  # 0, 2 = seek to end
                    log_position = f.tell()
                    continue  # Skip first iteration, just initialize position

                f.seek(log_position)
                new_content = f.read()
                log_position = f.tell()

            # Clean everything except ASCII printable + newline
            clean_content = ansi_escape.sub('', new_content)

            # Look for "Loading <model>" — split by lines and search each line
            for line in clean_content.split('\n'):
                if 'Loading' in line:
                    # Extract model name: "Loading Qwen3.5:9b" → "Qwen3.5:9b"
                    parts = line.split('Loading')
                    if len(parts) > 1:
                        # Get everything after "Loading", strip whitespace, take first token
                        after_loading = parts[1].strip().split()[0] if parts[1].strip() else None
                        if after_loading:
                            model_name = after_loading

                            if model_name not in opened_models:
                                opened_models.add(model_name)

                                # Find the corresponding log file
                                model_for_file = model_name.replace(":", "-")
                                logfile = log_dir / f"{model_for_file}.log"

                                if not logfile.exists():
                                    for candidate in log_dir.glob(f"*{model_for_file}*.log"):
                                        logfile = candidate
                                        break

                                if logfile.exists():
                                    _open_terminal_tail(str(logfile))
        except Exception:
            pass
        stop_event.wait(0.5)


def cmd_serve(args):
    """Start the Allama daemon."""
    be_verbose = getattr(args, "be_verbose", False)
    force = getattr(args, "force", False)
    be_verbose_stop = threading.Event()

    # Kill any process on port 9000 if --force is used
    if force:
        if _kill_port_user(ALLAMA_PORT):
            print(f"Killed process on port {ALLAMA_PORT}")
            time.sleep(0.5)  # Brief pause to let port release

    if be_verbose:
        watcher = threading.Thread(
            target=_be_verbose_watcher,
            args=(ALLAMA_DIR / "logs", be_verbose_stop, _models_loading_queue),
            daemon=True,
        )
        watcher.start()

    try:
        if args.verbose:
            # --verbose: foreground with server logs (runs allama.py directly, not as daemon)
            print("Starting Allama (verbose mode — Ctrl+C to stop)...")
            _run_watchdog(verbose=True)
        elif be_verbose:
            # --be-verbose only: silent daemon mode but keep watcher alive
            if _is_running():
                print("Allama is already running.")
                return
            label = ["Starting Allama with backend logs...  "]
            stop_spinner = threading.Event()
            spinner = threading.Thread(target=_run_spinner, args=(stop_spinner, label), daemon=True)
            spinner.start()
            _start_daemon()
            ok = _wait_for_server(30)
            stop_spinner.set()
            spinner.join()
            if ok:
                print("Allama ready. Backend terminals will open automatically.")
            else:
                print("Allama timed out. Check logs: allama logs")
            # Don't wait — let the watcher thread continue in background
        else:
            # Background daemon mode (no verbose, no be_verbose)
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
    finally:
        be_verbose_stop.set()


def cmd_restart(args):
    """Stop and restart the Allama server."""
    cmd_stop(args)
    time.sleep(1)
    # Clear the "already running" guard so cmd_serve proceeds
    if hasattr(args, "_restarting"):
        pass
    cmd_serve(args)


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


def cmd_run(args):
    """Load a model and open an interactive chat session."""
    model = args.model

    already_running = _is_running()

    label = ["Loading model..." if already_running else "Starting Allama..."]
    stop_spinner = threading.Event()
    spinner = threading.Thread(target=_run_simple_spinner, args=(stop_spinner, label), daemon=True)
    spinner.start()

    if not already_running:
        _start_daemon()
        if not _wait_for_server(30):
            stop_spinner.set()
            spinner.join()
            print("Allama failed to start. Check logs: allama logs")
            sys.exit(1)
        label[0] = "Loading model..."

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
        # Notify the -bv watcher which model we're loading
        _models_loading_queue.put(model)
        load_data = {"model": model}
        if args.gpu is not None:
            load_data["gpu_id"] = args.gpu
        _post("/v1/load", load_data, timeout=300.0)
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
    from rich.align import Align
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    import shutil as _shutil
    # Same warm cream / teal palette as the main server banner
    C_BG     = "#e8dfc8"
    C_SCREEN = "#d0c4a8"
    C_FG     = "#1a1408"
    C_DIM    = "#6a5a48"
    C_ACCENT = "#007878"
    C_BORDER = "#008888"
    W = min(max(_shutil.get_terminal_size().columns - 4, 40), 90)
    _S = f"on {C_BG}"

    def section(name: str) -> Text:
        t = Text()
        t.append("[ ", style=C_DIM)
        t.append(name, style=f"bold {C_ACCENT}")
        t.append(" ]", style=C_DIM)
        return t

    # Session info row
    session_tbl = Table.grid(expand=True, padding=(0, 1))
    session_tbl.add_column()
    info = Text(style=_S)
    info.append("  model  ", style=f"{C_DIM} on {C_BG}")
    info.append("▸  ", style=f"{C_DIM} on {C_BG}")
    info.append(model, style=f"bold {C_ACCENT} on {C_BG}")
    info.append("   port  ", style=f"{C_DIM} on {C_BG}")
    info.append("▸  ", style=f"{C_DIM} on {C_BG}")
    info.append(str(ALLAMA_PORT), style=f"bold {C_ACCENT} on {C_BG}")
    session_tbl.add_row(info)

    session_panel = Panel(
        session_tbl,
        title=section("Session"),
        title_align="left",
        box=_box.SQUARE,
        border_style=C_DIM,
        style=_S,
        padding=(0, 1),
    )

    # Commands row
    cmds_tbl = Table.grid(expand=True, padding=(0, 1))
    cmds_tbl.add_column()
    cmds = Text(style=_S)
    cmds.append("  ", style=f"on {C_BG}")
    for i, cmd in enumerate(["/exit", "/clear"]):
        cmds.append(cmd, style=f"bold {C_ACCENT} on {C_BG}")
        cmds.append("  ·  ", style=f"{C_DIM} on {C_BG}")
    cmds.append("Ctrl+C", style=f"bold {C_FG} on {C_BG}")
    cmds.append(" to quit", style=f"{C_DIM} on {C_BG}")
    cmds_tbl.add_row(cmds)

    cmds_panel = Panel(
        cmds_tbl,
        title=section("Commands"),
        title_align="left",
        box=_box.SQUARE,
        border_style=C_DIM,
        style=_S,
        padding=(0, 1),
    )

    body = Group(session_panel, cmds_panel)
    main_win = Panel(
        body,
        box=_box.DOUBLE,
        border_style=C_BORDER,
        style=f"on {C_SCREEN}",
        padding=(0, 0),
        width=W,
    )

    console.print()
    console.print(main_win)
    console.print()


def _repl(model: str):
    """Interactive chat REPL for a model."""
    try:
        import httpx
        import logging as _logging
        _logging.getLogger("httpx").setLevel(_logging.WARNING)
        _logging.getLogger("httpcore").setLevel(_logging.WARNING)
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

            spin_label = ["thinking..."]
            full_response = ""
            stop_spinner = threading.Event()
            spinner = threading.Thread(
                target=_run_simple_spinner,
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
    import shutil

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
        # Get terminal width dynamically (detects window resize)
        try:
            term_width = shutil.get_terminal_size().columns
        except Exception:
            term_width = 80

        elapsed = time.time() - start
        phrase = phrases[phrase_idx % len(phrases)]

        cview = (_SPINNER_CLOUDS    * 2)[ci % len(_SPINNER_CLOUDS):    ci % len(_SPINNER_CLOUDS)    + _WINDOW]
        sview = (_SPINNER_SKY       * 2)[si % len(_SPINNER_SKY):       si % len(_SPINNER_SKY)       + _WINDOW]
        nview = (_SPINNER_MOUNTAINS * 2)[ni % len(_SPINNER_MOUNTAINS): ni % len(_SPINNER_MOUNTAINS) + _WINDOW]
        cview, sview, nview = _inject_llama(cview, sview, nview, tick)

        cloud_line = f"  {cview}"
        sky_line   = f"  {sview}"
        spinner_part = f"  {nview}  "
        time_part = f"{elapsed:.1f}s"

        near_line = _limit_line_width(spinner_part, phrase, time_part, term_width)

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
            print("Allama failed to start. Check logs: allama logs")
            sys.exit(1)

    # 2 ─ Validate model exists
    data = _get("/v1/models")
    available = [m["id"] for m in (data or {}).get("data", [])]
    if model not in available:
        stop_spinner.set()
        spinner.join()
        print(f"Model '{model}' not found.")
        if available:
            print("Available models:")
            for m in sorted(available):
                print(f"  · {m}")
        sys.exit(1)

    # 3 ─ Pre-load model
    phase_ref[0] = "model"
    load_error = None
    try:
        payload = {"model": model}
        if args.gpu is not None:
            payload["gpu_id"] = args.gpu
        result = _post("/v1/load", payload, timeout=300.0)
    except KeyboardInterrupt:
        stop_spinner.set()
        spinner.join()
        print("\nCancelled.")
        sys.exit(0)
    except RuntimeError as e:
        load_error = str(e)
        result = None

    stop_spinner.set()
    spinner.join()

    if load_error:
        print(f"Failed to load model '{model}': {load_error}")
        print("Check logs: allama backend logs")
        sys.exit(1)

    if not result or result.get("status") != "loaded":
        print(f"Failed to load model '{model}': unexpected server response.")
        print("Check logs: allama backend logs")
        sys.exit(1)

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


def cmd_hardware_detect(args):
    """Force hardware re-detection and show results."""
    if _is_running():
        print("Getting hardware info from running server...")
        hw_data = _get("/v1/hardware")
        if not hw_data:
            print("Failed to get hardware info from server.")
            return

        profile = hw_data.get("profile")
        if profile:
            print("🔍 Hardware Profile:")
            print(f"   Driver: {profile.get('driver_version', 'unknown')}")
            print(f"   CUDA: {profile.get('cuda_version', 'unknown')}")
            print(f"   Total VRAM: {profile.get('total_vram_gb', 0):.1f}GB")
            print(f"   Available: {profile.get('available_vram_gb', 0):.1f}GB")
            print(f"   Max contiguous: {profile.get('max_contiguous_gb', 0):.1f}GB")
            print(f"\n   GPUs:")
            for gpu in profile.get("gpus", []):
                print(
                    f"     {gpu['index']}: {gpu['name']} (compute {gpu['compute_capability']}) "
                    f"— {gpu['total_memory_gb']:.1f}GB"
                )
        else:
            print("No hardware profile detected.")
    else:
        print("Allama server is not running.")
        print("Start with: allama serve")


def cmd_calibrate(args):
    """Pre-calibrate a model without loading it."""
    import asyncio

    sys.path.insert(0, str(ALLAMA_DIR))
    from core.bootstrap import BootstrapDetector
    from core.config import PHYSICAL_MODELS
    from core.gpu import get_model_vram_need

    if args.model not in PHYSICAL_MODELS:
        print(f"❌ Model '{args.model}' not found in configs.")
        print("Available physical models:")
        for name in sorted(PHYSICAL_MODELS.keys()):
            print(f"   · {name}")
        return

    print(f"🔍 Detecting hardware...")
    try:
        profile = asyncio.run(BootstrapDetector.detect_hardware())
    except Exception as e:
        print(f"❌ Hardware detection failed: {e}")
        return

    cfg = PHYSICAL_MODELS[args.model]
    model_size_gb = get_model_vram_need(cfg, args.model)

    print(f"📊 Calibrating {args.model}...")
    try:
        calib = asyncio.run(
            BootstrapDetector.calibrate_for_model(
                physical_name=args.model,
                model_size_gb=model_size_gb,
                hardware_profile=profile,
                config=cfg,
            )
        )

        print(f"✅ Calibration Result:")
        print(f"   Model: {args.model}")
        print(f"   Size: {model_size_gb:.1f}GB (estimated)")
        print(f"   Backend: {calib.backend}")
        print(f"   TP: {calib.recommended_tp}")
        print(f"   ubatch-size: {calib.recommended_ubatch_size}")
        print(f"   n_batch: {calib.recommended_n_batch}")
        print(f"   n_ctx: {calib.recommended_n_ctx}")
        print(f"   cache-dtype: {calib.recommended_cache_dtype}")
        print(f"   Confidence: {calib.confidence}")
        print(f"   Est. VRAM need: {calib.estimated_vram_need_gb:.1f}GB")

        if calib.warnings:
            print(f"\n   Warnings:")
            for warn in calib.warnings:
                print(f"      ⚠️  {warn}")

    except Exception as e:
        print(f"❌ Calibration failed: {e}")


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
    p_serve.add_argument(
        "--be-verbose", "-bv",
        action="store_true",
        dest="be_verbose",
        help="Open a new terminal window tailing each backend log as it loads",
    )
    p_serve.add_argument(
        "--force", "-f",
        action="store_true",
        help="Kill any process occupying port 9000 before starting",
    )
    p_serve.set_defaults(func=cmd_serve)

    # restart
    p_restart = sub.add_parser("restart", help="Stop and restart the Allama server")
    p_restart.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Restart in foreground verbose mode",
    )
    p_restart.add_argument(
        "--be-verbose", "-bv",
        action="store_true",
        dest="be_verbose",
        help="Open a new terminal window tailing each backend log as it loads",
    )
    p_restart.set_defaults(func=cmd_restart)

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
    p_run.add_argument("--gpu", type=int, default=None, metavar="N",
                      help="Pin model to GPU N (0-based). If not specified, auto-select by available VRAM")
    p_run.set_defaults(func=cmd_run)

    # launch
    p_launch = sub.add_parser("launch", help="Launch an AI client with a local model")
    launch_sub = p_launch.add_subparsers(dest="launch_target", metavar="<client>")
    launch_sub.required = True

    p_lc = launch_sub.add_parser("claude", help="Open Claude Code with a local model")
    p_lc.add_argument("model", help="Logical model name (e.g. 'Qwen3.5:27b-Code')")
    p_lc.add_argument("--gpu", type=int, default=None, metavar="ID",
                      help="Force model onto a specific GPU (0-indexed)")
    p_lc.add_argument("claude_args", nargs=argparse.REMAINDER,
                      help="Extra arguments forwarded to claude")
    p_lc.set_defaults(func=cmd_launch)

    # hardware-detect
    p_hw = sub.add_parser("hardware-detect", help="Detect hardware and show info")
    p_hw.set_defaults(func=cmd_hardware_detect)

    # calibrate
    p_calib = sub.add_parser("calibrate", help="Pre-calibrate a model")
    p_calib.add_argument("model", help="Physical model name (e.g. 'Qwen3.5-27b')")
    p_calib.set_defaults(func=cmd_calibrate)

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
