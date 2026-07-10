#!/usr/bin/env python3
"""
Allma CLI — allma serve / run / list / ps / stop / logs / launch
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
ALLMA_DIR  = Path(__file__).parent

# Load .env before reading any env vars so CLI and server always agree on ports
def _load_dotenv_cli():
    env_file = ALLMA_DIR / ".env"
    if not env_file.exists():
        return
    try:
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            # Strip inline comment (but not inside quotes)
            if val and not (val.startswith('"') or val.startswith("'")):
                val = val.split("#")[0]
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass

_load_dotenv_cli()

# Ensure core/ is importable regardless of CWD (needed for lazy imports inside functions)
if str(ALLMA_DIR) not in sys.path:
    sys.path.insert(0, str(ALLMA_DIR))

ALLMA_PORT = int(os.environ.get("ALLMA_PORT", "9000"))
BASE_URL    = f"http://127.0.0.1:{ALLMA_PORT}"
PID_FILE    = Path(os.environ.get("ALLMA_PID_FILE", "/tmp/allma_watchdog.pid"))
LOG_FILE    = ALLMA_DIR / "logs" / "allma.log"
_VENV_PYTHON = ALLMA_DIR / "venv" / "bin" / "python"
PYTHON      = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable
SERVER_SCRIPT = str(ALLMA_DIR / "allma.py")

# Re-exec under the project venv so `allma` works from anywhere without
# activating it manually (textual/rich/httpx live inside the venv).
# sys.prefix check avoids exec loops: inside the venv it equals the venv dir.
_VENV_DIR = ALLMA_DIR / "venv"
if _VENV_PYTHON.exists() and Path(sys.prefix).resolve() != _VENV_DIR.resolve():
    os.execv(str(_VENV_PYTHON),
             [str(_VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])

# Warm cream / teal palette shared by the REPL header, the model switcher and
# the backend-log banner (same scheme as the main server banner)
C_BG     = "#e8dfc8"
C_SCREEN = "#d0c4a8"
C_FG     = "#1a1408"
C_DIM    = "#6a5a48"
C_ACCENT = "#007878"
C_BORDER = "#008888"

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
    """Single-line braille spinner for allma run."""
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
    Loop forever restarting allma.py if it dies.
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
                f"[allma] server crashed {fast_fail_count} times in under {FAST_FAIL_SECS}s "
                f"(exit code {code}) — aborting. Check logs: {LOG_FILE}"
            )
            if verbose:
                print(msg, flush=True)
            else:
                with open(LOG_FILE, "a") as log:
                    log.write(msg + "\n")
            break

        restart_count += 1
        msg = f"[allma] process exited (code {code}), restarting in 3s... (#{restart_count})"
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


def _be_verbose_watcher(log_dir: Path, stop_event: threading.Event):
    """Watch for new or freshly-written backend log files and open a terminal for each."""
    opened: set[Path] = set()
    # Snapshot existing files + mtimes so we only react to changes from this point
    try:
        existing_mtimes = {
            p: p.stat().st_mtime
            for p in log_dir.glob("*.log")
            if p.name != "allma.log"
        }
    except Exception:
        existing_mtimes = {}

    while not stop_event.is_set():
        try:
            for logfile in log_dir.glob("*.log"):
                if logfile.name == "allma.log" or logfile in opened:
                    continue
                try:
                    mtime = logfile.stat().st_mtime
                except FileNotFoundError:
                    continue
                prev_mtime = existing_mtimes.get(logfile)
                if prev_mtime is None or mtime > prev_mtime:
                    opened.add(logfile)
                    _open_terminal_tail(str(logfile))
        except Exception:
            pass
        stop_event.wait(0.5)


def cmd_serve(args):
    """Start the Allma daemon."""
    be_verbose = getattr(args, "be_verbose", False)
    force = getattr(args, "force", False)
    be_verbose_stop = threading.Event()

    # Kill any process on port 9000 if --force is used
    if force:
        if _kill_port_user(ALLMA_PORT):
            print(f"Killed process on port {ALLMA_PORT}")
            time.sleep(0.5)  # Brief pause to let port release

    if be_verbose:
        watcher = threading.Thread(
            target=_be_verbose_watcher,
            args=(ALLMA_DIR / "logs", be_verbose_stop),
            daemon=True,
        )
        watcher.start()

    try:
        if args.verbose:
            # --verbose: foreground with server logs (runs allma.py directly, not as daemon)
            print("Starting Allma (verbose mode — Ctrl+C to stop)...")
            _run_watchdog(verbose=True)
        elif be_verbose:
            # --be-verbose only: start daemon, then block watching for backend logs
            if _is_running():
                print("Allma is already running.")
            else:
                label = ["Starting Allma with backend logs...  "]
                stop_spinner = threading.Event()
                spinner = threading.Thread(target=_run_spinner, args=(stop_spinner, label), daemon=True)
                spinner.start()
                _start_daemon()
                ok = _wait_for_server(30)
                stop_spinner.set()
                spinner.join()
                if not ok:
                    print("Allma timed out. Check logs: allma logs")
                    return
            print("Watching for backend loads — Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            # Background daemon mode (no verbose, no be_verbose)
            if _is_running():
                print("Allma is already running.")
                return
            label = ["Starting Allma...  "]
            stop_spinner = threading.Event()
            spinner = threading.Thread(target=_run_spinner, args=(stop_spinner, label), daemon=True)
            spinner.start()
            _start_daemon()
            ok = _wait_for_server(30)
            stop_spinner.set()
            spinner.join()
            if ok:
                print("Allma ready.")
            else:
                print("Allma timed out. Check logs: allma logs")
    finally:
        be_verbose_stop.set()


def cmd_restart(args):
    """Stop and restart the Allma server."""
    cmd_stop(args)
    time.sleep(1)
    cmd_serve(args)


def _kill_leftover_backends() -> int:
    """Kill any allma-managed vllm/llama-server processes still alive.

    Uses command-line fingerprints specific to allma-spawned backends
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
                is_llama = ("llama-server" in cmd or "llama_cpp.server" in cmd) and "--host 127.0.0.1" in cmd
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
    """Stop the Allma daemon and all backend processes."""
    if not _is_running() and not _read_pid():
        print("Allma is not running.")
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
        print("Allma did not stop cleanly. Check: allma logs")
    else:
        print("Stopped.")


def cmd_status(args):
    """Show server status and loaded models."""
    health = _get("/health")
    if not health:
        print("● Allma is not running")
        return

    active = health.get("active_servers", 0)
    print(f"● Allma is running  (port {ALLMA_PORT})")
    print(f"  Loaded models: {active}")

    ps_data = _get("/v1/models")
    if ps_data:
        models = [m["id"] for m in ps_data.get("data", [])]
        if models:
            print(f"  Available ({len(models)}):")
            for m in models:
                print(f"    · {m}")


def cmd_list(args):
    """List available profile models."""
    data = _get("/v1/models")
    if data is None:
        print("Allma is not running. Start with: allma serve")
        return
    models = [m["id"] for m in data.get("data", [])]
    if not models:
        print("No models configured.")
        return
    for m in sorted(models):
        print(m)


def cmd_ps(args):
    """Show currently loaded (running) models and recent crash errors."""
    if not _is_running():
        print("Allma is not running.")
        return
    data = _get("/v1/ps")
    if data is None:
        print("Failed to reach Allma server.")
        return
    servers = data.get("servers", [])
    errors = data.get("errors", {})
    if not servers and not errors:
        print("No models loaded.")
        return
    for s in servers:
        status = "● running" if s.get("alive") else "✗ dead"
        print(f"{status}  {s['name']}  (pid {s.get('pid')}, port {s.get('port')}, {s.get('backend')})")
    if errors:
        for name, err in errors.items():
            if not any(s["name"] == name and s.get("alive") for s in servers):
                print(f"\n✗ {name} crashed — {err['error_type']}")
                print(f"  {err['explanation']}")
                for suggestion in err.get("suggestions", []):
                    print(f"  → {suggestion}")


def cmd_unload(args):
    """Unload a running model immediately, freeing its VRAM."""
    if not _is_running():
        print("Allma is not running.")
        return
    model = args.model
    resp = _post("/v1/unload", {"model": model})
    if resp is None:
        print(f"Failed to reach Allma server.")
        return
    if "error" in resp:
        print(f"Error: {resp['error']}")
        loaded = resp.get("loaded", [])
        if loaded:
            print(f"Loaded base models: {', '.join(loaded)}")
        return
    print(f"Unloaded: {resp.get('model', model)}")


def cmd_reload(args):
    """Unload a running model then load it again — no server restart."""
    if not _is_running():
        print("Allma is not running.")
        return
    model = args.model
    resp = _post("/v1/unload", {"model": model})
    if resp is None:
        print("Failed to reach Allma server.")
        return
    if "error" in resp and "not loaded" not in resp["error"]:
        print(f"Unload error: {resp['error']}")
        return
    print(f"Unloaded: {model}. Reloading...")
    resp = _post("/v1/load", {"model": model})
    if resp is None:
        print("Failed to reach Allma server on reload.")
        return
    if "error" in resp:
        print(f"Load error: {resp['error']}")
        return
    print(f"Reloaded: {resp.get('model', model)}")


def cmd_logs(args):
    """Tail the Allma log file."""
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


def cmd_quickstart(args):
    """Guided first-run: pick a goal, pick a curated model that fits the
    hardware, then download + configure + chat via the normal run flow."""
    from core.quickstart import run_quickstart
    repo = run_quickstart()
    if not repo:
        return
    args.model = repo
    cmd_run(args)


def cmd_run(args):
    """Load a model and open an interactive chat session.

    Also accepts a HuggingFace repo id or URL: downloads the model
    (interactive quant picker with fit/recommendation preview), generates
    configs, and runs the resulting profile in one flow."""
    model = args.model

    # Profile names never contain "/" — a slash (or an HF URL) means the user
    # pasted a repo straight from HuggingFace.
    if "/" in model or "huggingface.co" in model:
        from core.downloader import run_download
        profile = run_download(model)
        if not profile:
            print("Download finished but no profile was generated — cannot run.")
            return
        # A running server loaded its configs at startup; pick up the new ones.
        if _is_running():
            _post("/v1/reload-configs", {})
        model = profile
        print(f"Starting chat with {model}...")

    already_running = _is_running()

    label = ["Loading model..." if already_running else "Starting Allma..."]
    stop_spinner = threading.Event()
    spinner = threading.Thread(target=_run_simple_spinner, args=(stop_spinner, label), daemon=True)
    spinner.start()

    if not already_running:
        _start_daemon()
        if not _wait_for_server(30):
            stop_spinner.set()
            spinner.join()
            print("Allma failed to start. Check logs: allma logs")
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

    # Pre-load model (triggers ensure_base_model server-side)
    be_verbose = getattr(args, "be_verbose", False)
    bv_stop = threading.Event()
    if be_verbose:
        bv_thread = threading.Thread(
            target=_be_verbose_watcher,
            args=(ALLMA_DIR / "logs", bv_stop),
            daemon=True,
        )
        bv_thread.start()

    load_error = None
    try:
        load_data = {"model": model}
        if args.gpu is not None:
            load_data["gpu_id"] = args.gpu
        result = _post("/v1/load", load_data, timeout=300.0)
        if result is None:
            load_error = "Server did not respond — model may have failed to start."
        elif "error" in result:
            load_error = result["error"]
    except KeyboardInterrupt:
        bv_stop.set()
        stop_spinner.set()
        spinner.join()
        print("\nCancelled.")
        sys.exit(0)
    finally:
        bv_stop.set()

    stop_spinner.set()
    spinner.join()

    if load_error:
        print(f"\n✗ Failed to load '{model}': {load_error}")
        print("  Run 'allma logs' to see what went wrong.")
        sys.exit(1)

    _repl(model)


def _print_repl_header(console, model: str):
    from rich import box as _box
    from rich.align import Align
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    import shutil as _shutil
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
    info.append(str(ALLMA_PORT), style=f"bold {C_ACCENT} on {C_BG}")
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
    for i, cmd in enumerate(["/exit", "/clear", "/clean", "/model"]):
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


def _repl_switch_model(current_model: str, console) -> str | None:
    """Show a picker of profile models sharing the same base backend. Returns new model name or None."""
    from core.config import PROFILE_MODELS

    current_base = PROFILE_MODELS.get(current_model, {}).get("base")
    if not current_base:
        print(f"  [model '{current_model}' not found in config]")
        return None

    siblings = sorted(
        name for name, cfg in PROFILE_MODELS.items()
        if cfg.get("base") == current_base
    )

    if len(siblings) <= 1:
        print(f"  [no other profiles share base '{current_base}']")
        return None

    if console:
        from rich import box as _box
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        import shutil as _shutil

        W = min(max(_shutil.get_terminal_size().columns - 4, 40), 90)
        _S = f"on {C_BG}"

        tbl = Table.grid(padding=(0, 2))
        tbl.add_column(justify="right", style=f"bold {C_ACCENT} on {C_BG}", width=3)
        tbl.add_column()

        for i, name in enumerate(siblings, 1):
            is_current = name == current_model
            marker = Text("▸ ", style=f"bold {C_ACCENT} on {C_BG}") if is_current else Text("  ", style=f"on {C_BG}")
            label = Text(style=f"on {C_BG}")
            label.append(marker)
            label.append(name, style=f"bold {C_ACCENT} on {C_BG}" if is_current else f"{C_FG} on {C_BG}")
            if is_current:
                label.append("  (current)", style=f"{C_DIM} on {C_BG}")
            tbl.add_row(Text(str(i), style=f"bold {C_ACCENT} on {C_BG}"), label)

        hint = Text(style=f"on {C_BG}")
        hint.append("\n  number to switch", style=f"{C_DIM} on {C_BG}")
        hint.append("  ·  ", style=f"{C_DIM} on {C_BG}")
        hint.append("Enter", style=f"bold {C_FG} on {C_BG}")
        hint.append(" to cancel", style=f"{C_DIM} on {C_BG}")

        def section(name):
            t = Text(); t.append("[ ", style=C_DIM); t.append(name, style=f"bold {C_ACCENT}"); t.append(" ]", style=C_DIM); return t

        body_tbl = Table.grid(padding=(0, 1)); body_tbl.add_column()
        phys_line = Text(style=_S)
        phys_line.append("  base  ", style=f"{C_DIM} on {C_BG}")
        phys_line.append("▸  ", style=f"{C_DIM} on {C_BG}")
        phys_line.append(current_base, style=f"bold {C_ACCENT} on {C_BG}")
        body_tbl.add_row(phys_line)
        body_tbl.add_row(Text("", style=_S))
        body_tbl.add_row(tbl)
        body_tbl.add_row(hint)

        panel = Panel(
            Panel(body_tbl, box=_box.SQUARE, border_style=C_DIM, style=_S, padding=(0, 1),
                  title=section("Switch Model"), title_align="left"),
            box=_box.DOUBLE, border_style=C_BORDER, style=f"on {C_SCREEN}", padding=(0, 0), width=W,
        )
        console.print(); console.print(panel); console.print()
    else:
        print(f"\n  base: {current_base}\n")
        for i, name in enumerate(siblings, 1):
            mark = "▸" if name == current_model else " "
            print(f"  {mark} {i}. {name}")
        print()

    try:
        choice = input("  → ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if not choice:
        return None

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(siblings):
            return siblings[idx]
        print("  [out of range]")
        return None

    if choice in PROFILE_MODELS and PROFILE_MODELS[choice].get("base") == current_base:
        return choice

    print("  [invalid model or different base]")
    return None


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
        print(f"  Port  : {ALLMA_PORT}")
        print(f"  /bye or Ctrl+C to exit")
        print(f"{'─'*50}\n")

    try:
        while True:
            try:
                model_tag = model.split(":")[-1]
                user_input = input(f"[{model_tag}]\n>>> ").strip()
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
            if user_input == "/clean":
                print("\033[2J\033[H", end="", flush=True)
                continue
            if user_input == "/model":
                switched = _repl_switch_model(model, _rich_console if _use_rich else None)
                if switched and switched != model:
                    model = switched
                    print(f"  ✓ {model}")
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
                print("[connection error — is Allma running?]")
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
    """Start allma, load a model, and open Claude Code pointed at it."""
    model = args.model

    stop_spinner = threading.Event()
    phase_ref = ["start"]
    spinner = threading.Thread(
        target=_run_llama_spinner, args=(stop_spinner, phase_ref), daemon=True
    )
    spinner.start()

    # 1 ─ Ensure allma is running
    if not _is_running():
        _start_daemon()
        if not _wait_for_server(45):
            stop_spinner.set()
            spinner.join()
            print("Allma failed to start. Check logs: allma logs")
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
        print("Check logs: allma backend logs")
        sys.exit(1)

    if not result or result.get("status") != "loaded":
        print(f"Failed to load model '{model}': unexpected server response.")
        print("Check logs: allma backend logs")
        sys.exit(1)

    import random as _random
    print(f"  ▲  {_random.choice(_LLAMA_PHRASES_READY)}")

    # 4 ─ Patch settings.json for local mode
    _apply_claude_local_fix()

    # 5 ─ Launch claude with model env vars
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{ALLMA_PORT}"
    env["ANTHROPIC_AUTH_TOKEN"] = "dummy"
    env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
    # Remove any previously set conflicting vars
    for var in ["ANTHROPIC_API_KEY"]:
        env.pop(var, None)

    os.execvpe("claude", ["claude"] + args.claude_args, env)


def cmd_launch_hermes(args):
    """Start allma, load a model, and open hermes-agent pointed at it."""
    from shutil import which as _which
    model = args.model

    hermes_bin = _which("hermes")
    if not hermes_bin:
        print("hermes not found in PATH. Install it first:")
        print("  pip install hermes-agent   # or follow https://github.com/NousResearch/hermes-agent")
        sys.exit(1)

    stop_spinner = threading.Event()
    phase_ref = ["start"]
    spinner = threading.Thread(
        target=_run_llama_spinner, args=(stop_spinner, phase_ref), daemon=True
    )
    spinner.start()

    # 1 ─ Ensure allma is running
    if not _is_running():
        _start_daemon()
        if not _wait_for_server(45):
            stop_spinner.set()
            spinner.join()
            print("Allma failed to start. Check logs: allma logs")
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
        print("Check logs: allma backend logs")
        sys.exit(1)

    if not result or result.get("status") != "loaded":
        print(f"Failed to load model '{model}': unexpected server response.")
        print("Check logs: allma backend logs")
        sys.exit(1)

    import random as _random
    print(f"  ▲  {_random.choice(_LLAMA_PHRASES_READY)}")

    # 4 ─ Launch hermes with model flag (hermes config already points to allma)
    hermes_cmd = [hermes_bin, "-m", model] + args.hermes_args
    os.execvp(hermes_bin, hermes_cmd)


def cmd_backend_logs(args):
    """Tail logs of running backend processes; colour-coded per GPU when multiple are active."""
    import time as _time
    import threading as _threading
    import queue as _queue

    # One colour per GPU slot. Cycles if > 6 GPUs.
    _GPU_COLORS = [
        "\033[0;36m",   # GPU 0 — cyan
        "\033[0;32m",   # GPU 1 — green
        "\033[1;33m",   # GPU 2 — yellow
        "\033[0;35m",   # GPU 3 — magenta
        "\033[0;34m",   # GPU 4 — blue
        "\033[0;31m",   # GPU 5 — red
    ]
    _NC = "\033[0m"
    _LABEL_W = 14  # fixed prefix width so columns stay aligned

    # rich colour names matching _GPU_COLORS order, for the sticky banner legend
    _GPU_RICH = ["cyan", "green", "yellow", "magenta", "blue", "red"]

    def _color_for(gpu_id):
        if gpu_id is None:
            return _GPU_COLORS[0]
        return _GPU_COLORS[int(gpu_id) % len(_GPU_COLORS)]

    def _rich_color_for(gpu_id):
        if gpu_id is None:
            return _GPU_RICH[0]
        return _GPU_RICH[int(gpu_id) % len(_GPU_RICH)]

    def _render_banner(servers):
        """Sticky header in the `allma serve --verbose` style. Returns (text, height)."""
        from io import StringIO
        from rich.console import Console
        from rich import box as _box
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        import shutil as _shutil

        W = min(max(_shutil.get_terminal_size().columns - 2, 40), 100)
        _S = f"on {C_BG}"

        title = Text()
        title.append("[ ", style=C_DIM)
        title.append("Backend Log" if len(servers) == 1 else f"Backend Logs · {len(servers)} models",
                     style=f"bold {C_ACCENT}")
        title.append(" ]", style=C_DIM)

        tbl = Table.grid(expand=True, padding=(0, 1))
        tbl.add_column()
        for s in servers:
            gpu = s.get("gpu")
            col = _rich_color_for(gpu)
            row = Text(style=_S)
            row.append("  ● ", style=f"bold {col} on {C_BG}")
            row.append(s["name"], style=f"bold {col} on {C_BG}")
            meta = f"   {s.get('backend', '?')}"
            meta += f"  ·  GPU {gpu}" if isinstance(gpu, int) else "  ·  CPU"
            if s.get("port"):
                meta += f"  ·  :{s['port']}"
            row.append(meta, style=f"{C_DIM} on {C_BG}")
            tbl.add_row(row)

        panel = Panel(
            tbl, title=title, title_align="left", box=_box.DOUBLE,
            border_style=C_BORDER, style=f"on {C_SCREEN}", padding=(0, 0), width=W,
        )
        buf = StringIO()
        Console(file=buf, force_terminal=True, color_system="truecolor", width=W).print(panel)
        text = buf.getvalue().rstrip("\n")
        return text, text.count("\n") + 1

    def _short_label(name):
        """Compact fixed-width label: strip common noisy suffixes, truncate/pad."""
        label = name
        for noise in ["-Instruct", "-instruct", "-Chat", "-chat", "-GGUF",
                      "-MTP", "-OCR", "-FP8", "-Q4", "-Q8",
                      "-Uncensored", "-Abliterated"]:
            label = label.replace(noise, "")
        label = label.replace(":", "-")
        return label[:_LABEL_W].ljust(_LABEL_W)

    def _tail_reader(proc, prefix, stop_event, out_queue):
        """Thread: stream tail -f output into the shared print queue.

        The tail process is owned by the main thread (_start_watcher/_stop_watcher);
        readline() may block until the log grows, so the only reliable way to stop
        this thread is terminating the process — which makes readline return EOF.
        """
        try:
            while not stop_event.is_set():
                line = proc.stdout.readline()
                if line:
                    out_queue.put(prefix + line.rstrip("\n"))
                elif proc.poll() is not None:
                    break
        except Exception:
            pass

    if not _is_running():
        print("Allma is not running.")
        return

    follow = args.follow if hasattr(args, "follow") else True

    # ── static (no -f): show last N lines from each matching backend ─────────
    if not follow:
        data = _get("/v1/ps")
        servers = (data or {}).get("servers", [])
        alive = [s for s in servers if s.get("alive")]
        if not alive:
            print("No backend is currently loaded.")
            return
        if args.name:
            alive = [s for s in alive if args.name.lower() in s["name"].lower()]
        if not alive:
            print(f"No backend matching '{args.name}'.")
            return
        for s in alive:
            logfile = s.get("logfile", "")
            if logfile and Path(logfile).exists():
                print(f"● {s['name']} ({s['backend']})  —  {logfile}\n")
                subprocess.run(["tail", f"-{args.lines}", logfile])
            else:
                print(f"Log file not found for '{s['name']}'.")
        return

    # ── follow mode: multiplex all alive backends under a sticky header ───────
    import shutil as _shutil
    tty = sys.stdout.isatty()
    CSI = "\033["

    out_queue = _queue.Queue()
    # logfile → (stop_event, thread, tail_proc)
    watchers: dict = {}
    last_check = 0.0
    banner_keys = None              # identity of the currently drawn banner
    current_servers: list = []
    resized = {"flag": False}

    old_winch = None
    if tty and hasattr(signal, "SIGWINCH"):
        def _on_winch(_sig, _frm):
            resized["flag"] = True
        old_winch = signal.signal(signal.SIGWINCH, _on_winch)

    def _draw_banner(servers):
        """Paint the fixed header and confine scrolling to the area below it."""
        if not tty or not servers:
            return
        text, h = _render_banner(servers)
        rows = _shutil.get_terminal_size((80, 24)).lines
        sys.stdout.write(CSI + "r")                   # reset any scroll region
        sys.stdout.write(CSI + "2J" + CSI + "H")      # clear + cursor home
        sys.stdout.write(text + "\n")
        if h + 2 < rows:                              # leave room for the log area
            sys.stdout.write(f"{CSI}{h + 1};{rows}r")  # lock region below banner
            sys.stdout.write(f"{CSI}{rows};1H")        # park cursor at the bottom
        sys.stdout.flush()

    def _start_watcher(s):
        logfile = s.get("logfile", "")
        if not logfile or not Path(logfile).exists() or logfile in watchers:
            return
        prefix = f"{_color_for(s.get('gpu'))}[{_short_label(s['name'])}]{_NC} "
        try:
            proc = subprocess.Popen(
                ["tail", "-f", "-n", "30", logfile],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
        except Exception:
            return
        stop_ev = _threading.Event()
        t = _threading.Thread(
            target=_tail_reader,
            args=(proc, prefix, stop_ev, out_queue),
            daemon=True,
        )
        t.start()
        watchers[logfile] = (stop_ev, t, proc)

    def _stop_watcher(logfile):
        if logfile not in watchers:
            return
        stop_ev, _t, proc = watchers.pop(logfile)
        stop_ev.set()
        # Terminating the tail unblocks the reader thread's readline() with EOF.
        try:
            proc.terminate()
        except Exception:
            pass

    try:
        while True:
            # Drain the print queue — block up to 0.1 s waiting for the first line,
            # then flush everything buffered without further waiting.
            try:
                print(out_queue.get(timeout=0.1), flush=True)
                try:
                    while True:
                        print(out_queue.get_nowait(), flush=True)
                except _queue.Empty:
                    pass
            except _queue.Empty:
                pass

            # Terminal was resized — repaint the header for the new geometry.
            if resized["flag"]:
                resized["flag"] = False
                _draw_banner(current_servers)

            # Refresh the backend list every 2 seconds
            now = _time.monotonic()
            if now - last_check < 2.0:
                continue
            last_check = now

            data = _get("/v1/ps")
            servers = (data or {}).get("servers", [])
            alive_servers = [
                s for s in servers
                if s.get("alive") and s.get("logfile") and Path(s["logfile"]).exists()
            ]
            if args.name:
                alive_servers = [s for s in alive_servers if args.name.lower() in s["name"].lower()]
            alive_servers.sort(
                key=lambda s: (s.get("gpu") if isinstance(s.get("gpu"), int) else 99, s["name"])
            )
            alive_logfiles = {s["logfile"] for s in alive_servers}

            if not alive_logfiles:
                for lf in list(watchers):
                    _stop_watcher(lf)
                if banner_keys != ():
                    if tty:
                        sys.stdout.write(CSI + "r" + CSI + "2J" + CSI + "H")
                    print("⏳ No backend loaded — waiting... (Ctrl+C to exit)", flush=True)
                    banner_keys = ()
                    current_servers = []
                continue

            keys = tuple((s["name"], s.get("gpu"), s["logfile"]) for s in alive_servers)

            # Stop watchers whose backends have unloaded
            for lf in list(watchers):
                if lf not in alive_logfiles:
                    _stop_watcher(lf)
            # Reap watchers whose tail died (e.g. log rotated/truncated) so they
            # get restarted below instead of going silent for an alive backend
            for lf, (_ev, t, proc) in list(watchers.items()):
                if proc.poll() is not None or not t.is_alive():
                    _stop_watcher(lf)
            # Start watchers for newly loaded backends
            for s in alive_servers:
                _start_watcher(s)

            # Repaint the sticky header only when the set of models changes.
            if keys != banner_keys:
                current_servers = alive_servers
                banner_keys = keys
                _draw_banner(alive_servers)

    except KeyboardInterrupt:
        pass
    finally:
        for lf in list(watchers):
            _stop_watcher(lf)
        if tty:
            sys.stdout.write(CSI + "r")        # release the scroll region
            sys.stdout.write(CSI + "?25h")     # make sure the cursor is visible
            sys.stdout.write("\n")
            sys.stdout.flush()
        if old_winch is not None:
            signal.signal(signal.SIGWINCH, old_winch)


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
            print("Hardware Profile:")
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
        print("Allma server is not running.")
        print("Start with: allma serve")


def cmd_calibrate(args):
    """Pre-calibrate a model without loading it."""
    import asyncio

    sys.path.insert(0, str(ALLMA_DIR))
    from core.bootstrap import BootstrapDetector
    from core.config import BASE_MODELS
    from core.gpu import get_model_vram_need

    if args.model not in BASE_MODELS:
        print(f"❌ Model '{args.model}' not found in configs.")
        print("Available base models:")
        for name in sorted(BASE_MODELS.keys()):
            print(f"   · {name}")
        return

    print(f"Detecting hardware...")
    try:
        profile = asyncio.run(BootstrapDetector.detect_hardware())
    except Exception as e:
        print(f"❌ Hardware detection failed: {e}")
        return

    cfg = BASE_MODELS[args.model]
    model_size_gb = get_model_vram_need(cfg, args.model)

    print(f"Calibrating {args.model}...")
    try:
        calib = asyncio.run(
            BootstrapDetector.calibrate_for_model(
                base_name=args.model,
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
                print(f"      ⚠  {warn}")

    except Exception as e:
        print(f"✕ Calibration failed: {e}")


def cmd_download(args):
    from core.downloader import run_download
    run_download(args.url)


def cmd_tui(_args):
    """Open the configuration panel (3-column TUI, replaces the old wizard)."""
    from allma_tui import AllmaTUI
    AllmaTUI().run()


def cmd_update(args):
    """Update allma, vLLM, and/or llama.cpp to their latest versions."""
    target = args.target  # "all" | "allma" | "vllm" | "llama"
    auto_yes = getattr(args, "yes", False)

    def _banner(title):
        print(f"\n  ── {title} {'─' * max(0, 44 - len(title))}")

    def _run(label, cmd, cwd=None, env=None):
        """Run a command, stream output, return success."""
        import subprocess as _sp
        print(f"  $ {' '.join(cmd)}")
        result = _sp.run(cmd, cwd=cwd, env=env)
        if result.returncode != 0:
            print(f"  ✗ {label} failed (exit {result.returncode})")
            return False
        print(f"  ✓ {label}")
        return True

    def _ver(cmd):
        """Return first line of a command's output, or ''."""
        import subprocess as _sp
        try:
            return _sp.check_output(cmd, stderr=_sp.DEVNULL, text=True).strip().split("\n")[0]
        except Exception:
            return ""

    def _has_uv() -> bool:
        """True when `uv` is on PATH — its resolver beats pip's for CUDA stacks."""
        import shutil as _sh
        return _sh.which("uv") is not None

    def _pip_install_cmd(*extra_args: str) -> list:
        """Build `pip install ...` — prefers `uv pip` when available.

        Always targets the venv Python explicitly so upgrades hit the right
        environment even when the user runs `allma update` outside of it."""
        if _has_uv():
            return ["uv", "pip", "install", "--python", PYTHON, *extra_args]
        return [PYTHON, "-m", "pip", "install", *extra_args]

    def _confirm(question: str) -> bool:
        """Interactive y/N gate. Auto-approved when `--yes` was passed or
        stdin is not a TTY (e.g. running from a script)."""
        if auto_yes:
            return True
        import sys as _sys
        if not _sys.stdin.isatty():
            print(f"  {question} [non-interactive: aborting; pass --yes to skip]")
            return False
        try:
            reply = input(f"  {question} [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        return reply in ("y", "yes")

    def _backends_running() -> list:
        """Return the list of backend names currently loaded — the caller
        should refuse to touch Python libs while any backend is alive."""
        import urllib.request as _ur
        import urllib.error as _ue
        import json as _json
        try:
            with _ur.urlopen("http://127.0.0.1:9000/v1/ps", timeout=1.5) as r:
                data = _json.load(r)
        except (_ue.URLError, TimeoutError, ConnectionError, OSError):
            return []
        except Exception:
            return []
        return [s["name"] for s in data.get("servers", []) if s.get("alive")]

    updated_anything = False

    # Refuse to touch Python libs when a backend is still holding them open.
    # (Doesn't gate the llama.cpp binary rebuild — that copies over disk-only.)
    python_libs_targeted = target in ("all", "allma", "vllm")
    if python_libs_targeted:
        alive = _backends_running()
        if alive:
            print()
            print(f"  ✗ {len(alive)} backend(s) still running: {', '.join(alive)}")
            print("  Python libs (allma/vllm) can't be safely upgraded while they")
            print("  are in use. Stop everything first:")
            print("      allma stop")
            return

    # ── allma (git pull) ────────────────────────────────────────────────────────
    if target in ("all", "allma"):
        _banner("Allma")
        git_dir = ALLMA_DIR / ".git"
        if not git_dir.exists():
            print("  Not a git repository — skipping allma update")
        else:
            before = _ver(["git", "-C", str(ALLMA_DIR), "rev-parse", "--short", "HEAD"])
            ok = _run("git pull", ["git", "-C", str(ALLMA_DIR), "pull", "--ff-only"])
            if ok:
                after = _ver(["git", "-C", str(ALLMA_DIR), "rev-parse", "--short", "HEAD"])
                if before != after:
                    print(f"  {before} → {after}")
                    updated_anything = True
                    # Reinstall Python deps in case requirements changed
                    req = ALLMA_DIR / "requirements.txt"
                    if req.exists():
                        _run("install requirements",
                             _pip_install_cmd("-q", "-r", str(req)))
                else:
                    print("  Already up to date.")

    # ── vLLM ────────────────────────────────────────────────────────────────────
    if target in ("all", "vllm"):
        _banner("vLLM")
        before = _ver([PYTHON, "-c", "import vllm; print(vllm.__version__)"])
        if not before:
            print("  vLLM not installed. Install it with:")
            print(f"  {' '.join(_pip_install_cmd('vllm'))}")
        else:
            using = "uv" if _has_uv() else "pip"
            print(f"  Current: {before}  (using {using})")

            # Dry-run first — surfaces breaking upgrades (torch major bump,
            # xformers/triton downgrade) BEFORE anything touches the venv.
            if _has_uv():
                dry = ["uv", "pip", "install", "--python", PYTHON,
                       "--upgrade", "vllm", "--dry-run"]
            else:
                dry = [PYTHON, "-m", "pip", "install", "--upgrade", "vllm",
                       "--dry-run"]

            print("  Planning changes:")
            import subprocess as _sp3
            plan = _sp3.run(dry, capture_output=True, text=True)
            plan_text = (plan.stdout or "") + (plan.stderr or "")
            for line in plan_text.splitlines():
                print(f"    {line}")

            # Cheap sanity checks against known-bad transitions.
            import re as _re3
            warnings: list[str] = []
            for m in _re3.finditer(
                r"(torch|triton|xformers|flash-attn|vllm-flash-attn)\s+"
                r"([0-9.+a-z]+)\s*(?:->|→)\s*([0-9.+a-z]+)",
                plan_text.lower(),
            ):
                pkg, old, new = m.group(1), m.group(2), m.group(3)
                if old.split(".")[0] > new.split(".")[0]:
                    warnings.append(f"    ⚠ {pkg}: {old} → {new} (major DOWNGRADE)")
                elif pkg == "torch" and old.split(".")[0] != new.split(".")[0]:
                    warnings.append(f"    ⚠ {pkg}: {old} → {new} (major bump — may break flash-attn/triton)")
            if warnings:
                print("  Sanity check flagged:")
                for w in warnings:
                    print(w)

            if plan.returncode != 0:
                print("  ✗ dry-run failed — aborting; check output above.")
            elif not _confirm("Apply this vLLM upgrade?"):
                print("  Aborted. No changes.")
            else:
                ok = _run("upgrade vllm",
                          _pip_install_cmd("--upgrade", "vllm"))
                if ok:
                    after = _ver([PYTHON, "-c", "import vllm; print(vllm.__version__)"])
                    if before != after:
                        print(f"  {before} → {after}")
                        updated_anything = True
                        print(f"  Rollback if needed:  "
                              f"{' '.join(_pip_install_cmd(f'vllm=={before}'))}")
                    else:
                        print("  Already up to date.")

    # ── llama.cpp ────────────────────────────────────────────────────────────────
    if target in ("all", "llama"):
        _banner("llama.cpp")
        from core.config import LLAMA_CPP_PATH, LLAMA_CPP_PYTHON_BACKEND

        if LLAMA_CPP_PYTHON_BACKEND:
            # pip-based install
            before = _ver([PYTHON, "-c", "import llama_cpp; print(llama_cpp.__version__)"])
            print(f"  Current: llama-cpp-python {before}")
            # Detect CUDA version for the right wheel index
            import subprocess as _sp2
            cuda_ver = ""
            try:
                smi = _sp2.check_output(["nvidia-smi"], text=True)
                import re as _re2
                m = _re2.search(r"CUDA Version: (\d+)", smi)
                if m:
                    cuda_ver = m.group(1)
            except Exception:
                pass
            pip_cmd = _pip_install_cmd("--upgrade", "llama-cpp-python[server]")
            if cuda_ver:
                pip_cmd += ["--extra-index-url", f"https://abetlen.github.io/llama-cpp-python/whl/cu{cuda_ver}"]
            ok = _run("upgrade llama-cpp-python", pip_cmd)
            if ok:
                after = _ver([PYTHON, "-c", "import llama_cpp; print(llama_cpp.__version__)"])
                if before != after:
                    print(f"  {before} → {after}")
                    updated_anything = True
                else:
                    print("  Already up to date.")
        else:
            # Native binary — rebuild from source if the build dir exists
            build_dir = Path.home() / ".local" / "share" / "llama.cpp"
            if not (build_dir / ".git").exists():
                # Try common manual build locations
                for d in [Path.home() / "llama.cpp", Path.home() / "AI" / "llama.cpp"]:
                    if (d / ".git").exists():
                        build_dir = d
                        break
                else:
                    print(f"  llama-server found at: {LLAMA_CPP_PATH}")
                    print("  Source directory not found — cannot auto-update.")
                    print("  To update, re-run:  bash scripts/install-llama-cpp.sh")
                    build_dir = None

            if build_dir:
                before = _ver([LLAMA_CPP_PATH, "--version"])
                print(f"  Current: {before or LLAMA_CPP_PATH}")
                ok = _run("git pull", ["git", "-C", str(build_dir), "pull", "--ff-only"])
                if ok:
                    import multiprocessing as _mp
                    jobs = str(_mp.cpu_count() or 4)
                    ok2 = _run("cmake build", ["cmake", "--build", str(build_dir / "build"),
                                               "--config", "Release", "-j", jobs, "--target", "llama-server"])
                    if ok2:
                        # Copy updated binary — skip when LLAMA_CPP_PATH already
                        # points at the build output (avoids SameFileError).
                        built = build_dir / "build" / "bin" / "llama-server"
                        if built.exists():
                            try:
                                same = built.resolve() == Path(LLAMA_CPP_PATH).resolve()
                            except (OSError, RuntimeError):
                                same = False
                            if not same:
                                import shutil as _sh
                                _sh.copy2(str(built), LLAMA_CPP_PATH)
                        after = _ver([LLAMA_CPP_PATH, "--version"])
                        if before != after:
                            print(f"  {before} → {after}")
                            updated_anything = True
                        else:
                            print("  Already up to date.")

    print()
    if updated_anything:
        print("  Update complete. Restart allma to apply changes:")
        print("  allma restart")
    else:
        print("  Everything is already up to date.")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    # Internal: running as watchdog daemon
    if len(sys.argv) > 1 and sys.argv[1] == "__watchdog__":
        verbose = "--verbose" in sys.argv
        _run_watchdog(verbose=verbose)
        return

    parser = argparse.ArgumentParser(
        prog="allma",
        description="Allma — local LLM manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
commands:
  server:
    serve                            start daemon in background
    serve -v                         run in foreground, printing logs to terminal
    serve -bv                        open a terminal window tailing each backend log
    serve -f                         kill whatever is on ALLMA_PORT, then start
    restart                          stop + restart
    stop                             stop server and all backends
    status                           show if server is running

  models:
    list                             list available profiles
    ps                               show loaded models and GPU usage
    run <profile>                    interactive chat with a model
    unload <model>                   unload a model and free its VRAM

  logs:
    logs                             show last 50 lines of allma log
    logs -f                          follow allma log live
    logs -n 200                      show last 200 lines
    backend logs [name]              tail a backend process log (-f to follow)

  clients:
    launch claude <profile>          open Claude Code pointed at a local model
    launch hermes <profile>          open hermes-agent pointed at a local model

  setup:
    download <hf-repo>               download a model from HuggingFace
    wizard                           interactive wizard to create configs
    hardware-detect                  show detected GPUs and VRAM
    calibrate <model>                pre-calibrate model (warmup KV cache)
    update [all|allma|vllm|llama]    update allma and/or backends
"""
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # serve
    p_serve = sub.add_parser("serve", help="Start the Allma server")
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
        help=f"Kill any process occupying port {ALLMA_PORT} before starting",
    )
    p_serve.set_defaults(func=cmd_serve)

    # restart
    p_restart = sub.add_parser("restart", help="Stop and restart the Allma server")
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
    p_stop = sub.add_parser("stop", help="Stop the Allma server")
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

    # unload
    p_unload = sub.add_parser("unload", help="Unload a model and free its VRAM")
    p_unload.add_argument("model", help="Base model name (e.g. 'Qwen3.5-27b') or profile name")
    p_unload.set_defaults(func=cmd_unload)

    p_reload = sub.add_parser("reload", help="Unload a model and load it again (pick up config changes)")
    p_reload.add_argument("model", help="Base model name or profile name")
    p_reload.set_defaults(func=cmd_reload)

    # logs
    p_logs = sub.add_parser("logs", help="Show Allma logs")
    p_logs.add_argument("-f", "--follow", action="store_true", help="Follow log output")
    p_logs.add_argument("-n", "--lines", type=int, default=50, metavar="N", help="Lines to show (default: 50)")
    p_logs.set_defaults(func=cmd_logs)

    # run
    p_run = sub.add_parser("run", help="Chat with a model interactively")
    p_run.add_argument("model", help="Model name (e.g. 'Qwen3.5:27b') or a HuggingFace repo/URL")
    p_run.add_argument("--gpu", type=int, default=None, metavar="N",
                      help="Pin model to GPU N (0-based). If not specified, auto-select by available VRAM")
    p_run.set_defaults(func=cmd_run)

    p_qs = sub.add_parser("quickstart", help="Guided first model: goal → curated pick → download → chat")
    p_qs.add_argument("--gpu", type=int, default=None, metavar="N",
                      help="Pin model to GPU N (0-based)")
    p_qs.set_defaults(func=cmd_quickstart, model=None)

    # launch
    p_launch = sub.add_parser("launch", help="Launch an AI client with a local model")
    launch_sub = p_launch.add_subparsers(dest="launch_target", metavar="<client>")
    launch_sub.required = True

    p_lc = launch_sub.add_parser("claude", help="Open Claude Code with a local model")
    p_lc.add_argument("model", help="Profile model name (e.g. 'Qwen3.5:27b-Code')")
    p_lc.add_argument("--gpu", type=int, default=None, metavar="ID",
                      help="Force model onto a specific GPU (0-indexed)")
    p_lc.add_argument("claude_args", nargs=argparse.REMAINDER,
                      help="Extra arguments forwarded to claude")
    p_lc.set_defaults(func=cmd_launch)

    p_lh = launch_sub.add_parser("hermes", help="Open hermes-agent with a local model")
    p_lh.add_argument("model", help="Profile model name (e.g. 'Qwen3.6_ABT:27B-Q4')")
    p_lh.add_argument("--gpu", type=int, default=None, metavar="ID",
                      help="Force model onto a specific GPU (0-indexed)")
    p_lh.add_argument("hermes_args", nargs=argparse.REMAINDER,
                      help="Extra arguments forwarded to hermes")
    p_lh.set_defaults(func=cmd_launch_hermes)

    # hardware-detect
    p_hw = sub.add_parser("hardware-detect", help="Detect hardware and show info")
    p_hw.set_defaults(func=cmd_hardware_detect)

    # calibrate
    p_calib = sub.add_parser("calibrate", help="Pre-calibrate a model")
    p_calib.add_argument("model", help="Base model name (e.g. 'Qwen3.5-27b')")
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

    # download
    p_dl = sub.add_parser("download", help="Download a model from HuggingFace and create configs")
    p_dl.add_argument("url", help="HuggingFace URL or repo id (e.g. Qwen/Qwen2.5-7B-Instruct)")
    p_dl.set_defaults(func=cmd_download)

    # wizard
    p_tui = sub.add_parser("tui", aliases=["wizard"],
                           help="Configuration panel (models, base config, profiles)")
    p_tui.set_defaults(func=cmd_tui)

    # update
    p_up = sub.add_parser("update", help="Update allma and/or backends to latest versions")
    p_up.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["all", "allma", "vllm", "llama"],
        help="What to update: all (default), allma, vllm, llama",
    )
    p_up.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip the confirmation prompt after the dry-run (useful for scripts)",
    )
    p_up.set_defaults(func=cmd_update)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
