#!/usr/bin/env python3
"""
Allama CLI — allama serve / run / list / ps / stop / logs
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
    """Retro DOS-style bouncing bar spinner. label_ref[0] can be mutated to change the label."""
    track = 8
    positions = list(range(track)) + list(range(track - 2, 0, -1))
    start = time.time()
    last_len = 0
    i = 0
    while not stop_event.is_set():
        elapsed = time.time() - start
        pos = positions[i % len(positions)]
        bar = "░" * pos + "▓" + "░" * (track - pos - 1)
        line = f"  [{bar}]  {label_ref[0]}{elapsed:.1f}s"
        sys.stdout.write(f"\r{line}")
        sys.stdout.flush()
        last_len = len(line)
        time.sleep(0.08)
        i += 1
    sys.stdout.write("\r" + " " * (last_len + 2) + "\r")
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


def cmd_run(args):
    """Load a model and open an interactive chat session."""
    model = args.model
    provider = getattr(args, "provider", None)
    on_keyword = getattr(args, "on", None)
    persist = getattr(args, "persist", False)

    # Handle new syntax: allama run <model> on <provider>
    if on_keyword == "on" and provider:
        from core.provider_resolver import resolve_remote_model
        from core.config import load_dynamic_models, save_dynamic_models, DYNAMIC_MODELS

        logger.info(f"🌐 Resolving {model} from {provider}...")

        # Try to resolve the model from remote provider
        metadata = resolve_remote_model(provider, model)
        if not metadata:
            print(f"❌ Model '{model}' not found on {provider}")
            sys.exit(1)

        # Create a temporary logical model name
        logical_model_name = f"{provider}/{model}"

        # Check if already cached
        dynamic_models = load_dynamic_models()
        if logical_model_name not in dynamic_models:
            # Create physical model config for remote provider
            dynamic_models[logical_model_name] = {
                "backend": provider,
                "model_id": model,
                "base_url": metadata.get("base_url"),
                "context_window": metadata.get("context_window", 4096),
                "max_tokens": metadata.get("max_tokens", 2048),
            }

            if persist:
                save_dynamic_models(dynamic_models)
                print(f"💾 Saved {logical_model_name} to dynamic_models.json")

        model = logical_model_name
    elif on_keyword and provider:
        print(f"❌ Invalid syntax. Use: allama run <model> on <provider>")
        sys.exit(1)
    elif on_keyword == "on" or provider:
        print(f"❌ Invalid syntax. Use: allama run <model> on <provider>")
        sys.exit(1)

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
    p_run.add_argument("model", help="Model name or remote model (e.g. 'Qwen3.5:27b' or 'gpt-4')")
    p_run.add_argument("on", nargs="?", default=None, help="Keyword 'on' (internal)")
    p_run.add_argument("provider", nargs="?", default=None, help="Remote provider (opencode, openclaw)")
    p_run.add_argument("--persist", action="store_true", help="Save remote model for future use")
    p_run.set_defaults(func=cmd_run)

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
