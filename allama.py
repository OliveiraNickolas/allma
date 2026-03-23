import asyncio
import json
import logging
import logging.handlers
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import psutil
import uvicorn
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from rich import box
    from rich.align import Align
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

class JSONFormatter(logging.Formatter):
    """Structured JSON logging for production-friendly logs."""
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)
        extra_data = getattr(record, "extra_data", None)
        if extra_data is not None:
            log_entry["data"] = extra_data
        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Colored console output with emojis for human readability."""
    colors = {
        "DEBUG": "\033[36m ",
        "INFO": "\033[32m ",
        "WARNING": "\033[33m ",
        "ERROR": "\033[31m ",
        "CRITICAL": "\033[35m ",
    }
    reset = "\033[0m"
    datefmt = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        emoji = {
            logging.DEBUG: "🔍",
            logging.INFO: "ℹ️",
            logging.WARNING: "⚠️",
            logging.ERROR: "❌",
            logging.CRITICAL: "💥",
        }.get(record.levelno, "📝")
        color = self.colors.get(record.levelname, "")
        level_style = f"{color}{record.levelname}{self.reset}"
        return f"{self.formatTime(record)} - {emoji} {level_style} - {record.getMessage()}"



# Global config
ALLAMA_PORT = int(os.environ.get("ALLAMA_PORT", "9000"))


def format_user_agent(ua: str) -> str:
    """Simplify user agent for logging."""
    if not ua or ua == "unknown":
        return "unknown"
    ua_lower = ua.lower()
    if "claude" in ua_lower or "claude-code" in ua_lower:
        if "vscode" in ua_lower or "extension" in ua_lower:
            return "Claude - VSCode"
        return "Claude - Terminal"
    if "openwebui" in ua_lower:
        return "OpenWebUI"
    if "fastapi" in ua_lower or "uvicorn" in ua_lower:
        return "FastAPI"
    if "python" in ua_lower and "requests" in ua_lower:
        return "Python/requests"
    # OpenWebUI sends aiohttp with Mozilla/5.0 prefix
    if ua.startswith("Mozilla/5.0") and "aiohttp" in ua_lower and "python" in ua_lower:
        return "OpenWebUI"
    if "python" in ua_lower and "aiohttp" in ua_lower:
        return "Python/aiohttp"
    if "curl" in ua_lower:
        return "curl"
    if "wget" in ua_lower:
        return "wget"
    # OpenWebUI Chrome - desktop (Linux, macOS, Windows)
    if "Mozilla/5.0" in ua and "Chrome/" in ua and "Safari/537.36" in ua and "Mobile" not in ua and "Tablet" not in ua:
        return "OpenWebUI Desktop"
    # OpenWebUI Chrome mobile (Android)
    if "Mozilla/5.0" in ua and "CriOS/" in ua and "Mobile" in ua:
        return "OpenWebUI Mobile"
    # OpenWebUI Safari iOS (iPhone, iPad)
    if "Mozilla/5.0" in ua and "Version/26.3" in ua and "Safari/605.1" in ua:
        if "iPhone" in ua:
            return "OpenWebUI Mobile"
        if "iPad" in ua or "Tablet" in ua:
            return "OpenWebUI Tablet"
        return "OpenWebUI Desktop"
    # Truncate longer agents
    if len(ua) > 50:
        return ua[:47] + "..."
    return ua


# Global config
VLLM_BASE_PORT = int(os.environ.get("VLLM_BASE_PORT", "8000"))
LLAMA_BASE_PORT = int(os.environ.get("LLAMA_BASE_PORT", "9001"))
KEEP_ALIVE_SECONDS = int(os.environ.get("KEEP_ALIVE_SECONDS", "600"))
HEALTH_CHECK_INTERVAL = int(os.environ.get("HEALTH_CHECK_INTERVAL", "60"))
GPU_MEMORY_THRESHOLD_GB = float(os.environ.get("GPU_MEMORY_THRESHOLD_GB", "1.0"))
AUTO_SWAP_ENABLED = os.environ.get("AUTO_SWAP_ENABLED", "true").lower() == "true"
SWAP_IDLE_THRESHOLD = int(os.environ.get("SWAP_IDLE_THRESHOLD", "300"))
ALLAMA_LOG_DIR = Path(os.environ.get("ALLAMA_LOG_DIR", "./logs"))
CONFIG_DIR = Path(os.environ.get("ALLAMA_CONFIG_DIR", "./configs"))
PATH_TO_ALLAMA = os.environ.get("PATH_TO_ALLAMA", "/home/nick/AI/vllm/allama")
LLAMA_CPP_PATH = os.environ.get(
    "LLAMA_CPP_PATH", "/home/nick/AI/llama.cpp/build/bin/llama-server"
)
MAX_MESSAGES = int(os.environ.get("MAX_MESSAGES", "15"))

# Logging setup
ALLAMA_LOG_DIR.mkdir(exist_ok=True)

file_handler = logging.handlers.RotatingFileHandler(
    f"{ALLAMA_LOG_DIR}/allama.log",
    maxBytes=10_485_760,
    backupCount=5,
    encoding="utf-8",
)
file_handler.setFormatter(JSONFormatter())

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColoredFormatter())

# Configurar root logger manualmente para evitar conflitos
root_logger = logging.getLogger()
root_logger.handlers = []
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
root_logger.setLevel(logging.INFO)

# Suppress uvicorn logs - we have custom logging
logger = logging.getLogger("Allama")

# Configure uvicorn loggers to use same formatters, but only show errors
for uv_logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.info"]:
    uv_logger = logging.getLogger(uv_logger_name)
    uv_logger.handlers = []
    uv_logger.addHandler(console_handler)
    uv_logger.addHandler(file_handler)
    uv_logger.setLevel(logging.ERROR)  # Only errors, not access logs

# Also suppress httpx access logs - we log manually before sending requests
logging.getLogger("httpx").setLevel(logging.ERROR)
# ==============================================================================
# GLOBAL STATE
# ==============================================================================
active_servers: Dict[str, dict] = {}
server_idle_time: Dict[str, float] = {}
next_vllm_port = VLLM_BASE_PORT
next_llama_port = LLAMA_BASE_PORT
global_lock = threading.Lock()
running = True
ALLAMA_PID = os.getpid()
gpu_allocation: Dict[str, int] = {}

# Model configurations
PHYSICAL_MODELS: Dict[str, Dict[str, Any]] = {}
LOGICAL_MODELS: Dict[str, Dict[str, Any]] = {}


def load_models_from_configs() -> tuple[dict, dict]:
    try:
        from configs.loader import load_models_from_configs as load_configs
        return load_configs(str(CONFIG_DIR))
    except FileNotFoundError as e:
        logger.warning(f"Config directory not found: {e}")
        return {}, {}
    except Exception as e:
        logger.error(f"Failed to load configs: {e}")
        return {}, {}


PHYSICAL_MODELS, LOGICAL_MODELS = load_models_from_configs()


def is_port_free(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False


def get_free_gpu_memory() -> list[dict]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,index", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        free_mbs = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = line.split(",")
                gpu_id = int(parts[-1].strip()) if len(parts) > 1 else 0
                free_mb = float(parts[0].strip()) if parts[0].strip() else 0.0
                free_mbs.append({"index": gpu_id, "free_mb": free_mb, "free_gb": free_mb / 1024})
        return free_mbs
    except Exception as e:
        logger.error(f"Error reading VRAM: {e}")
        return []


def get_best_gpu() -> int:
    gpus = get_free_gpu_memory()
    if not gpus:
        logger.warning("No GPU info available, using GPU 0")
        return 0
    best = max(gpus, key=lambda g: g["free_gb"])
    logger.debug(f"Selected GPU {best['index']} with {best['free_gb']:.1f}GB free")
    return best["index"]


def get_model_vram_need(cfg: Dict[str, Any], physical_name: str) -> float:
    backend = cfg.get("backend", "vllm")
    try:
        if backend == "vllm":
            model_path = cfg.get("path", "")
            if not model_path or not os.path.isdir(model_path):
                return 4.0
            total_size_gb = sum(
                os.path.getsize(os.path.join(root, file))
                for root, _, files in os.walk(model_path)
                for file in files
            ) / (1024**3)
            model_len = int(cfg.get("max_model_len", "40960"))
            max_seqs = int(cfg.get("max_num_seqs", "8"))
            kv_cache_overhead = (model_len * max_seqs * 2) / (1024**3)
            return total_size_gb * 1.06 + kv_cache_overhead + 1.0
        elif backend == "llama.cpp":
            model_file = cfg.get("model", "")
            if not model_file or not os.path.isfile(model_file):
                return 4.0
            size_gb = os.path.getsize(model_file) / (1024**3)
            n_ctx = int(cfg.get("n_ctx", "40960"))
            context_overhead = (n_ctx * 2 * 16) / (1024**3)
            return size_gb * 1.06 + context_overhead + 0.5
    except Exception as e:
        logger.error(f"Error estimating VRAM for {physical_name}: {e}")
    return 4.0


def kill_vram_fast():
    """
    Kill ONLY processes managed by Allama (active_servers).
    Does NOT kill external vllm/llama processes - that was the bug.
    """
    logger.info("🔨 Aggressive VRAM shutdown initiated...")
    pids_killed = []

    # First, get list of known PIDs from active_servers
    known_pids = set()
    with global_lock:
        for name, server in active_servers.items():
            known_pids.add(server["pid"])

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.error(f"nvidia-smi failed: {result.stderr}")
            return

        lines = result.stdout.strip().split("\n")
        for line in lines:
            if not line.strip():
                continue
            parts = line.split(",", 1)
            if len(parts) < 2:
                continue
            pid = int(parts[0].strip())
            procname = parts[1].strip().lower()

            if pid == ALLAMA_PID:
                logger.debug(f"Skipping ALLAMA itself (PID {ALLAMA_PID})")
                continue

            # BUG FIX: Only kill if we know this process (active_servers)
            if pid not in known_pids:
                logger.debug(f"Skipping external process {pid} ({procname}) - not managed by Allama")
                continue

            # Now actually kill it
            logger.info(f"Killing managed PID {pid} ({procname})")
            try:
                kill_process_tree(pid, timeout=1)
                pids_killed.append(pid)
                logger.info(f"Killed managed PID {pid} {procname}")
            except Exception as e:
                logger.error(f"Error killing {pid}: {e}")

        if pids_killed:
            logger.info(f"Shutdown complete: {len(pids_killed)} ALLAMA-managed processes")
            time.sleep(2)
            gpus = get_free_gpu_memory()
            freegb = sum(g["free_gb"] for g in gpus)
            logger.info(f"VRAM Free after shutdown: {freegb:.1f}GB")
        else:
            logger.info("No ALLAMA-managed processes to shutdown")
    except Exception as e:
        logger.error(f"Error in killvramfast: {e}")
        import traceback
        logger.error(traceback.format_exc())


def build_vllm_cmd(physical_name: str) -> tuple[list, int, int]:
    global next_vllm_port
    cfg = PHYSICAL_MODELS[physical_name]
    attempts = 0
    while attempts < 1000:
        check_port = VLLM_BASE_PORT + (next_vllm_port + attempts) % (65535 - VLLM_BASE_PORT)
        if is_port_free(check_port):
            port = check_port
            next_vllm_port = port + 1
            
            break
        attempts += 1
    else:
        raise RuntimeError(f"Could not find free port for {physical_name} in 1000 attempts")

    if physical_name not in gpu_allocation:
        gpu_id = get_best_gpu()
        gpu_allocation[physical_name] = gpu_id
    else:
        gpu_id = gpu_allocation[physical_name]
    logger.info(f"🎯 {physical_name} ➡ GPU {gpu_id}")

    cmd = [
        "vllm", "serve", cfg["path"],
        "--tokenizer", cfg["tokenizer"],
        "--tensor-parallel-size", cfg["tensor_parallel"],
        "--gpu-memory-utilization", cfg.get("gpu_memory_utilization", "0.90"),
        "--max-model-len", cfg["max_model_len"],
        "--max-num-seqs", cfg.get("max_num_seqs", "8"),
        "--generation-config", "vllm",
        "--port", str(port),
        "--host", "127.0.0.1",
        "--api-key", "dummy",
    ]
    cmd.extend(cfg.get("extra_args", []))
    return cmd, port, gpu_id


def build_llama_cmd(physical_name: str) -> tuple[list, int, int]:
    global next_llama_port
    cfg = PHYSICAL_MODELS[physical_name]
    attempts = 0
    while attempts < 1000:
        check_port = LLAMA_BASE_PORT + (next_llama_port + attempts) % (65535 - LLAMA_BASE_PORT)
        if is_port_free(check_port):
            port = check_port
            next_llama_port = port + 1
            
            break
        attempts += 1
    else:
        raise RuntimeError(f"Could not find free port for {physical_name} in 1000 attempts")

    gpu_id = gpu_allocation.get(physical_name)
    if gpu_id is None:
        gpu_id = get_best_gpu()
        gpu_allocation[physical_name] = gpu_id
    logger.info(f"🎯 {physical_name} ➡ GPU {gpu_id}")

    cmd = [
        LLAMA_CPP_PATH,
        "-m", cfg["model"],
        "--host", "127.0.0.1",
        "--port", str(port),
        "-t", cfg.get("n_threads", "16"),
        "-c", cfg.get("n_ctx", "40960"),
        "-b", cfg.get("n_batch", "1024"),
        "-ngl", cfg.get("n_gpu_layers", "-1"),
    ]
    if cfg.get("mmproj") and os.path.exists(cfg["mmproj"]):
        cmd.extend(["--mmproj", cfg["mmproj"]])
    if cfg.get("chat_template_file") and os.path.exists(cfg["chat_template_file"]):
        cmd.extend(["--chat-template-file", cfg["chat_template_file"]])
    cmd.extend(cfg.get("extra_args", []))
    return cmd, port, gpu_id


def kill_process_tree(pid: int, timeout: int = 2) -> bool:
    try:
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(children, timeout=timeout)
        try:
            parent.kill()
            parent.wait(timeout=1)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            pass
        return True
    except psutil.NoSuchProcess:
        return True
    except Exception as e:
        logger.error(f"Error killing process tree {pid}: {e}")
        return False


def shutdown_server(physicalname: str, reason: str = "user", fast: bool = False):
    global active_servers, server_idle_time
    proc = None
    port = None
    backend = None

    with global_lock:
        if physicalname not in active_servers:
            logger.warning(f"{physicalname} not active")
            return
        server = active_servers[physicalname]
        proc = server["process"]
        pid = proc.pid
        port = server["port"]
        backend = server.get("backend", "unknown")

    logger.info(f"📤 Unload {physicalname}:{port} ({reason})")

    if proc and proc.poll() is None:
        logger.info(f"💀 Killing PID {pid} ({backend})")
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            kill_process_tree(pid, timeout=3 if fast else 5)
        time.sleep(3)

    # Called before global_lock release - port allocated to this server is now free
    # We don't reset the next_port here since we rely on is_port_free() to find available ports

    with global_lock:
        active_servers.pop(physicalname, None)
        server_idle_time.pop(physicalname, None)
        # Clear GPU allocation to allow reassignment
        gpu_allocation.pop(physicalname, None)

    gpus = get_free_gpu_memory()
    freegb = sum(g["free_gb"] for g in gpus)
    logger.info(f"🗑️  {physicalname} unloaded. VRAM free: {freegb:.1f}GB")


def list_gpu_processes(gpu_ids: Optional[list[int]] = None) -> list[Dict[str, Any]]:
    """Lista processos usando VRAM em uma ou mais GPUs."""
    result = []
    cmd = ["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_memory_usage", "--format=csv,noheader"]
    if gpu_ids:
        cmd = ["nvidia-smi", "-i", ",".join(map(str, gpu_ids))] + cmd[1:]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        for line in out.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    pid = int(parts[0])
                    name = parts[1]
                    mem_mb = int(parts[2].replace("MiB", "").strip())
                    result.append({"pid": pid, "name": name, "memory_mb": mem_mb})
                except (ValueError, IndexError):
                    pass
    except subprocess.SubprocessError:
        pass
    return result


class LoadingSpinner:
    def __init__(self, message: str = "Loading"):
        self.message = message
        self.spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.running = False
        self._thread = None
        self._start_time = None

    def _spin(self):
        while self.running:
            for frame in self.spinner:
                if not self.running:
                    break
                elapsed = time.time() - self._start_time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                sys.stdout.write(
                    f"\r{timestamp} - {frame} {self.message} [{elapsed:.0f}s]"
                )
                sys.stdout.flush()
                time.sleep(0.1)

    def start(self):
        self._start_time = time.time()
        self.running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self, success: bool = True):
        self.running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        if self._start_time:
            elapsed = time.time() - self._start_time
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()
            status = "👍 OK" if success else "❌ FAIL"
            logger.info(f"{status} {self.message} [{elapsed:.0f}s]")
        else:
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()


async def wait_for_model_ready(
    proc,
    port: int,
    backend: str,
    logfilepath: Path,
    displayname: str,
    timeout: int = 300,
    log_start_position: int = 0,
) -> bool:
    READY_SIGNALS = {
        "vllm": [
            "Application startup complete",
            "Uvicorn running on",
        ],
        "llama.cpp": [
            "main: starting the main loop",
            "main: server is listening",
        ],
    }

    use_tcp_fallback = backend == "vllm"
    signals = READY_SIGNALS.get(backend, [])
    deadline = time.time() + timeout
    log_position = log_start_position

    spinner = LoadingSpinner(f"Loading {displayname} - Timeout: {timeout}s")
    spinner.start()

    try:
        while time.time() < deadline:
            try:
                if proc.poll() is not None:
                    returncode = proc.poll()
                    spinner.stop(success=False)
                    logger.error(f"{displayname} exited with code {returncode} during loading")
                    return False
            except Exception as e:
                spinner.stop(success=False)
                logger.error(f"{displayname}: error checking process state: {e}")
                return False

            try:
                with open(logfilepath, "r", errors="replace") as f:
                    f.seek(log_position)
                    new_content = f.read()
                    log_position = f.tell()

                    for line in new_content.splitlines():
                        if line.strip():
                            logger.debug(f"[{displayname}] {line}")
                        for signal in signals:
                            if signal in line:
                                spinner.stop(success=True)
                                logger.info(f"🎉 {displayname} ready: {signal}")
                                await asyncio.sleep(1)
                                return True
            except Exception as e:
                logger.debug(f"Error reading log for {displayname}: {e}")

            if use_tcp_fallback:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(0.5)
                        if s.connect_ex(("127.0.0.1", port)) == 0:
                            spinner.stop(success=True)
                            logger.info(f"{displayname}:{port} ready")
                            return True
                except Exception:
                    pass

            await asyncio.sleep(1)

        spinner.stop(success=False)
        logger.error(f"Timeout {timeout}s waiting for {displayname}")
        return False

    except Exception:
        spinner.stop(success=False)
        raise


async def ensure_physical_model(physicalname: str, logicalname: Optional[str] = None):
    if physicalname not in PHYSICAL_MODELS:
        raise RuntimeError(f"Model {physicalname} not configured")

    cfg = PHYSICAL_MODELS[physicalname]
    backend = cfg.get("backend", "vllm")
    displayname = logicalname or physicalname

    with global_lock:
        if physicalname in active_servers:
            proc = active_servers[physicalname]["process"]
            if proc.poll() is None:
                port = active_servers[physicalname]["port"]
                server_idle_time[physicalname] = time.time()
                logger.debug(f"♻️  Reusing {displayname}:{port}")
                return port

    NEEDGB = get_model_vram_need(cfg, physicalname)
    logger.info(f"🧮 {displayname} needs {NEEDGB:.1f}GB VRAM")

    # Tensor parallelism (TP) vai a disciplina da VRAM check
    tp_size = int(cfg.get("tensor_parallel", "1"))
    if tp_size > 1:
        logger.info(f"🔄 Tensor parallel={tp_size} - using total VRAM")

    maxretries = 3  # Reduzido drasticamente - não tem sentido tentar 25x
    no_progress_count = 0
    last_total_gb = 0.0

    for attempt in range(maxretries):
        gpus = get_free_gpu_memory()
        max_free_gb = max((g["free_gb"] for g in gpus), default=0.0)
        total_free_gb = sum(g["free_gb"] for g in gpus)  # FIX: agora é float, não generator
        logger.info(
            f"📊 VRAM max single: {max_free_gb:.1f}GB / total: {total_free_gb:.1f}GB - "
            f"Attempt {attempt + 1}/{maxretries} (need {NEEDGB:.1f}GB)"
        )
        # Compare with total VRAM for TP models, single GPU for non-TP models
        available_gb = total_free_gb if tp_size > 1 else max_free_gb
        if available_gb >= NEEDGB:
            logger.info(f"✅ VRAM sufficient ({available_gb:.1f}GB)")
            break

        with global_lock:
            names_to_unload = [n for n in active_servers if n != physicalname]

        for name in names_to_unload:
            logger.info(f"📤 Unloading {name} (dynamic swap)")
            shutdown_server(name, "swap-dynamic", fast=True)
            last_total_gb = total_free_gb

        kill_vram_fast()

        gpus = get_free_gpu_memory()
        new_total_gb = sum(g["free_gb"] for g in gpus)
        if new_total_gb <= last_total_gb + 0.5:  # Sem progresso significativo
            no_progress_count += 1
            if no_progress_count >= 2:
                # Logging detalhado dos processos usando VRAM
                gpu_procs = list_gpu_processes()
                active_procs = [p for p in gpu_procs if p["memory_mb"] > 100]
                if active_procs:
                    logger.error("🚨 Processes using VRAM:")
                    for p in sorted(active_procs, key=lambda x: x["memory_mb"], reverse=True)[:5]:
                        is_allama = any(p["pid"] == s["process"].pid for s in active_servers.values())
                        marker = "ALLAMA" if is_allama else "external"
                        logger.error(f"  PID {p['pid']} ({p['name']}): {p['memory_mb']//1024:.0f}GB [{marker}]")
                logger.error(
                    f"⚠️  Failed to free enough VRAM. "
                    f"Required: {NEEDGB:.1f}GB, available: {new_total_gb:.1f}GB."
                )
                raise RuntimeError(
                    f"Not enough VRAM to load {displayname}. "
                    f"Model needs {NEEDGB:.1f}GB but system only has {new_total_gb:.1f}GB free."
                )
        last_total_gb = new_total_gb
        await asyncio.sleep(5)

    await asyncio.sleep(2)
    gpus = get_free_gpu_memory()
    available_final = sum(g["free_gb"] for g in gpus) if tp_size > 1 else max((g["free_gb"] for g in gpus), default=0.0)
    if available_final < NEEDGB:
        logger.warning(
            f"⚠️  VRAM insuficiente: {available_final:.1f}GB < {NEEDGB:.1f}GB, "
            f"continuando e esperando que o backend gerencie memória."
        )

    logger.info(f"⏳ Loading {displayname} ({backend})")

    if backend == "vllm":
        cmd, port, gpu_id = build_vllm_cmd(physicalname)
    else:
        cmd, port, gpu_id = build_llama_cmd(physicalname)

    logfilepath = ALLAMA_LOG_DIR / f"{physicalname}.log"
    logfile = None
    try:
        logfile = open(logfilepath, "a")
        log_start_position = logfile.tell()
    except Exception as e:
        logger.error(f"Failed to open log file {logfilepath}: {e}")
        raise

    subprocess_env = os.environ.copy()
    # Bug fix: CUDA_VISIBLE_DEVICES quebra tensor_parallelism
    # Para vLLM com TP>1, NÃO definir CVD - deixe ele ver todas as GPUs
    # Para llama.cpp com -ngl -1, também não é necessário - ele usa todas as GPUs
    # Só definimos CVD explicitamente para llama.cpp single-GPU específicas
    if backend == "llama.cpp" and int(cfg.get("tensor_parallel", "1")) == 1:
        subprocess_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # else: leave system GPU discovery untouched

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            env=subprocess_env,
        )

        with global_lock:
            active_servers[physicalname] = {
                "process": proc,
                "pid": proc.pid,
                "port": port,
                "backend": backend,
                "logfile": logfilepath,
            }
            server_idle_time[physicalname] = time.time()

        logger.info(f"✅ Process started PID {proc.pid} on port {port}")
        logger.info(f"📄 Log: tail -f {PATH_TO_ALLAMA}/{logfilepath}")

        ready = await wait_for_model_ready(
            proc, port, backend, logfilepath, displayname,
            timeout=300, log_start_position=log_start_position,
        )

        if not ready:
            returncode = proc.poll()
            if returncode is not None:
                logger.error(f"{displayname}:{port} exited with code {returncode}")
                raise RuntimeError(f"{displayname} startup failed (code {returncode})")
            raise RuntimeError(f"{displayname} not ready after 300s")

        logger.info(f"🚀 {displayname} loaded and ready")
        return port
    finally:
        if logfile and not logfile.closed:
            logfile.close()


_health_monitor_running = threading.Event()
_health_monitor_thread: Optional[threading.Thread] = None


def health_monitor():
    _health_monitor_running.set()
    logger.info("🏥 Health monitor started")
    try:
        while _health_monitor_running.is_set():
            try:
                now = time.time()
                to_unload: list[str] = []

                with global_lock:
                    for physical_name, server in list(active_servers.items()):
                        proc = server["process"]
                        port = server["port"]

                        if proc.poll() is not None:
                            logger.error(f"💥 {physical_name}:{port} crashed")
                            active_servers.pop(physical_name, None)
                            server_idle_time.pop(physical_name, None)
                            continue

                        idle = now - server_idle_time.get(physical_name, 0)
                        if idle > KEEP_ALIVE_SECONDS:
                            logger.info(f"⏰ {physical_name} idle {idle:.0f}s, unloading")
                            to_unload.append(physical_name)

                for physical_name in to_unload:
                    # Bug fix: não usar with global_lock aqui - shutdown_server já adquire o lock internamente
                    if physical_name in active_servers:
                        shutdown_server(physical_name, reason="idle", fast=True)

                # Health monitor can now be woken up on shutdown
                _health_monitor_running.wait(HEALTH_CHECK_INTERVAL)
            except Exception as e:
                logger.error(f"Error in monitor: {e}")
                _health_monitor_running.wait(10)
    finally:
        _health_monitor_running.clear()


# ==============================================================================
# HTTP CLIENT POOL
# ==============================================================================
httpx_client: Optional[httpx.AsyncClient] = None


async def get_http_client():
    global httpx_client
    if httpx_client is None or httpx_client.is_closed:
        httpx_client = httpx.AsyncClient(
            timeout=600.0,
            headers={"Authorization": "Bearer dummy"},
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return httpx_client


async def close_http_client():
    global httpx_client
    if httpx_client is not None and not httpx_client.is_closed:
        await httpx_client.aclose()
        httpx_client = None


# ==============================================================================
# API
# ==============================================================================
async def lifespan(app: FastAPI):
    """Lifespan handler for startup/shutdown events (modern FastAPI alternative to on_event)."""
    yield
    logger.info("🛑 Shutting down Allama...")
    await close_http_client()

app = FastAPI(title="Allama - LLM API", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: dict = Body(...)):
    model_name = body.get("model", "")
    client_host = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    logger.info(f"📤 [HTTP] {request.method} {request.url.path} from {client_host} (🖥️  {format_user_agent(user_agent)})")

    if model_name not in LOGICAL_MODELS:
        return JSONResponse(
            status_code=404,
            content={"error": f"Model '{model_name}' not found"},
        )

    if "messages" in body and MAX_MESSAGES > 0:
        body["messages"] = body["messages"][-MAX_MESSAGES:]

    logical_cfg = LOGICAL_MODELS[model_name]
    physical_name = logical_cfg["physical"]
    port = await ensure_physical_model(physical_name, model_name)

    cfg = PHYSICAL_MODELS[physical_name]
    backend = cfg.get("backend", "vllm")
    if backend == "vllm":
        body["model"] = cfg["path"]
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
    else:
        body["model"] = cfg["model"]
        url = f"http://127.0.0.1:{port}/chat/completions"

    sampling = logical_cfg.get("sampling", {})
    for key in [
        "temperature", "top_p", "top_k", "min_p",
        "presence_penalty", "repetition_penalty",
    ]:
        if key in sampling:
            if key not in body or body[key] is None:
                body[key] = sampling[key]
        else:
            body.pop(key, None)

    if "instruct" in model_name.lower():
        body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False

    logger.debug(f"{model_name} -> {physical_name}:{port} ({backend})")

    if "messages" not in body or not body["messages"]:
        logger.warning("⚠️  Empty messages request")
        if body.get("stream", False):
            async def empty_stream():
                yield "data: [DONE]\n\n"
            return StreamingResponse(empty_stream(), media_type="text/event-stream")
        return JSONResponse(status_code=200, content={"choices": [{"message": {"content": ""}}]})

    client = await get_http_client()
    try:
        if body.get("stream", False):
            async def generate():
                try:
                    async with client.stream("POST", url, json=body) as response:
                        if response.status_code != 200:
                            logger.error(f"Backend returned {response.status_code}")
                            yield 'data: {"error": "Backend error"}\n\n'
                            return
                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line:
                                continue
                            if line.startswith("data: "):
                                yield line + "\n\n"
                            elif line == "[DONE]":
                                yield "data: [DONE]\n\n"
                                break
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            resp = await client.post(url, json=body)
            if resp.status_code == 400:
                logger.warning("vLLM 400 - returning empty response")
                return JSONResponse(
                    status_code=200,
                    content={"choices": [{"message": {"content": ""}}]},
                )
            logger.debug("Output sent")
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except httpx.ConnectError as e:
        logger.warning(f"Connection failed: {e}")
        return JSONResponse(
            status_code=503, content={"error": "Model server unavailable"}
        )
    except Exception as e:
        logger.error(f"Request error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/v1/messages")
async def messages(request: Request, body: dict = Body(...)):
    model_name = body.get("model", "")
    request_id = request.headers.get("x-request-id", "unknown")
    client_host = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    logger.info(f"📤 [HTTP] {request.method} {request.url.path} from {client_host} (🖥️  {format_user_agent(user_agent)})")

    if model_name not in LOGICAL_MODELS:
        # Check if we can auto-switch to a loaded model
        with global_lock:
            loaded_vllm = [
                name
                for name, srv in active_servers.items()
                if srv.get("backend") == "vllm" and srv.get("process") and srv["process"].poll() is None
            ]
        if loaded_vllm:
            physical = loaded_vllm[0]
            model_name = next(
                (k for k, v in LOGICAL_MODELS.items() if v["physical"] == physical),
                None,
            )
            if model_name and model_name != body.get("model"):
                logger.info(f"🔄 Auto-switch: {body.get('model')} -> {model_name} (using loaded {physical})")
        if not model_name or model_name not in LOGICAL_MODELS:
            return JSONResponse(
                status_code=404,
                content={"error": f"Model '{body.get('model')}' not found"},
            )

    logical_cfg = LOGICAL_MODELS[model_name]
    physical_name = logical_cfg["physical"]
    cfg = PHYSICAL_MODELS[physical_name]

    if cfg.get("backend", "vllm") != "vllm":
        return JSONResponse(
            status_code=400,
            content={"error": "Anthropic Messages API requires vllm backend"},
        )

    # Check if model is being switched
    current_loaded = None
    with global_lock:
        for name, srv in active_servers.items():
            if srv.get("backend") == "vllm" and srv.get("process") and srv["process"].poll() is None:
                current_loaded = name
                break

    if current_loaded and current_loaded != physical_name:
        logger.info(f"🔄 Model switch: {current_loaded} -> {physical_name} ({model_name})")

    port = await ensure_physical_model(physical_name, model_name)
    server_idle_time[physical_name] = time.time()

    max_model_len = int(cfg.get("max_model_len", "40960"))
    max_output_cap = max_model_len // 4
    requested = body.get("max_tokens", max_output_cap)
    if requested > max_output_cap:
        logger.info(f"⚠️  max_tokens {requested} -> {max_output_cap}")
        body["max_tokens"] = max_output_cap

    body["model"] = cfg["path"]
    url = f"http://127.0.0.1:{port}/v1/messages"

    logger.debug(f"{model_name} -> {physical_name}:{port}")

    client = await get_http_client()
    try:
        if body.get("stream", False):
            async def generate():
                try:
                    async with client.stream("POST", url, json=body) as response:
                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line:
                                continue
                            yield line + "\n\n"
                except Exception as e:
                    logger.error(f"Stream error: {e}")

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            resp = await client.post(url, json=body)
            return JSONResponse(content=resp.json(), status_code=resp.status_code)

    except httpx.ConnectError as e:
        logger.warning(f"Connection failed: {e}")
        return JSONResponse(status_code=503, content={"error": "Backend unavailable"})
    except Exception as e:
        logger.error(f"Claude Code error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{"id": k, "object": "model"} for k in LOGICAL_MODELS],
    }


@app.get("/health")
async def health():
    with global_lock:
        active = len(active_servers)
    return {
        "status": "healthy",
        "active_servers": active,
        "running": running,
    }


if RICH_AVAILABLE:
    from collections import OrderedDict

    # Palette
    W = 120
    C_YELLOW = "#f5c518"
    C_WHITE = "#fafafa"
    C_TEAL = "#5fd7af"
    C_TERRA = "#d7875f"
    C_BERRY = "#9b6dff"
    C_DIM = "#2d2d4e"
    C_MARRON = "#ed333b"

    console = Console(width=W)

    def make_table(panels, cols=3):
        tbl = Table.grid(expand=True, padding=0)
        for _ in range(cols):
            tbl.add_column(ratio=1)
        row = []
        for p in panels:
            row.append(p)
            if len(row) == cols:
                tbl.add_row(*row)
                row = []
        if row:
            row += [Text("")] * (cols - len(row))
            tbl.add_row(*row)
        return tbl

    banner_txt = Text(justify="center")
    banner_txt.append("🦙 ALLAMA 🦙\n", style=f"bold {C_YELLOW}")
    banner_txt.append(f"http://127.0.0.1:{ALLAMA_PORT}", style=f"italic {C_WHITE}")
    console.print(Panel(
        Align(banner_txt, align="center", vertical="top"),
        box=box.HORIZONTALS,
        border_style=C_YELLOW,
        padding=(0, 2),
        width=W,
    ))

    console.print(Panel(
        Align(Text("🔧 CONFIGURATION", style=f"bold {C_WHITE}"), align="center"),
        box=box.ROUNDED,
        border_style=C_DIM,
        padding=(0, 1),
        width=W,
        expand=False,
    ), justify="center")

    config_items = [
        ("📦 Physical Models", f"[bold {C_MARRON}]{len(PHYSICAL_MODELS)}[/]"),
        ("🧠 Logical Models", f"[bold {C_MARRON}]{len(LOGICAL_MODELS)}[/]"),
        ("⏰Keep Alive", f"[bold {C_MARRON}]{KEEP_ALIVE_SECONDS}s[/]"),
    ]
    cfg_panels = [
        Panel(
            Align(Text.from_markup(val), align="center", vertical="top"),
            title=f"[dim {C_WHITE}]{lbl}[/]",
            box=box.ROUNDED,
            border_style=C_WHITE,
            padding=(0, 1),
            expand=True,
        )
        for lbl, val in config_items
    ]
    console.print(make_table(cfg_panels, cols=3))
    console.print()

    console.print(Panel(
        Align(Text("📦 PHYSICAL MODELS", style=f"bold {C_TEAL}"), align="center"),
        box=box.ROUNDED,
        border_style=C_DIM,
        padding=(0, 1),
        width=W,
        expand=False,
    ), justify="center")

    phys_panels = []
    for name, cfg in PHYSICAL_MODELS.items():
        backend = cfg.get("backend", "vllm")
        if backend == "vllm":
            bcolor, icon, title_c = C_TEAL, "🔰 vLLM", C_TEAL
        else:
            bcolor, icon, title_c = C_TERRA, "🔆 llama.cpp", C_TERRA
        phys_panels.append(Panel(
            Align(Text(name, style=f"bold {C_WHITE}", justify="center"), align="center", vertical="top"),
            title=f"[dim {title_c}]{icon}[/]",
            box=box.ROUNDED,
            border_style=bcolor,
            padding=(0, 1),
            expand=True,
        ))
    console.print(make_table(phys_panels, cols=3))
    console.print()

    console.print(Panel(
        Align(Text("🧠 LOGICAL MODELS", style=f"bold {C_BERRY}"), align="center"),
        box=box.ROUNDED,
        border_style=C_DIM,
        padding=(0, 1),
        width=W,
        expand=False,
    ), justify="center")

    grouped: OrderedDict = OrderedDict()
    for log_name, log_cfg in LOGICAL_MODELS.items():
        phys = log_cfg["physical"]
        grouped.setdefault(phys, []).append(log_name)

    log_panels = []
    for phys, names in grouped.items():
        body = Text()
        for i, n in enumerate(names):
            body.append(f"  > {n}", style=C_WHITE)
            if i < len(names) - 1:
                body.append("\n")
        log_panels.append(Panel(
            Align(body, align="left", vertical="top"),
            title=f"[bold {C_BERRY}]{phys}[/]",
            box=box.ROUNDED,
            border_style=C_BERRY,
            padding=(0, 1),
            expand=True,
        ))
    console.print(make_table(log_panels, cols=3))
    console.print()

else:
    logger.info("=" * 60)
    logger.info("Allama Started")
    logger.info("=" * 60)
    logger.info("Backend logs:")
    for name in PHYSICAL_MODELS:
        logger.info(
            f"   - {name} - tail -f {PATH_TO_ALLAMA}{ALLAMA_LOG_DIR}/{name}.log"
        )
    logger.info("=" * 60)
    logger.info(
        f"Models configured: {len(PHYSICAL_MODELS)} physical, {len(LOGICAL_MODELS)} logical"
    )
    logger.info(f"Keep-alive: {KEEP_ALIVE_SECONDS}s")
    logger.info(
        f"Auto swap: {'ON' if AUTO_SWAP_ENABLED else 'OFF'}"
    )
    logger.info(f"API: http://127.0.0.1:{ALLAMA_PORT}")
    logger.info("=" * 60)


def main():
    global running, _health_monitor_thread

    def signal_handler(sig: int, frame: Any) -> None:
        global running
        if not running:
            logger.critical("Second signal received, force kill")
            # Stop health monitor first - use .clear() to break the is_set() loop
            _health_monitor_running.clear()
            time.sleep(0.5)
            os.kill(os.getpid(), signal.SIGKILL)
            return
        logger.info("🛑 Shutdown requested...")
        running = False
        # Signal health monitor to stop - use .clear() to break the is_set() loop
        _health_monitor_running.clear()
        with global_lock:
            for name in list(active_servers.keys()):
                server = active_servers.get(name)
                if server and server.get("process"):
                    pid = server["process"].pid
                    kill_process_tree(pid)  # Kill whole tree including vLLM workers
                    logger.info(f"🔥 Killed process tree for {name} (PID {pid})")
        # Wait for health monitor thread to finish
        if _health_monitor_thread:
            _health_monitor_thread.join(timeout=5)
        time.sleep(1)
        os._exit(0)

    # Signal handler uses .clear() to stop the monitor - loop is "while is_set()"
    _health_monitor_thread = threading.Thread(target=health_monitor, daemon=True)
    _health_monitor_thread.start()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    uvicorn.run(app, host="127.0.0.1", port=ALLAMA_PORT)


if __name__ == "__main__":
    main()
