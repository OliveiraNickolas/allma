"""
Process management: build commands, kill, shutdown, VRAM cleanup, PID registry.
"""
import json
import os
import signal
import subprocess
import time
import traceback
from typing import Any, Dict, Optional

import psutil

from core.config import logger, BASE_MODELS, LLAMA_CPP_PATH, LLAMA_CPP_PYTHON_BACKEND, VLLM_PATH, ALLMA_LOG_DIR
from core.model_detect import get_auto_extra_args, get_auto_max_model_len, get_family_label
from core.gpu import find_optimal_tp_and_gpus, get_best_gpu, get_free_gpu_memory
import core.state as state

# ==============================================================================
# PID REGISTRY — persist backend PIDs to disk for orphan recovery
# ==============================================================================
_PID_REGISTRY = ALLMA_LOG_DIR / "backends.json"


def _load_registry() -> dict:
    try:
        return json.loads(_PID_REGISTRY.read_text()) if _PID_REGISTRY.exists() else {}
    except Exception:
        return {}


def _save_registry(data: dict):
    ALLMA_LOG_DIR.mkdir(parents=True, exist_ok=True)
    _PID_REGISTRY.write_text(json.dumps(data, indent=2))


def save_backend_pid(name: str, pid: int, port: int, backend: str):
    """Register a backend process on disk."""
    reg = _load_registry()
    reg[name] = {"pid": pid, "port": port, "backend": backend}
    _save_registry(reg)


def remove_backend_pid(name: str):
    """Remove a backend from the PID registry."""
    reg = _load_registry()
    if name in reg:
        del reg[name]
        _save_registry(reg)


def clear_backend_registry():
    """Clear all entries from the PID registry."""
    _save_registry({})


def cleanup_orphaned_backends():
    """Kill backend processes left over from a previous allma session."""
    reg = _load_registry()
    if not reg:
        return

    killed = 0
    for name, info in list(reg.items()):
        pid = info.get("pid")
        if not pid:
            continue
        try:
            proc = psutil.Process(pid)
            cmdline = " ".join(proc.cmdline())
            # Verify it's actually a backend process (not a random PID reuse)
            is_vllm = "vllm" in cmdline and "serve" in cmdline
            is_llama = "llama-server" in cmdline or "llama.cpp" in cmdline
            if is_vllm or is_llama:
                logger.info(f"Killing orphaned backend: {name} (PID {pid}, {info.get('backend')})")
                kill_process_tree(pid, timeout=3)
                killed += 1
            else:
                logger.debug(f"PID {pid} is no longer a backend process, skipping")
        except psutil.NoSuchProcess:
            logger.debug(f"Orphan PID {pid} ({name}) already dead")
        except Exception as e:
            logger.warning(f"Error cleaning up {name} (PID {pid}): {e}")

    # Clear the registry
    _save_registry({})

    if killed:
        time.sleep(2)
        gpus = get_free_gpu_memory()
        freegb = sum(g["free_gb"] for g in gpus)
        logger.info(f"Cleaned up {killed} orphaned backend(s). VRAM free: {freegb:.1f}GB")


# ==============================================================================
# COMMAND BUILDERS
# ==============================================================================
def build_vllm_cmd(base_name: str, skip_gpu: int | None = None, gpu_id: int | None = None) -> tuple[list, int, int]:
    """Build vLLM command with GPU and tensor parallelism configuration."""
    cfg = BASE_MODELS[base_name]

    port = state.get_next_vllm_port()
    while not state.is_port_free(port):
        port = state.get_next_vllm_port()

    # Priority: explicit arg > config gpu_id > auto-select
    if gpu_id is not None:
        selected_gpu = gpu_id
        tp_size = int(cfg.get("tensor_parallel", "1"))
        logger.info(f"{base_name}: Using explicit GPU {gpu_id} (TP={tp_size})")
    else:
        try:
            cfg_pin = int(cfg.get("gpu_id", -1))
        except (TypeError, ValueError):
            cfg_pin = -1
        if cfg_pin >= 0:
            selected_gpu = cfg_pin
            tp_size = int(cfg.get("tensor_parallel", "1"))
            logger.info(f"{base_name}: Pinned to GPU {cfg_pin} (config, TP={tp_size})")
        else:
            adj_tp, selected_gpu = find_optimal_tp_and_gpus(base_name, skip_gpu)
            tp_size = adj_tp

    max_model_len = get_auto_max_model_len(cfg)
    extra_args = get_auto_extra_args(cfg, "vllm")

    if "extra_args" not in cfg:
        family = get_family_label(cfg["path"])
        logger.info(f"{base_name}: auto-detected family '{family}', applying preset args")

    cmd = [
        VLLM_PATH, "serve", cfg["path"],
        "--tokenizer", cfg.get("tokenizer", cfg["path"]),
        "--tensor-parallel-size", str(tp_size),
        "--gpu-memory-utilization", str(cfg.get("gpu_memory_utilization", "0.90")),
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(cfg.get("max_num_seqs", "8")),
        "--generation-config", "vllm",
        "--port", str(port),
        "--host", "127.0.0.1",
        "--api-key", "dummy",
    ]

    if "max_num_batched_tokens" in cfg:
        cmd += ["--max-num-batched-tokens", str(cfg["max_num_batched_tokens"])]
    cmd.extend(extra_args)
    state.gpu_allocation[base_name] = selected_gpu
    return cmd, port, selected_gpu


def build_llama_cmd(base_name: str, gpu_id: int | None = None) -> tuple[list, int, int]:
    """Build llama.cpp command with GPU configuration."""
    cfg = BASE_MODELS[base_name]

    port = state.get_next_llama_port()
    while not state.is_port_free(port):
        port = state.get_next_llama_port()

    # Priority: explicit arg > config gpu_id > cached allocation > auto-select
    if gpu_id is not None:
        state.gpu_allocation[base_name] = gpu_id
        logger.info(f"{base_name} → GPU {gpu_id} (explicit)")
    else:
        try:
            cfg_pin = int(cfg.get("gpu_id", -1))
        except (TypeError, ValueError):
            cfg_pin = -1
        if cfg_pin >= 0:
            gpu_id = cfg_pin
            state.gpu_allocation[base_name] = gpu_id
            logger.info(f"{base_name} → GPU {gpu_id} (pinned from config)")
        else:
            gpu_id = state.gpu_allocation.get(base_name)
            if gpu_id is None:
                gpu_id = get_best_gpu()
                state.gpu_allocation[base_name] = gpu_id
            logger.info(f"{base_name} → GPU {gpu_id}")

    n_batch = cfg.get("n_batch", "1024")
    n_ctx = cfg.get("n_ctx", "40960")
    extra_args = get_auto_extra_args(cfg, "llama.cpp")

    if "extra_args" not in cfg:
        family = get_family_label(cfg.get("model", ""))
        logger.info(f"{base_name}: auto-detected family '{family}', applying preset args")

    if LLAMA_CPP_PYTHON_BACKEND:
        cmd = _build_llama_cpp_python_cmd(cfg, port, n_ctx, n_batch, gpu_id, extra_args)
    else:
        cmd = [
            LLAMA_CPP_PATH,
            "-m", cfg["model"],
            "--host", "127.0.0.1",
            "--port", str(port),
            "-t", str(cfg.get("n_threads", "16")),
            "-c", str(n_ctx),
            "-b", str(n_batch),
            "-ngl", str(cfg.get("n_gpu_layers", "-1")),
        ]
        if cfg.get("mmproj") and os.path.exists(cfg["mmproj"]):
            cmd.extend(["--mmproj", cfg["mmproj"]])
        if cfg.get("chat_template_file") and os.path.exists(cfg["chat_template_file"]):
            cmd.extend(["--chat-template-file", cfg["chat_template_file"]])
        cmd.extend(extra_args)
    return cmd, port, gpu_id


def _build_llama_cpp_python_cmd(
    cfg: dict, port: int, n_ctx: str, n_batch: str, gpu_id: int, extra_args: list
) -> list:
    """Build command for llama-cpp-python server (python -m llama_cpp.server).

    Translates llama-server CLI flags to llama-cpp-python settings flags.
    Some advanced features (KV quantization, jinja templates, flash-attn) are
    not supported by llama-cpp-python and will be silently skipped.
    """
    import sys as _sys
    cmd = [
        _sys.executable, "-m", "llama_cpp.server",
        "--model", cfg["model"],
        "--host", "127.0.0.1",
        "--port", str(port),
        "--n_ctx", str(n_ctx),
        "--n_batch", str(n_batch),
        "--n_threads", str(cfg.get("n_threads", "16")),
        "--n_gpu_layers", str(cfg.get("n_gpu_layers", "-1")),
    ]
    # Map subset of extra_args that llama-cpp-python supports
    _SUPPORTED = {"--chat-format", "--rope-scaling", "--rope-freq-base", "--rope-freq-scale"}
    skip_next = False
    for i, arg in enumerate(extra_args):
        if skip_next:
            skip_next = False
            continue
        if arg in _SUPPORTED:
            cmd.append(arg)
            skip_next = True  # include the value too
    logger.warning(
        f"Using llama-cpp-python server (fallback). Some features are unavailable: "
        "KV cache quantization, flash-attn, jinja templates, mmproj vision. "
        "Install llama-server for full support: bash scripts/install-llama-cpp.sh"
    )
    return cmd


# ==============================================================================
# PROCESS MANAGEMENT
# ==============================================================================
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


def shutdown_server(basename: str, reason: str = "user", fast: bool = False):
    proc = None
    port = None
    backend = None

    with state.global_lock:
        if basename not in state.active_servers:
            logger.warning(f"{basename} not active")
            return
        server = state.active_servers[basename]
        proc = server["process"]
        pid = proc.pid
        port = server["port"]
        backend = server.get("backend", "unknown")

    remove_backend_pid(basename)
    logger.info(f"Unload {basename}:{port} ({reason})")

    if proc and proc.poll() is None:
        logger.info(f"Killing PID {pid} ({backend})")
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            kill_process_tree(pid, timeout=3 if fast else 5)
        time.sleep(1 if fast else 3)

    with state.global_lock:
        state.active_servers.pop(basename, None)
        state.server_idle_time.pop(basename, None)
        state.gpu_allocation.pop(basename, None)

    gpus = get_free_gpu_memory()
    freegb = sum(g["free_gb"] for g in gpus)
    logger.info(f" {basename} unloaded. VRAM free: {freegb:.1f}GB")


def list_gpu_processes(gpu_ids: Optional[list[int]] = None) -> list[Dict[str, Any]]:
    """List processes using VRAM on one or more GPUs."""
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


def kill_vram_fast():
    """Kill only processes managed by Allma (active_servers)."""
    logger.info("Aggressive VRAM shutdown initiated...")
    pids_killed = []

    known_pids = set()
    with state.global_lock:
        for name, server in state.active_servers.items():
            known_pids.add(server["pid"])

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            logger.error(f"nvidia-smi failed: {result.stderr}")
            return

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(",", 1)
            if len(parts) < 2:
                continue
            pid = int(parts[0].strip())
            procname = parts[1].strip().lower()

            if pid == state.ALLMA_PID:
                logger.debug(f"Skipping ALLMA itself (PID {state.ALLMA_PID})")
                continue
            if pid not in known_pids:
                logger.debug(f"Skipping external process {pid} ({procname}) - not managed by Allma")
                continue

            logger.info(f"Killing managed PID {pid} ({procname})")
            try:
                kill_process_tree(pid, timeout=1)
                pids_killed.append(pid)
                logger.info(f"Killed managed PID {pid} {procname}")
            except Exception as e:
                logger.error(f"Error killing {pid}: {e}")

        if pids_killed:
            logger.info(f"Shutdown complete: {len(pids_killed)} ALLMA-managed processes")
            time.sleep(2)
            gpus = get_free_gpu_memory()
            freegb = sum(g["free_gb"] for g in gpus)
            logger.info(f"VRAM Free after shutdown: {freegb:.1f}GB")
        else:
            logger.info("No ALLMA-managed processes to shutdown")
    except Exception as e:
        logger.error(f"Error in killvramfast: {e}")
        logger.error(traceback.format_exc())
