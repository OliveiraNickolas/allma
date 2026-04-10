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

from core.config import logger, PHYSICAL_MODELS, LLAMA_CPP_PATH, ALLAMA_LOG_DIR
import core.state as state

# ==============================================================================
# PID REGISTRY — persist backend PIDs to disk for orphan recovery
# ==============================================================================
_PID_REGISTRY = ALLAMA_LOG_DIR / "backends.json"


def _load_registry() -> dict:
    try:
        return json.loads(_PID_REGISTRY.read_text()) if _PID_REGISTRY.exists() else {}
    except Exception:
        return {}


def _save_registry(data: dict):
    ALLAMA_LOG_DIR.mkdir(parents=True, exist_ok=True)
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
    """Kill backend processes left over from a previous allama session."""
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
                logger.info(f"🧹 Killing orphaned backend: {name} (PID {pid}, {info.get('backend')})")
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
        from core.gpu import get_free_gpu_memory
        gpus = get_free_gpu_memory()
        freegb = sum(g["free_gb"] for g in gpus)
        logger.info(f"🧹 Cleaned up {killed} orphaned backend(s). VRAM free: {freegb:.1f}GB")
from core.gpu import (
    find_optimal_tp_and_gpus,
    get_best_gpu,
    get_free_gpu_memory,
    get_all_gpus,
)


# ==============================================================================
# COMMAND BUILDERS
# ==============================================================================
def build_vllm_cmd(physical_name: str, skip_gpu: int | None = None) -> tuple[list, int, int]:
    """Build vLLM command with GPU and tensor parallelism configuration."""
    cfg = PHYSICAL_MODELS[physical_name]

    port = state.get_next_vllm_port()
    while not state.is_port_free(port):
        port = state.get_next_vllm_port()

    tp_size = int(cfg.get("tensor_parallel", "1"))
    adj_tp, selected_gpu = find_optimal_tp_and_gpus(physical_name, skip_gpu)
    if adj_tp != tp_size:
        logger.info(f"🔄 {physical_name}: Adjusting TP from {tp_size} to {adj_tp} for GPU fit")
    tp_size = adj_tp

    cmd = [
        "vllm", "serve", cfg["path"],
        "--tokenizer", cfg["tokenizer"],
        "--tensor-parallel-size", str(tp_size),
        "--gpu-memory-utilization", str(cfg.get("gpu_memory_utilization", "0.90")),
        "--max-model-len", str(cfg["max_model_len"]),
        "--max-num-seqs", str(cfg.get("max_num_seqs", "8")),
        "--generation-config", "vllm",
        "--port", str(port),
        "--host", "127.0.0.1",
        "--api-key", "dummy",
    ]
    cmd.extend(cfg.get("extra_args", []))
    state.gpu_allocation[physical_name] = selected_gpu
    return cmd, port, selected_gpu


def build_llama_cmd(physical_name: str) -> tuple[list, int, int]:
    """Build llama.cpp command with GPU configuration."""
    cfg = PHYSICAL_MODELS[physical_name]

    port = state.get_next_llama_port()
    while not state.is_port_free(port):
        port = state.get_next_llama_port()

    gpu_id = state.gpu_allocation.get(physical_name)
    if gpu_id is None:
        gpu_id = get_best_gpu()
        state.gpu_allocation[physical_name] = gpu_id
    logger.info(f"🎯 {physical_name} -> GPU {gpu_id}")

    cmd = [
        LLAMA_CPP_PATH,
        "-m", cfg["model"],
        "--host", "127.0.0.1",
        "--port", str(port),
        "-t", str(cfg.get("n_threads", "16")),
        "-c", str(cfg.get("n_ctx", "40960")),
        "-b", str(cfg.get("n_batch", "1024")),
        "-ngl", str(cfg.get("n_gpu_layers", "-1")),
    ]
    if cfg.get("mmproj") and os.path.exists(cfg["mmproj"]):
        cmd.extend(["--mmproj", cfg["mmproj"]])
    if cfg.get("chat_template_file") and os.path.exists(cfg["chat_template_file"]):
        cmd.extend(["--chat-template-file", cfg["chat_template_file"]])
    cmd.extend(cfg.get("extra_args", []))
    return cmd, port, gpu_id


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


def shutdown_server(physicalname: str, reason: str = "user", fast: bool = False):
    proc = None
    port = None
    backend = None

    with state.global_lock:
        if physicalname not in state.active_servers:
            logger.warning(f"{physicalname} not active")
            return
        server = state.active_servers[physicalname]
        proc = server["process"]
        pid = proc.pid
        port = server["port"]
        backend = server.get("backend", "unknown")

    remove_backend_pid(physicalname)
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
        time.sleep(1 if fast else 3)

    with state.global_lock:
        state.active_servers.pop(physicalname, None)
        state.server_idle_time.pop(physicalname, None)
        state.gpu_allocation.pop(physicalname, None)

    gpus = get_free_gpu_memory()
    freegb = sum(g["free_gb"] for g in gpus)
    logger.info(f"🗑️  {physicalname} unloaded. VRAM free: {freegb:.1f}GB")


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
    """Kill only processes managed by Allama (active_servers)."""
    logger.info("🔨 Aggressive VRAM shutdown initiated...")
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

            if pid == state.ALLAMA_PID:
                logger.debug(f"Skipping ALLAMA itself (PID {state.ALLAMA_PID})")
                continue
            if pid not in known_pids:
                logger.debug(f"Skipping external process {pid} ({procname}) - not managed by Allama")
                continue

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
        logger.error(traceback.format_exc())
