"""
Model loading: spinner, readiness check, ensure_physical_model.
"""
import asyncio
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from core.config import logger, PHYSICAL_MODELS, ALLAMA_LOG_DIR, PATH_TO_ALLAMA
import core.state as state
from core.gpu import get_free_gpu_memory, get_model_vram_need, get_all_gpus
from core.process import (
    build_vllm_cmd,
    build_llama_cmd,
    shutdown_server,
    kill_vram_fast,
    list_gpu_processes,
    save_backend_pid,
)


# ==============================================================================
# LOADING UI
# ==============================================================================
_CLOUDS = (
    "     ▁▂▃▂▁▁         "
    "  ▁▂▃▄▃▂▁▁          "
    "       ▁▁▂▃▃▂▁▁     "
    "   ▁▂▂▃▂▂▁▁         "
    "           ▁▂▃▄▃▂▁▁ "
    "     ▁▂▃▂▁▂▃▂▁      "
    "  ▁▁▂▃▂▁▁           "
    "        ▁▂▃▃▂▁▁     "
) * 4

_SKY = (
    "                    "
    "   ▁▁               "
    "            ▁       "
    "                ▁▁  "
    "      ▁▁▁           "
    "                    "
    "           ▁▁       "
    "    ▁               "
) * 4

_MOUNTAINS = (
    "▁▁▁▂▄▆█▆▄▂▁▁"
    "▁▁▁▂▃▅▇█▇▅▃▂▁▁"
    "▁▁▂▄▆█▆▄▂▁▁"
    "▁▁▁▁▂▄▆█▆▄▂▁▁▁"
    "▁▁▂▃▄▆█▆▄▃▂▁▁"
    "▁▁▁▂▅▇█▇▅▂▁▁"
    "▁▁▂▃▄▅▇█▇▅▄▃▂▁▁"
    "▁▁▁▂▄▅▇█▇▅▄▂▁▁"
) * 4

_WIN = 36


class LoadingSpinner:
    def __init__(self, message: str = "Loading"):
        self.message = message
        self.running = False
        self._thread = None
        self._start_time = None

    # 3-row llama sprite (mirrored from allama_cli)
    _L_HEAD  = "▄▄"
    _L_NECK  = "▓"
    _L_BODY  = ["▓▒▓", "▓▓▒", "▓▒▓", "▒▓▓"]
    _L_POS   = 4
    _L_SPEED = 7

    def _inject(self, cview, sview, nview, tick):
        frame = self._L_BODY[(tick // self._L_SPEED) % len(self._L_BODY)]
        cl, sl, nl = list(cview), list(sview), list(nview)
        for k, ch in enumerate(self._L_HEAD):
            p = self._L_POS + 2 + k
            if p < len(cl): cl[p] = ch
        p = self._L_POS + 2
        if p < len(sl): sl[p] = self._L_NECK
        for k, ch in enumerate(frame):
            p = self._L_POS + k
            if p < len(nl): nl[p] = ch
        return "".join(cl), "".join(sl), "".join(nl)

    def _spin(self):
        ci = si = ni = 0
        last_c = last_s = last_n = 0
        tick = 0
        sys.stdout.write("\n\n")
        while self.running:
            elapsed = time.time() - self._start_time
            cview = (_CLOUDS * 2)[ci % len(_CLOUDS): ci % len(_CLOUDS) + _WIN]
            sview = (_SKY    * 2)[si % len(_SKY):    si % len(_SKY)    + _WIN]
            nview = (_MOUNTAINS * 2)[ni % len(_MOUNTAINS): ni % len(_MOUNTAINS) + _WIN]
            cview, sview, nview = self._inject(cview, sview, nview, tick)
            cloud_line = f"  {cview}"
            sky_line   = f"  {sview}"
            near_line  = f"  {nview}  {self.message}  [{elapsed:.0f}s]"
            sys.stdout.write(f"\033[2A\r{' ' * last_c}\r{cloud_line}\n")
            sys.stdout.write(f"\r{' ' * last_s}\r{sky_line}\n")
            sys.stdout.write(f"\r{' ' * last_n}\r{near_line}")
            sys.stdout.flush()
            last_c = len(cloud_line)
            last_s = len(sky_line)
            last_n = len(near_line)
            time.sleep(0.06)
            tick += 1
            if tick % 5 == 0: ci += 1
            if tick % 3 == 0: si += 1
            if tick % 2 == 0: ni += 1
        sys.stdout.write(f"\r{' ' * last_n}\r")
        sys.stdout.write(f"\033[1A\r{' ' * last_s}\r")
        sys.stdout.write(f"\033[1A\r{' ' * last_c}\r")
        sys.stdout.flush()

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
            status = "OK" if success else "FAIL"
            logger.info(f"{status} {self.message} [{elapsed:.0f}s]")


# ==============================================================================
# READINESS CHECK
# ==============================================================================
async def wait_for_model_ready(
    proc,
    port: int,
    backend: str,
    logfilepath,
    displayname: str,
    timeout: int = 300,
    log_start_position: int = 0,
) -> bool:
    READY_SIGNALS = {
        "vllm": ["Application startup complete", "Uvicorn running on"],
        "llama.cpp": ["main: starting the main loop", "main: server is listening"],
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
                returncode = proc.poll()
                if returncode is not None:
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
                            for expected_signal in signals:
                                if expected_signal in line:
                                    spinner.stop(success=True)
                                    logger.info(f"🎉 {displayname} ready: {expected_signal}")
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


# ==============================================================================
# MODEL LOADING
# ==============================================================================
async def ensure_physical_model(physicalname: str, logicalname: Optional[str] = None):
    if physicalname not in PHYSICAL_MODELS:
        raise RuntimeError(f"Model {physicalname} not configured")

    cfg = PHYSICAL_MODELS[physicalname]
    backend = cfg.get("backend", "vllm")
    displayname = logicalname or physicalname

    with state.global_lock:
        if physicalname in state.active_servers:
            proc = state.active_servers[physicalname]["process"]
            if proc.poll() is None:
                port = state.active_servers[physicalname]["port"]
                state.server_idle_time[physicalname] = time.time()
                logger.debug(f"♻️  Reusing {displayname}:{port}")
                return port

        already_loading = physicalname in state.loading_models
        if not already_loading:
            state.loading_models.add(physicalname)

    if already_loading:
        logger.info(f"⏳ {displayname} already loading, waiting...")
        for _ in range(150):
            await asyncio.sleep(2)
            with state.global_lock:
                if physicalname in state.active_servers:
                    proc = state.active_servers[physicalname]["process"]
                    if proc and proc.poll() is None:
                        port = state.active_servers[physicalname]["port"]
                        state.server_idle_time[physicalname] = time.time()
                        logger.debug(f"♻️  Reusing {displayname}:{port} after wait")
                        return port
                if physicalname not in state.loading_models:
                    # Primary loader finished — check if it succeeded
                    if physicalname in state.active_servers:
                        proc = state.active_servers[physicalname]["process"]
                        if proc and proc.poll() is None:
                            port = state.active_servers[physicalname]["port"]
                            state.server_idle_time[physicalname] = time.time()
                            return port
                    raise RuntimeError(f"{displayname} failed to load")
        raise RuntimeError(f"Loading timeout for {displayname} - load may have stuck")

    success = False
    try:
        port = await _load_model_impl(physicalname, cfg, backend, displayname)
        success = True
        return port
    finally:
        state.loading_models.discard(physicalname)
        if not success:
            with state.global_lock:
                state.active_servers.pop(physicalname, None)
                state.server_idle_time.pop(physicalname, None)


async def _load_model_impl(physicalname: str, cfg: dict, backend: str, displayname: str) -> int:
    """Internal implementation of model loading."""
    NEEDGB = get_model_vram_need(cfg, physicalname)
    logger.info(f"🧮 {displayname} needs {NEEDGB:.1f}GB VRAM")

    # Determine if this model needs more than one GPU can provide.
    # We compare NEEDGB against the usable capacity of the best single GPU
    # (total * gpu_memory_utilization). If it exceeds that, the model will
    # require tensor parallelism across multiple GPUs and we should check
    # total VRAM rather than max single-GPU free VRAM.
    _all_gpus_info = get_all_gpus()
    _gpu_mem_util = float(cfg.get("gpu_memory_utilization", "0.90"))
    _max_single_gpu_usable = max(
        (g["total_gb"] * _gpu_mem_util for g in _all_gpus_info), default=0.0
    )
    needs_multi_gpu = backend == "vllm" and NEEDGB > _max_single_gpu_usable

    if needs_multi_gpu:
        logger.info(
            f"🔄 {displayname}: needs {NEEDGB:.1f}GB > single GPU usable {_max_single_gpu_usable:.1f}GB "
            f"— will use multi-GPU (TP>1)"
        )

    maxretries = 3
    no_progress_count = 0
    last_total_gb = 0.0
    max_free_gb = 0.0

    for attempt in range(maxretries):
        gpus = get_free_gpu_memory()
        max_free_gb = max((g["free_gb"] for g in gpus), default=0.0)
        total_free_gb = sum(g["free_gb"] for g in gpus)
        logger.info(
            f"📊 VRAM max single: {max_free_gb:.1f}GB / total: {total_free_gb:.1f}GB - "
            f"Attempt {attempt + 1}/{maxretries} (need {NEEDGB:.1f}GB)"
        )
        available_gb = total_free_gb if (needs_multi_gpu or backend == "llama.cpp") else max_free_gb
        if available_gb >= NEEDGB:
            logger.info(f"✅ VRAM sufficient ({available_gb:.1f}GB)")
            break

        with state.global_lock:
            names_to_unload = [n for n in state.active_servers if n != physicalname]

        for name in names_to_unload:
            logger.info(f"📤 Unloading {name} (dynamic swap)")
            shutdown_server(name, "swap-dynamic", fast=True)
            last_total_gb = total_free_gb

        kill_vram_fast()

        gpus = get_free_gpu_memory()
        new_total_gb = sum(g["free_gb"] for g in gpus)
        if new_total_gb <= last_total_gb + 0.5:
            no_progress_count += 1
            if no_progress_count >= 2:
                gpu_procs = list_gpu_processes()
                active_procs = [p for p in gpu_procs if p["memory_mb"] > 100]
                if active_procs:
                    logger.error("🚨 Processes using VRAM:")
                    for p in sorted(active_procs, key=lambda x: x["memory_mb"], reverse=True)[:5]:
                        is_allama = any(p["pid"] == s["process"].pid for s in state.active_servers.values())
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
    available_final = (
        sum(g["free_gb"] for g in gpus)
        if (needs_multi_gpu or backend == "llama.cpp")
        else max((g["free_gb"] for g in gpus), default=0.0)
    )
    if available_final < NEEDGB:
        logger.warning(
            f"⚠️  VRAM insuficiente: {available_final:.1f}GB < {NEEDGB:.1f}GB, "
            f"continuando e esperando que o backend gerencie memória."
        )

    logger.info(f"⏳ Loading {displayname} ({backend})")

    max_attempts = 5
    skip_gpu: int | None = None

    for attempt in range(max_attempts):
        logfilepath = ALLAMA_LOG_DIR / f"{physicalname}.log"
        logfile = None
        try:
            logfile = open(logfilepath, "a+")
            log_start_position = logfile.tell()
        except Exception as e:
            logger.error(f"Failed to open log file {logfilepath}: {e}")
            raise

        if backend == "vllm":
            cmd, port, current_gpu_id = build_vllm_cmd(physicalname, skip_gpu=skip_gpu)
        else:
            cmd, port, current_gpu_id = build_llama_cmd(physicalname)

        tp_from_cmd = 1
        try:
            tp_idx = cmd.index("--tensor-parallel-size") + 1
            if tp_idx < len(cmd):
                tp_from_cmd = int(cmd[tp_idx])
        except (ValueError, IndexError):
            pass
        logger.info(f"🎯 Attempt {attempt + 1}/{max_attempts}: {displayname} → GPU {current_gpu_id} (TP={tp_from_cmd})")
        proc = None
        try:
            subprocess_env = os.environ.copy()
            # Prepend the venv bin dir so vllm sub-processes (ninja, etc.) are found
            venv_bin = str(Path(__file__).parent.parent / "venv" / "bin")
            current_path = subprocess_env.get("PATH", "")
            if venv_bin not in current_path.split(os.pathsep):
                subprocess_env["PATH"] = venv_bin + os.pathsep + current_path
            if backend == "vllm":
                if tp_from_cmd <= 1:
                    subprocess_env["CUDA_VISIBLE_DEVICES"] = str(current_gpu_id)
                    logger.info(f"🎮 vLLM TP=1 pinned to GPU {current_gpu_id} via CUDA_VISIBLE_DEVICES")
                else:
                    gpu_indices = ",".join(str(current_gpu_id + i) for i in range(tp_from_cmd))
                    subprocess_env["CUDA_VISIBLE_DEVICES"] = gpu_indices
                    logger.info(f"🎮 vLLM TP={tp_from_cmd} pinned to GPUs [{gpu_indices}] via CUDA_VISIBLE_DEVICES")
            elif backend == "llama.cpp" and int(cfg.get("tensor_parallel", "1")) == 1:
                if max_free_gb >= NEEDGB:
                    subprocess_env["CUDA_VISIBLE_DEVICES"] = str(current_gpu_id)
                    logger.info(f"🎮 llama.cpp TP=1 pinned to GPU {current_gpu_id} via CUDA_VISIBLE_DEVICES")
                else:
                    logger.info(f"🎮 llama.cpp TP=1 needs multi-GPU offload ({NEEDGB:.1f}GB > {max_free_gb:.1f}GB) — all GPUs visible")

            import subprocess as _sp
            proc = _sp.Popen(
                cmd,
                stdout=logfile,
                stderr=_sp.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                env=subprocess_env,
            )

            with state.global_lock:
                state.active_servers[physicalname] = {
                    "process": proc,
                    "pid": proc.pid,
                    "port": port,
                    "backend": backend,
                    "logfile": logfilepath,
                }
                state.server_idle_time[physicalname] = time.time()

            save_backend_pid(physicalname, proc.pid, port, backend)
            logger.info(f"✅ Process started PID {proc.pid} on port {port}, GPU {current_gpu_id}")
            logger.info(f"📄 Log: tail -f {logfilepath}")

            ready = await wait_for_model_ready(
                proc, port, backend, logfilepath, displayname,
                timeout=300, log_start_position=log_start_position,
            )

            if not ready:
                returncode = proc.poll()
                if returncode is not None:
                    logger.error(f"{displayname}:{port} exited with code {returncode}")
                    logfile.seek(log_start_position)
                    log_content = logfile.read()
                    if "Free memory" in log_content and "less than desired" in log_content:
                        logger.warning(
                            f"💥 VRAM allocation failed on GPU {current_gpu_id}, retrying with adjusted config..."
                        )
                        raise RuntimeError("VRAM allocation failed")
                    raise RuntimeError(f"{displayname} startup failed (code {returncode})")
                raise RuntimeError(f"{displayname} not ready after 300s")

            logger.info(f"🚀 {displayname} loaded and ready on GPU {current_gpu_id}")
            return port

        except RuntimeError as runtime_err:
            with state.global_lock:
                if state.active_servers.get(physicalname):
                    p = state.active_servers[physicalname].get("process")
                    if p:
                        try:
                            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                        except Exception:
                            pass
                    state.active_servers.pop(physicalname, None)
                    state.server_idle_time.pop(physicalname, None)
            if attempt < max_attempts - 1 and "VRAM allocation" in str(runtime_err):
                logger.warning(f"💥 Attempt {attempt + 1} failed (GPU {current_gpu_id}), retrying with different GPU...")
                skip_gpu = current_gpu_id
                proc = None
                if logfile and not logfile.closed:
                    logfile.close()
                    logfile = open(logfilepath, "a+")
                    log_start_position = logfile.tell()
                await asyncio.sleep(3)
                continue
            raise
        finally:
            if logfile and not logfile.closed:
                logfile.close()

    raise RuntimeError(f"Model {displayname} failed to load after all attempts")
