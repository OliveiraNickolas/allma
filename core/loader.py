"""
Model loading: spinner, readiness check, ensure_base_model.
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

from core.config import logger, BASE_MODELS, ALLMA_LOG_DIR
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
from core.error_detector import ErrorDetector, tail_file


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

    # 3-row llama sprite (mirrored from allma_cli)
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
        import shutil
        ci = si = ni = 0
        last_c = last_s = last_n = 0
        tick = 0
        sys.stdout.write("\n\n")
        while self.running:
            elapsed = time.time() - self._start_time
            try:
                term_w = shutil.get_terminal_size().columns
            except Exception:
                term_w = 80
            cview = (_CLOUDS * 2)[ci % len(_CLOUDS): ci % len(_CLOUDS) + _WIN]
            sview = (_SKY    * 2)[si % len(_SKY):    si % len(_SKY)    + _WIN]
            nview = (_MOUNTAINS * 2)[ni % len(_MOUNTAINS): ni % len(_MOUNTAINS) + _WIN]
            cview, sview, nview = self._inject(cview, sview, nview, tick)
            cloud_line = f"  {cview}"
            sky_line   = f"  {sview}"
            prefix     = f"  {nview}  "
            time_part  = f"  [{elapsed:.0f}s]"
            # Truncate message so the full near_line fits within terminal width
            max_msg = term_w - len(prefix) - len(time_part) - 1
            msg = self.message if len(self.message) <= max_msg else self.message[:max_msg - 1] + "…"
            near_line  = f"{prefix}{msg}{time_part}"
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

    spinner = LoadingSpinner(f"Loading {displayname}")
    spinner.start()

    try:
        while time.time() < deadline:
            returncode = proc.poll()
            if returncode is not None:
                spinner.stop(success=False)
                logger.error(f"{displayname} exited with code {returncode} during loading")
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
                                    return True
            except Exception as e:
                logger.debug(f"Error reading log for {displayname}: {e}")

            if use_tcp_fallback:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(0.5)
                        if s.connect_ex(("127.0.0.1", port)) == 0:
                            spinner.stop(success=True)
                            return True
                except (ConnectionRefusedError, OSError, TimeoutError):
                    pass

            await asyncio.sleep(1)

        spinner.stop(success=False)
        logger.error(f"Timeout {timeout}s waiting for {displayname}")
        return False
    except Exception as e:
        spinner.stop(success=False)
        logger.error(f"Error waiting for {displayname}: {e}")
        raise


# ==============================================================================
# AUTO-FIX HELPERS
# ==============================================================================
def save_config_to_file(basename: str, cfg: dict) -> bool:
    """Save config back to .allm file. Returns True if successful."""
    try:
        from core.config import CONFIG_DIR
        config_file = CONFIG_DIR / "base" / f"{basename}.allm"
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}")
            return False

        # Read original file to preserve formatting
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Update only the specific keys that were auto-fixed
        lines = content.split('\n')
        new_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Check if this line is a key we need to update
            if '=' in stripped and not stripped.startswith('#'):
                key = stripped.split('=')[0].strip()

                if key == 'extra_args':
                    # Handle multi-line extra_args
                    new_lines.append('extra_args = [')
                    extra_args = cfg.get('extra_args', [])
                    for arg in extra_args:
                        new_lines.append(f'    "{arg}",')
                    new_lines.append(']')
                    # Skip original extra_args block
                    i += 1
                    while i < len(lines) and ']' not in lines[i]:
                        i += 1
                    i += 1
                    continue

                elif key in ['n_ctx', 'n_batch', 'max_num_seqs', 'max_model_len']:
                    if key in cfg:
                        new_lines.append(f'{key} = {cfg[key]}')
                        i += 1
                        continue

            new_lines.append(line)
            i += 1

        # Write back
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))

        logger.info(f"Config saved: {config_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return False


def apply_auto_fix(basename: str, cfg: dict, error_analysis) -> bool:
    """Apply auto-fix to config based on error analysis. Returns True if applied."""
    if not error_analysis or not error_analysis.auto_fix_available:
        return False

    action = error_analysis.auto_fix_action
    attempt_count = state.auto_fix_attempt_count.get(basename, 0)

    if attempt_count >= 3:
        logger.warning(f" Auto-fix max attempts (3) reached for {basename}, skipping")
        return False

    if action == "reduce_ubatch_size":
        current = int(cfg.get("max_num_seqs") or "8")
        new = max(2, current // 2)
        if new < current:
            cfg["max_num_seqs"] = str(new)
            state.auto_fix_attempt_count[basename] = attempt_count + 1
            logger.info(f"Auto-fix: reduced max_num_seqs {current} → {new} (attempt {attempt_count + 1}/3)")
            save_config_to_file(basename, cfg)
            return True

    elif action == "reduce_batch_params":
        # Try reducing ubatch-size first, then n_batch
        if "max_num_seqs" in cfg:
            current = int(cfg["max_num_seqs"])
            new = max(2, current // 2)
            if new < current:
                cfg["max_num_seqs"] = str(new)
                state.auto_fix_attempt_count[basename] = attempt_count + 1
                logger.info(f"Auto-fix: reduced max_num_seqs {current} → {new}")
                save_config_to_file(basename, cfg)
                return True

    elif action == "reduce_context_length":
        current = int(cfg.get("max_model_len", "262144"))
        new = max(8192, current // 2)
        if new < current:
            cfg["max_model_len"] = str(new)
            state.auto_fix_attempt_count[basename] = attempt_count + 1
            logger.info(f"Auto-fix: reduced max_model_len {current} → {new}")
            save_config_to_file(basename, cfg)
            return True

    elif action == "increase_defrag_threshold":
        # Add or increase defrag threshold in extra_args
        extra_args = cfg.get("extra_args", [])
        for i, arg in enumerate(extra_args):
            if arg == "--defrag-thold" and i + 1 < len(extra_args):
                try:
                    current_val = float(extra_args[i + 1])
                    new_val = max(0.01, current_val / 2)
                    extra_args[i + 1] = str(new_val)
                    state.auto_fix_attempt_count[basename] = attempt_count + 1
                    logger.info(f"Auto-fix: increased defrag from {current_val} → {new_val}")
                    save_config_to_file(basename, cfg)
                    return True
                except ValueError:
                    pass

    return False


# ==============================================================================
# MODEL LOADING
# ==============================================================================
async def ensure_base_model(basename: str, profilename: Optional[str] = None, gpu_id: Optional[int] = None):
    if basename not in BASE_MODELS:
        raise RuntimeError(f"Model {basename} not configured")

    cfg = BASE_MODELS[basename]
    # Honour pinned_gpu from config if caller did not specify a gpu_id
    if gpu_id is None and "pinned_gpu" in cfg:
        gpu_id = int(cfg["pinned_gpu"])
        logger.info(f"{basename}: pinned_gpu={gpu_id} from config")
    backend = cfg.get("backend", "vllm")
    displayname = profilename or basename

    with state.global_lock:
        if basename in state.active_servers:
            proc = state.active_servers[basename]["process"]
            if proc.poll() is None:
                port = state.active_servers[basename]["port"]
                state.server_idle_time[basename] = time.time()
                logger.debug(f" Reusing {displayname}:{port}")
                return port

        already_loading = basename in state.loading_models
        if not already_loading:
            state.loading_models.add(basename)

    if already_loading:
        logger.info(f"{displayname} already loading, waiting...")
        for _ in range(150):
            await asyncio.sleep(2)
            with state.global_lock:
                if basename in state.active_servers:
                    proc = state.active_servers[basename]["process"]
                    if proc and proc.poll() is None:
                        port = state.active_servers[basename]["port"]
                        state.server_idle_time[basename] = time.time()
                        logger.debug(f" Reusing {displayname}:{port} after wait")
                        return port
                if basename not in state.loading_models:
                    # Primary loader finished — check if it succeeded
                    if basename in state.active_servers:
                        proc = state.active_servers[basename]["process"]
                        if proc and proc.poll() is None:
                            port = state.active_servers[basename]["port"]
                            state.server_idle_time[basename] = time.time()
                            return port
                    raise RuntimeError(f"{displayname} failed to load")
        raise RuntimeError(f"Loading timeout for {displayname} - load may have stuck")

    success = False
    try:
        port = await _load_model_impl(basename, cfg, backend, displayname, gpu_id=gpu_id)
        success = True
        return port
    finally:
        state.loading_models.discard(basename)
        if not success:
            with state.global_lock:
                state.active_servers.pop(basename, None)
                state.server_idle_time.pop(basename, None)


async def _load_model_impl(basename: str, cfg: dict, backend: str, displayname: str, gpu_id: Optional[int] = None) -> int:
    """Internal implementation of model loading."""
    NEEDGB = get_model_vram_need(cfg, basename)
    logger.info(f"{displayname} needs {NEEDGB:.1f}GB VRAM")

    # VRAM check — unload other models if needed
    _all_gpus_info = get_all_gpus()
    _gpu_mem_util = float(cfg.get("gpu_memory_utilization", "0.90"))
    _max_single_usable = max((g["total_gb"] * _gpu_mem_util for g in _all_gpus_info), default=0.0)
    gpus = get_free_gpu_memory()
    max_free_gb = max((g["free_gb"] for g in gpus), default=0.0)
    _llama_pinned   = backend == "llama.cpp" and int(cfg.get("gpu_id", -1)) >= 0
    needs_multi_gpu = (backend == "vllm" and NEEDGB > _max_single_usable) or \
                      (backend == "llama.cpp" and not _llama_pinned and NEEDGB > max_free_gb)
    available_gb = sum(g["free_gb"] for g in gpus) if needs_multi_gpu else max_free_gb

    if available_gb < NEEDGB:
        with state.global_lock:
            names_to_unload = [n for n in state.active_servers if n != basename]
        loop = asyncio.get_event_loop()
        for name in names_to_unload:
            logger.info(f"Unloading {name} to free VRAM")
            await loop.run_in_executor(None, lambda n=name: shutdown_server(n, "swap", fast=True))
        await loop.run_in_executor(None, kill_vram_fast)
        # Poll until VRAM is actually freed (max 6s) instead of blind sleep
        for _ in range(12):
            await asyncio.sleep(0.5)
            gpus_check = get_free_gpu_memory()
            chk_free = sum(g["free_gb"] for g in gpus_check) if needs_multi_gpu else max((g["free_gb"] for g in gpus_check), default=0.0)
            if chk_free >= NEEDGB:
                break

        gpus = get_free_gpu_memory()
        max_free_gb = max((g["free_gb"] for g in gpus), default=0.0)
        available_gb = sum(g["free_gb"] for g in gpus) if needs_multi_gpu else max_free_gb
        if available_gb < NEEDGB:
            gpu_procs = list_gpu_processes()
            active_procs = [p for p in gpu_procs if p["memory_mb"] > 100]
            if active_procs:
                with state.global_lock:
                    allma_pids = {s["process"].pid for s in state.active_servers.values()}
                for p in sorted(active_procs, key=lambda x: x["memory_mb"], reverse=True)[:5]:
                    tag = "allma" if p["pid"] in allma_pids else "external"
                    logger.error(f"  PID {p['pid']} ({p['name']}): {p['memory_mb']//1024:.0f}GB [{tag}]")
            raise RuntimeError(
                f"Not enough VRAM: {displayname} needs {NEEDGB:.1f}GB, only {available_gb:.1f}GB free"
            )

    logger.info(f"Loading {displayname} ({backend})")

    import subprocess as _sp

    logfilepath = ALLMA_LOG_DIR / f"{basename}.log"
    logfile = None
    try:
        logfile = open(logfilepath, "a+")
        log_start_position = logfile.tell()
    except Exception as e:
        logger.error(f"Failed to open log file {logfilepath}: {e}")
        raise

    if backend == "vllm":
        cmd, port, current_gpu_id = build_vllm_cmd(basename, gpu_id=gpu_id)
    else:
        cmd, port, current_gpu_id = build_llama_cmd(basename, gpu_id=gpu_id)

    tp_from_cmd = 1
    try:
        tp_idx = cmd.index("--tensor-parallel-size") + 1
        if tp_idx < len(cmd):
            tp_from_cmd = int(cmd[tp_idx])
    except (ValueError, IndexError):
        pass

    proc = None
    try:
        subprocess_env = os.environ.copy()
        venv_bin = str(Path(__file__).parent.parent / "venv" / "bin")
        current_path = subprocess_env.get("PATH", "")
        if venv_bin not in current_path.split(os.pathsep):
            subprocess_env["PATH"] = venv_bin + os.pathsep + current_path

        if backend == "vllm":
            if tp_from_cmd <= 1:
                subprocess_env["CUDA_VISIBLE_DEVICES"] = str(current_gpu_id)
            else:
                gpu_indices = ",".join(str(current_gpu_id + i) for i in range(tp_from_cmd))
                subprocess_env["CUDA_VISIBLE_DEVICES"] = gpu_indices
                logger.info(f"TP={tp_from_cmd} on GPUs [{gpu_indices}]")
        elif backend == "llama.cpp":
            if max_free_gb >= NEEDGB:
                subprocess_env["CUDA_VISIBLE_DEVICES"] = str(current_gpu_id)
            else:
                # Primary GPU first so it becomes CUDA0 (larger shard + mmproj land there)
                all_gpu_ids = [g["index"] for g in _all_gpus_info]
                ordered = [current_gpu_id] + [g for g in all_gpu_ids if g != current_gpu_id]
                subprocess_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in ordered)
                logger.info(f"llama.cpp multi-GPU: [{subprocess_env['CUDA_VISIBLE_DEVICES']}]")

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
            state.active_servers[basename] = {
                "process": proc,
                "pid": proc.pid,
                "port": port,
                "backend": backend,
                "logfile": logfilepath,
            }
            state.server_idle_time[basename] = time.time()

        save_backend_pid(basename, proc.pid, port, backend)
        logger.info(f"PID {proc.pid} on port {port} (GPU {current_gpu_id}, TP={tp_from_cmd})")

        ready = await wait_for_model_ready(
            proc, port, backend, logfilepath, displayname,
            timeout=300, log_start_position=log_start_position,
        )

        if not ready:
            returncode = proc.poll()
            if returncode is not None:
                logger.error(f"{displayname} exited with code {returncode}")
                logfile.seek(log_start_position)
                log_content = logfile.read()
                error_analysis = ErrorDetector.analyze_log(log_content)
                if error_analysis:
                    logger.error(f"{error_analysis.error_type}: {error_analysis.explanation}")
                    for suggestion in error_analysis.suggestions:
                        logger.error(f"   • {suggestion}")
                    state.last_error_analysis[basename] = error_analysis
                raise RuntimeError(f"{displayname} startup failed (code {returncode})")
            raise RuntimeError(f"{displayname} not ready after 300s")

        logger.info(f"{displayname} loaded on GPU {current_gpu_id}")
        return port

    except Exception:
        with state.global_lock:
            if state.active_servers.get(basename):
                p = state.active_servers[basename].get("process")
                if p:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        try:
                            os.kill(p.pid, signal.SIGKILL)
                        except Exception as kill_err:
                            logger.warning(f"Failed to kill {basename} PID {p.pid}: {kill_err}")
                    except Exception as e:
                        logger.warning(f"killpg failed for {basename}: {e}")
                state.active_servers.pop(basename, None)
                state.server_idle_time.pop(basename, None)
        raise
    finally:
        if logfile and not logfile.closed:
            logfile.close()
