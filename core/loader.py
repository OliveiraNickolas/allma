"""
Model loading: spinner, readiness check, ensure_base_model.
"""
import asyncio
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from core.config import logger, BASE_MODELS, ALLMA_LOG_DIR, GPU_MEMORY_THRESHOLD_GB, MODEL_LOAD_TIMEOUT
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
# BACKEND LOG ROTATION
# ==============================================================================
# Backends (vLLM, llama-server) write directly to a file — no Python logging
# framework in the middle — so we rotate here, right before handing the FD to
# Popen. Every reload of a model checks the previous log's size and moves it
# aside if it grew past the cap; the next open starts fresh. Long-running
# backends (weeks without an unload) will still grow, but that's a rare case
# in practice — a reload trims them.

_BACKEND_LOG_MAX_BYTES = 20 * 1024 * 1024   # 20 MB per active log
_BACKEND_LOG_KEEP = 3                        # number of rotated copies to keep


def _rotate_backend_log(path: Path) -> None:
    """Move oversized backend logs aside so the new run gets a clean file.

    Errors are swallowed on purpose — a rotate failure must never block a
    model load. If the FS is full or perms are broken the open below will
    surface a real error anyway.
    """
    try:
        if not path.exists() or path.stat().st_size < _BACKEND_LOG_MAX_BYTES:
            return
        for i in range(_BACKEND_LOG_KEEP, 0, -1):
            older = path.with_suffix(path.suffix + f".{i}")
            if i == _BACKEND_LOG_KEEP and older.exists():
                older.unlink()
            newer = path.with_suffix(path.suffix + f".{i - 1}") if i > 1 else path
            if newer.exists():
                newer.rename(older)
        logger.debug(f"Rotated backend log {path.name} (was ≥{_BACKEND_LOG_MAX_BYTES // 1024 // 1024} MB)")
    except Exception as e:
        logger.debug(f"Backend log rotate skipped for {path.name}: {e}")


# ==============================================================================
# LOADING UI
# ==============================================================================

class LoadingSpinner:
    def __init__(self, message: str = "Loading"):
        self.message = message
        self.running = False
        self._thread = None
        self._start_time = None

    def _spin(self):
        import shutil
        from core.ghost_art import render_rows, WIN, ROWS
        tick = 0
        sys.stdout.write("\n" * (ROWS - 1))
        while self.running:
            elapsed = time.time() - self._start_time
            try:
                term_w = shutil.get_terminal_size().columns
            except Exception:
                term_w = 80
            canvas = render_rows(tick)
            time_part = f"  [{elapsed:.0f}s]"
            # visible width of the last-row prefix is fixed: 2 + WIN + 2
            max_msg = term_w - (WIN + 4) - len(time_part) - 1
            msg = self.message if len(self.message) <= max_msg else self.message[:max_msg - 1] + "…"
            sys.stdout.write(f"\033[{ROWS - 1}A")
            for r in canvas[:-1]:
                sys.stdout.write(f"\r\033[K  {r}\n")
            sys.stdout.write(f"\r\033[K  {canvas[-1]}  {msg}{time_part}")
            sys.stdout.flush()
            time.sleep(0.06)
            tick += 1
        sys.stdout.write("\r\033[K")
        for _ in range(ROWS - 1):
            sys.stdout.write("\033[1A\r\033[K")
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
        # Match by substring so we survive llama.cpp log-format changes. Newer
        # builds log "srv  llama_server: server is listening on http://..."; older
        # builds logged "main: server is listening on ...". Keep the trailing "on"
        # — llama.cpp binds its HTTP port BEFORE loading the model, and upstream
        # builds log "main: HTTP server is listening, hostname: ..." at that point,
        # which must NOT count as ready.
        "llama.cpp": ["server is listening on", "all slots are idle"],
    }

    # Fallback probe: GET /health. Both backends expose it and only answer 200
    # once the model is actually ready — keeps us working even if the log strings
    # above change again. A bare TCP-connect check is not enough: llama.cpp binds
    # its port before loading the model and serves 503 while loading.
    use_health_fallback = backend in ("vllm", "llama.cpp")
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

            if use_health_fallback:
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=1
                    ) as resp:
                        if resp.status == 200:
                            spinner.stop(success=True)
                            return True
                except (urllib.error.URLError, OSError, TimeoutError):
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
# MODEL LOADING
# ==============================================================================
def _kv_ceiling_from_log(basename: str) -> int:
    """Extract vLLM's 'estimated maximum model length is N' from the tail of
    the backend log. Returns the LAST (most recent) match, or 0."""
    import re as _re
    logpath = ALLMA_LOG_DIR / f"{basename}.log"
    try:
        with open(logpath, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 65536))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return 0
    matches = _re.findall(r"estimated maximum model length is (\d+)", tail)
    return int(matches[-1]) if matches else 0


async def ensure_base_model(basename: str, profilename: Optional[str] = None, gpu_id: Optional[int] = None):
    if basename not in BASE_MODELS:
        raise RuntimeError(f"Model {basename} not configured")

    cfg = state.effective_base_cfg(basename)
    # Honour pinned GPU from config if caller did not specify a gpu_id.
    # Accepts both "gpu_id" (used by process builders) and legacy "pinned_gpu".
    if gpu_id is None:
        for _field in ("gpu_id", "pinned_gpu"):
            if _field in cfg:
                try:
                    _val = int(cfg[_field])
                    if _val >= 0:
                        gpu_id = _val
                        logger.info(f"{basename}: pinned to GPU {gpu_id} (config field '{_field}')")
                        break
                except (TypeError, ValueError):
                    pass
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
        try:
            port = await _load_model_impl(basename, cfg, backend, displayname, gpu_id=gpu_id)
        except RuntimeError as first_err:
            # Auto-fit: vLLM refuses to start when the configured context
            # doesn't fit the KV cache budget, but tells us the ceiling:
            #   "the estimated maximum model length is 46464"
            # Retry ONCE at 95% of that ceiling (session-only override — the
            # .allm on disk is untouched) instead of failing on the user.
            fitted = _kv_ceiling_from_log(basename) if backend == "vllm" else 0
            configured = int(cfg.get("max_model_len", 0) or 0)
            if not fitted or (configured and fitted >= configured):
                raise
            new_len = max(4096, (int(fitted * 0.95) // 4096) * 4096)
            logger.warning(
                f"{displayname}: max_model_len={configured or '?'} doesn't fit "
                f"(vLLM ceiling {fitted}). Auto-fit: retrying once with "
                f"max_model_len={new_len}. Edit the .allm to make this permanent."
            )
            retry_cfg = dict(cfg)
            retry_cfg["max_model_len"] = new_len
            with state.global_lock:
                state.session_load_overrides.setdefault(basename, {})["max_model_len"] = new_len
            try:
                port = await _load_model_impl(basename, retry_cfg, backend, displayname, gpu_id=gpu_id)
            except RuntimeError:
                raise first_err
        success = True
        return port
    finally:
        state.loading_models.discard(basename)
        if not success:
            with state.global_lock:
                state.active_servers.pop(basename, None)
                state.server_idle_time.pop(basename, None)


def _cfg_tp_size(cfg: dict) -> int:
    """Tensor-parallel size a vLLM config will use (extra_args wins over the field)."""
    raw_extra = cfg.get("extra_args") or []
    if "--tensor-parallel-size" in raw_extra:
        try:
            return max(1, int(raw_extra[raw_extra.index("--tensor-parallel-size") + 1]))
        except (ValueError, IndexError):
            pass
    try:
        return max(1, int(cfg.get("tensor_parallel", "1")))
    except (TypeError, ValueError):
        return 1


async def _load_model_impl(basename: str, cfg: dict, backend: str, displayname: str, gpu_id: Optional[int] = None) -> int:
    """Internal implementation of model loading."""
    NEEDGB = get_model_vram_need(cfg, basename)
    logger.info(f"{displayname} needs {NEEDGB:.1f}GB VRAM")

    # vLLM validates its exact KV budget at startup, and the auto-fit retry
    # resolves context overshoot — so the hard gate below only has to prove
    # the model can start AT ALL (weights + a minimal 8k context). The full
    # estimate still drives the swap/unload decisions.
    if backend == "vllm":
        _min_cfg = dict(cfg)
        _min_cfg["max_model_len"] = 8192
        NEED_GATE = get_model_vram_need(_min_cfg, basename)
    else:
        NEED_GATE = NEEDGB

    # VRAM check — unload other models if needed
    _all_gpus_info = get_all_gpus()
    _gpu_mem_util = float(cfg.get("gpu_memory_utilization", "0.90"))
    _max_single_usable = max((g["total_gb"] * _gpu_mem_util for g in _all_gpus_info), default=0.0)
    gpus = get_free_gpu_memory()
    max_free_gb = max((g["free_gb"] for g in gpus), default=0.0)
    # Pinned placement. By the time we get here, gpu_id already reflects both an
    # explicit --gpu flag and a config pin (merged upstream in ensure_base_model).
    # A pinned model runs on a KNOWN GPU set (CUDA_VISIBLE_DEVICES) — llama.cpp on
    # that single GPU, vLLM on TP contiguous GPUs starting there — so the VRAM
    # check must look at those GPUs' free memory, not the most-free GPU elsewhere
    # that the model can't even use.
    # Explicit multi-GPU for llama.cpp: `@gpus 0,1` spreads the model (weights
    # AND KV cache) across those GPUs — the way to run a huge context that a
    # single card's VRAM can't hold. Distinct from `@gpu N` (single pin).
    _explicit_gpus = None
    _g = cfg.get("gpus")
    if backend == "llama.cpp" and isinstance(_g, list) and len(_g) > 1:
        try:
            _explicit_gpus = [int(x) for x in _g]
        except (TypeError, ValueError):
            _explicit_gpus = None

    if _explicit_gpus:
        _target_gpus = _explicit_gpus
    elif gpu_id is not None:
        _target_gpus = list(range(gpu_id, gpu_id + _cfg_tp_size(cfg))) \
            if backend == "vllm" else [gpu_id]
    else:
        _target_gpus = None
    _pinned = _target_gpus is not None
    _where = f" on GPU {','.join(map(str, _target_gpus))}" if _pinned else ""

    needs_multi_gpu = (backend == "vllm" and NEEDGB > _max_single_usable) or \
                      (backend == "llama.cpp" and not _pinned and NEEDGB > max_free_gb)

    def _available(gpu_list) -> float:
        """Free VRAM usable by THIS load given its placement."""
        if _pinned:
            return sum(g["free_gb"] for g in gpu_list if g["index"] in _target_gpus)
        if needs_multi_gpu:
            return sum(g["free_gb"] for g in gpu_list)
        return max((g["free_gb"] for g in gpu_list), default=0.0)

    available_gb = _available(gpus)

    if available_gb < NEEDGB + GPU_MEMORY_THRESHOLD_GB:
        with state.global_lock:
            if _pinned:
                # Surgical swap: free only the model(s) occupying the target GPU(s),
                # leaving models on other GPUs untouched. "gpus" covers every GPU a
                # server occupies (TP shards included); gpu_allocation is a
                # single-GPU fallback for entries loaded before this field existed.
                names_to_unload = [
                    n for n, srv in state.active_servers.items()
                    if n != basename and any(
                        g in srv.get("gpus", [state.gpu_allocation.get(n)])
                        for g in _target_gpus
                    )
                ]
            else:
                names_to_unload = [n for n in state.active_servers if n != basename]
        loop = asyncio.get_event_loop()
        for name in names_to_unload:
            logger.info(f"Unloading {name} to free VRAM{_where}")
            await loop.run_in_executor(None, lambda n=name: shutdown_server(n, "swap", fast=True))
        # kill_vram_fast nukes ALL allma backends — only safe when NOT doing a
        # surgical, GPU-targeted swap (otherwise it would kill models on other GPUs).
        if not _pinned:
            await loop.run_in_executor(None, kill_vram_fast)
        # Poll until VRAM is actually freed (max 6s) instead of blind sleep
        for _ in range(12):
            await asyncio.sleep(0.5)
            if _available(get_free_gpu_memory()) >= NEEDGB:
                break

        gpus = get_free_gpu_memory()
        max_free_gb = max((g["free_gb"] for g in gpus), default=0.0)
        available_gb = _available(gpus)
        if available_gb < NEED_GATE:
            gpu_procs = list_gpu_processes()
            active_procs = [p for p in gpu_procs if p["memory_mb"] > 100]
            if active_procs:
                with state.global_lock:
                    allma_pids = {s["process"].pid for s in state.active_servers.values()}
                for p in sorted(active_procs, key=lambda x: x["memory_mb"], reverse=True)[:5]:
                    tag = "allma" if p["pid"] in allma_pids else "external"
                    logger.error(f"  PID {p['pid']} ({p['name']}): {p['memory_mb']//1024:.0f}GB [{tag}]")
            raise RuntimeError(
                f"Not enough VRAM{_where}: {displayname} needs {NEED_GATE:.1f}GB "
                f"(weights + minimal context), only {available_gb:.1f}GB free"
            )
        if available_gb < NEEDGB:
            logger.info(
                f"{displayname}: configured context may not fit "
                f"(estimate {NEEDGB:.1f}GB vs {available_gb:.1f}GB free) — "
                f"vLLM will decide; auto-fit adjusts if needed"
            )

    logger.info(f"Loading {displayname} ({backend})")

    import subprocess as _sp

    logfilepath = ALLMA_LOG_DIR / f"{basename}.log"
    _rotate_backend_log(logfilepath)
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

        # Every GPU this server will occupy — recorded in active_servers so the
        # surgical swap above can tell which models sit on a given GPU.
        server_gpus = [current_gpu_id]
        if backend == "vllm":
            if tp_from_cmd > 1:
                server_gpus = [current_gpu_id + i for i in range(tp_from_cmd)]
                logger.info(f"TP={tp_from_cmd} on GPUs [{','.join(map(str, server_gpus))}]")
            subprocess_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in server_gpus)
            # Fix VRAM fragmentation (marlin_gemm OOM, quantized kernels) — zero downside
            subprocess_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        elif backend == "llama.cpp":
            # CPU-only (n_gpu_layers=0 or --device none): hide all GPUs so the
            # CUDA-enabled binary doesn't even create a CUDA context (which alone
            # eats ~1GB VRAM). This is what makes it a true 0-VRAM CPU run.
            _ea = cfg.get("extra_args", []) or []
            _dev_none = any(
                _ea[i + 1].strip().lower() == "none"
                for i, a in enumerate(_ea[:-1]) if a in ("--device", "-dev")
            )
            _cpu_only = str(cfg.get("n_gpu_layers", "")).strip() == "0" or _dev_none
            if _cpu_only:
                subprocess_env["CUDA_VISIBLE_DEVICES"] = ""
                logger.info(f"{basename}: CPU-only — GPUs hidden (CUDA_VISIBLE_DEVICES='')")
            else:
                if _explicit_gpus:
                    # `@gpus 0,1` — user asked for a specific multi-GPU split
                    # (usually to hold a bigger context than one card can).
                    server_gpus = _explicit_gpus
                    logger.info(f"llama.cpp multi-GPU (@gpus): [{','.join(map(str, server_gpus))}]")
                elif not (_pinned or max_free_gb >= NEEDGB):
                    # Doesn't fit on one GPU and isn't pinned — spread across all GPUs,
                    # primary first so it becomes CUDA0 (larger shard + mmproj land there)
                    all_gpu_ids = [g["index"] for g in _all_gpus_info]
                    server_gpus = [current_gpu_id] + [g for g in all_gpu_ids if g != current_gpu_id]
                    logger.info(f"llama.cpp multi-GPU (auto): [{','.join(map(str, server_gpus))}]")
                subprocess_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in server_gpus)

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
                "gpus": server_gpus,
                "logfile": logfilepath,
                "pin_loaded": bool(cfg.get("pin_loaded")),
            }
            state.server_idle_time[basename] = time.time()

        save_backend_pid(basename, proc.pid, port, backend)
        # New process = fresh VRAM state; drop the cached snapshot so the next
        # allocator decision sees ground truth instead of the pre-load view.
        try:
            from core.gpu import invalidate_gpu_cache
            invalidate_gpu_cache()
        except Exception:
            pass
        logger.info(f"PID {proc.pid} on port {port} (GPU {current_gpu_id}, TP={tp_from_cmd})")

        ready = await wait_for_model_ready(
            proc, port, backend, logfilepath, displayname,
            timeout=MODEL_LOAD_TIMEOUT, log_start_position=log_start_position,
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
            raise RuntimeError(f"{displayname} not ready after {MODEL_LOAD_TIMEOUT}s")

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
