"""
All shared mutable runtime state.
"""
import os
import socket
import threading
import time
from typing import Any, Dict, Optional

from core.config import VLLM_BASE_PORT, LLAMA_BASE_PORT

# ==============================================================================
# RUNTIME STATE
# ==============================================================================
active_servers: Dict[str, dict] = {}
server_idle_time: Dict[str, float] = {}
global_lock = threading.Lock()
loading_models: set = set()
running = True
ALLMA_PID = os.getpid()
gpu_allocation: Dict[str, int] = {}
startup_time: float = time.time()

# Health monitor
_health_monitor_running = threading.Event()

# HTTP client (managed by server.py)
httpx_client = None

# Default profile to use when an unknown model name is requested (e.g. "claude-sonnet-4-5").
# Set by /v1/load so that `allma launch` pins a specific profile for the session.
default_profile: Optional[str] = None

# Error detection
last_error_analysis: Dict[str, Any] = {}  # {base_name: ErrorAnalysis}

# Hardware detection & calibration (bootstrap)
hardware_profile: Any = None  # HardwareProfile or None
hardware_detected_at: Any = None  # ISO timestamp
bootstrap_calibrations: Dict[str, Any] = {}  # {base_name: CalibrationResult}

# ==============================================================================
# EPHEMERAL OVERRIDES — set by /v1/load (TUI "load one-time"), never written to
# .allm files. They survive idle unload/reload (the session continues) and are
# cleared on server restart or when /v1/load runs without them.
# ==============================================================================
session_load_overrides: Dict[str, dict] = {}  # {base_name: base-cfg overrides}
session_sampling: Dict[str, dict] = {}        # {profile_name: sampling overrides}


def effective_base_cfg(base_name: str) -> dict:
    """Base config with any session (one-time) overrides applied."""
    from core.config import BASE_MODELS  # late import: BASE_MODELS mutates on /v1/reload-configs
    cfg = BASE_MODELS.get(base_name, {})
    overrides = session_load_overrides.get(base_name)
    return {**cfg, **overrides} if overrides else cfg


def effective_sampling(profile_name: str, logical_cfg: dict) -> dict:
    """Profile sampling with any session (one-time) overrides applied."""
    sampling = logical_cfg.get("sampling", {})
    overrides = session_sampling.get(profile_name)
    return {**sampling, **overrides} if overrides else sampling

# ==============================================================================
# PORT ALLOCATION
# ==============================================================================
_next_vllm_port = VLLM_BASE_PORT
_next_llama_port = LLAMA_BASE_PORT
_port_lock = threading.Lock()


def _ports_in_use(base_port: int, count: int) -> set[int]:
    """Ports currently owned by an alive backend in `base_port .. base_port+count`.

    We check `active_servers` (owned by allma right now) rather than probing
    every socket — a poll-and-probe would race with `is_port_free`.
    """
    live = set()
    for srv in active_servers.values():
        p = srv.get("port")
        if isinstance(p, int) and base_port <= p < base_port + count:
            live.add(p)
    return live


def get_next_vllm_port(window: int = 100) -> int:
    """Return the lowest port in the vLLM window that no live backend owns.

    Long sessions used to leak the port counter upward — 100+ load/unload
    cycles would eventually raise "no free port" even with everything idle.
    Reusing the vacated slots keeps the range bounded.
    """
    global _next_vllm_port
    with _port_lock:
        in_use = _ports_in_use(VLLM_BASE_PORT, window)
        for candidate in range(VLLM_BASE_PORT, VLLM_BASE_PORT + window):
            if candidate not in in_use:
                _next_vllm_port = candidate + 1
                return candidate
        # Fully saturated window — fall back to the old ever-growing counter
        port = _next_vllm_port
        _next_vllm_port += 1
        return port


def get_next_llama_port(window: int = 100) -> int:
    global _next_llama_port
    with _port_lock:
        in_use = _ports_in_use(LLAMA_BASE_PORT, window)
        for candidate in range(LLAMA_BASE_PORT, LLAMA_BASE_PORT + window):
            if candidate not in in_use:
                _next_llama_port = candidate + 1
                return candidate
        port = _next_llama_port
        _next_llama_port += 1
        return port


def is_port_free(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False
