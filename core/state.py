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

# Error detection & auto-resolution
last_error_analysis: Dict[str, Any] = {}  # {base_name: ErrorAnalysis}
auto_fix_attempt_count: Dict[str, int] = {}  # {base_name: attempt_count}

# Hardware detection & calibration (bootstrap)
hardware_profile: Any = None  # HardwareProfile or None
hardware_detected_at: Any = None  # ISO timestamp
bootstrap_calibrations: Dict[str, Any] = {}  # {base_name: CalibrationResult}

# ==============================================================================
# PORT ALLOCATION
# ==============================================================================
_next_vllm_port = VLLM_BASE_PORT
_next_llama_port = LLAMA_BASE_PORT
_port_lock = threading.Lock()


def get_next_vllm_port() -> int:
    global _next_vllm_port
    with _port_lock:
        port = _next_vllm_port
        _next_vllm_port += 1
    return port


def get_next_llama_port() -> int:
    global _next_llama_port
    with _port_lock:
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
