"""
All shared mutable runtime state.
"""
import os
import socket
import threading
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
ALLAMA_PID = os.getpid()
gpu_allocation: Dict[str, int] = {}

# Health monitor
_health_monitor_running = threading.Event()
_health_monitor_thread: Optional[threading.Thread] = None

# HTTP client (managed by server.py)
httpx_client = None

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
