#!/usr/bin/env python3
"""
Allama — entry point.
"""
import os
import signal
import sys
import threading
import time
from typing import Any

# Ensure the project root is on sys.path so `core` and `configs` are importable
sys.path.insert(0, os.path.dirname(__file__))

import uvicorn

from core.config import ALLAMA_PORT, logger
from core.server import app, close_http_client, show_banner
from core.health import health_monitor
from core.process import kill_process_tree
import core.state as state


def main():
    show_banner()

    _health_monitor_thread = threading.Thread(target=health_monitor, daemon=True)
    _health_monitor_thread.start()

    def signal_handler(sig: int, frame: Any) -> None:
        if not state.running:
            logger.critical("Second signal received, force kill")
            state._health_monitor_running.clear()
            time.sleep(0.5)
            os.kill(os.getpid(), signal.SIGKILL)
            return
        logger.info("🛑 Shutdown requested...")
        state.running = False
        state._health_monitor_running.clear()
        with state.global_lock:
            for name in list(state.active_servers.keys()):
                server = state.active_servers.get(name)
                if server and server.get("process"):
                    pid = server["process"].pid
                    kill_process_tree(pid)
                    logger.info(f"🔥 Killed process tree for {name} (PID {pid})")
        _health_monitor_thread.join(timeout=5)
        time.sleep(1)
        os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    uvicorn.run(app, host="127.0.0.1", port=ALLAMA_PORT)


if __name__ == "__main__":
    main()
