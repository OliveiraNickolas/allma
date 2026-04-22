#!/usr/bin/env python3
"""
Allma — entry point.
"""
import asyncio
import os
import signal
import sys
import threading
import time
from typing import Any

# Ensure the project root is on sys.path so `core` and `configs` are importable
sys.path.insert(0, os.path.dirname(__file__))

import uvicorn

from core.config import ALLMA_PORT, logger
from core.server import app, close_http_client, show_banner
from core.health import health_monitor
from core.process import kill_process_tree, cleanup_orphaned_backends, clear_backend_registry
from core.bootstrap import BootstrapDetector
import core.state as state


def main():
    show_banner()
    cleanup_orphaned_backends()

    # Bootstrap GPU detection
    try:
        logger.info("🔍 Detecting hardware...")
        start_time = time.time()

        profile = asyncio.run(BootstrapDetector.detect_hardware())
        duration_ms = (time.time() - start_time) * 1000

        logger.info(
            f"✅ Hardware detected ({duration_ms:.0f}ms):\n"
            f"   Driver: {profile.driver_version} | CUDA: {profile.cuda_version}\n"
            f"   GPUs: {len(profile.gpus)} | Total VRAM: {profile.total_vram_gb:.1f}GB "
            f"({profile.available_vram_gb:.1f}GB free)"
        )

        for gpu in profile.gpus:
            logger.info(
                f"   GPU {gpu.index}: {gpu.name} (compute {gpu.compute_capability}) "
                f"— {gpu.total_memory_gb:.1f}GB total, {gpu.free_memory_gb:.1f}GB free"
            )

        state.hardware_profile = profile
        state.hardware_detected_at = profile.detected_at

        # Save profile for debugging
        profile_path = BootstrapDetector.save_profile_to_file(profile)
        logger.debug(f"📄 Hardware profile saved: {profile_path}")

    except Exception as e:
        logger.warning(f"⚠️  Hardware detection failed: {e}")
        logger.warning("   Continuing with static config (may have issues)")
        state.hardware_profile = None
        state.hardware_detected_at = None

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
        clear_backend_registry()
        _health_monitor_thread.join(timeout=5)
        time.sleep(1)
        os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    uvicorn.run(app, host="127.0.0.1", port=ALLMA_PORT)


if __name__ == "__main__":
    main()
