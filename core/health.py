"""
Health monitor: idle-timeout unloading and crash detection.
"""
import time
from core.config import logger, AUTO_SWAP_ENABLED, KEEP_ALIVE_SECONDS, HEALTH_CHECK_INTERVAL
import core.state as state
from core.process import shutdown_server


def health_monitor():
    state._health_monitor_running.set()
    logger.info("🏥 Health monitor started")
    try:
        while state._health_monitor_running.is_set():
            try:
                now = time.time()
                to_unload: list[str] = []

                with state.global_lock:
                    for physical_name, server in list(state.active_servers.items()):
                        proc = server["process"]
                        port = server["port"]

                        if proc.poll() is not None:
                            logger.error(f"💥 {physical_name}:{port} crashed")
                            state.active_servers.pop(physical_name, None)
                            state.server_idle_time.pop(physical_name, None)
                            continue

                        idle = now - state.server_idle_time.get(physical_name, 0)
                        if AUTO_SWAP_ENABLED and idle > KEEP_ALIVE_SECONDS:
                            logger.info(f"⏰ {physical_name} idle {idle:.0f}s, unloading")
                            to_unload.append(physical_name)

                for physical_name in to_unload:
                    if physical_name in state.active_servers:
                        shutdown_server(physical_name, reason="idle", fast=True)

                state._health_monitor_running.wait(HEALTH_CHECK_INTERVAL)
            except Exception as e:
                logger.error(f"Error in health monitor cycle: {type(e).__name__}: {e}", exc_info=True)
                state._health_monitor_running.wait(10)
    finally:
        state._health_monitor_running.clear()
