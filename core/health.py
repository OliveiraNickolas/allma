"""
Health monitor: idle-timeout unloading and crash detection.
"""
import time
from pathlib import Path
from core.config import logger, AUTO_SWAP_ENABLED, KEEP_ALIVE_SECONDS, HEALTH_CHECK_INTERVAL, ALLMA_LOG_DIR
import core.state as state
from core.process import shutdown_server
from core.error_detector import ErrorDetector, tail_file


def health_monitor():
    state._health_monitor_running.set()
    logger.info("Health monitor started")
    try:
        while state._health_monitor_running.is_set():
            try:
                now = time.time()
                to_unload: list[str] = []

                with state.global_lock:
                    for base_name, server in list(state.active_servers.items()):
                        proc = server["process"]
                        port = server["port"]

                        if proc.poll() is not None:
                            # Try to extract crash reason from log
                            log_path = Path(ALLMA_LOG_DIR) / f"{base_name}.log"
                            if log_path.exists():
                                last_lines = tail_file(str(log_path), lines=100)
                                error_analysis = ErrorDetector.analyze_log(last_lines)
                                if error_analysis:
                                    logger.error(
                                        f"{base_name}:{port} crashed — {error_analysis.error_type}\n"
                                        f"   {error_analysis.explanation}"
                                    )
                                    for suggestion in error_analysis.suggestions:
                                        logger.error(f"   → {suggestion}")
                                    state.last_error_analysis[base_name] = error_analysis
                                else:
                                    logger.error(f"{base_name}:{port} crashed (exit code {proc.poll()})")
                            else:
                                logger.error(f"{base_name}:{port} crashed")

                            state.active_servers.pop(base_name, None)
                            state.server_idle_time.pop(base_name, None)
                            continue

                        if state.active_servers.get(base_name, {}).get("pin_loaded"):
                            continue
                        idle = now - state.server_idle_time.get(base_name, 0)
                        if AUTO_SWAP_ENABLED and idle > KEEP_ALIVE_SECONDS:
                            logger.info(f"⏰ {base_name} idle {idle:.0f}s, unloading")
                            to_unload.append(base_name)

                for base_name in to_unload:
                    if base_name in state.active_servers:
                        shutdown_server(base_name, reason="idle", fast=True)

                # BUG that hid here for a long time: threading.Event.wait()
                # returns *immediately* when the event is SET — and the running
                # flag is set for the entire lifetime of the monitor, so this
                # used to spin at 100% CPU forever. time.sleep() in 1s slices
                # gives us proper idle sleeping AND a quick reaction to
                # _health_monitor_running.clear() at shutdown.
                for _ in range(int(HEALTH_CHECK_INTERVAL)):
                    if not state._health_monitor_running.is_set():
                        return
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error in health monitor cycle: {type(e).__name__}: {e}", exc_info=True)
                for _ in range(10):
                    if not state._health_monitor_running.is_set():
                        return
                    time.sleep(1)
    finally:
        state._health_monitor_running.clear()
