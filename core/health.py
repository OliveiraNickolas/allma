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
    logger.info("🏥 Health monitor started")
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
                            # Tentar extrair razão do crash analisando log
                            log_path = Path(ALLMA_LOG_DIR) / f"{base_name}.log"
                            if log_path.exists():
                                last_lines = tail_file(str(log_path), lines=100)
                                error_analysis = ErrorDetector.analyze_log(last_lines)
                                if error_analysis:
                                    logger.error(
                                        f"💥 {base_name}:{port} crashed\n"
                                        f"   Razão: {error_analysis.error_type}\n"
                                        f"   {error_analysis.explanation}"
                                    )
                                    state.last_error_analysis[base_name] = error_analysis
                                else:
                                    logger.error(f"💥 {base_name}:{port} crashed (causa desconhecida)")
                            else:
                                logger.error(f"💥 {base_name}:{port} crashed")

                            state.active_servers.pop(base_name, None)
                            state.server_idle_time.pop(base_name, None)
                            continue

                        idle = now - state.server_idle_time.get(base_name, 0)
                        if AUTO_SWAP_ENABLED and idle > KEEP_ALIVE_SECONDS:
                            logger.info(f"⏰ {base_name} idle {idle:.0f}s, unloading")
                            to_unload.append(base_name)

                for base_name in to_unload:
                    if base_name in state.active_servers:
                        shutdown_server(base_name, reason="idle", fast=True)

                state._health_monitor_running.wait(HEALTH_CHECK_INTERVAL)
            except Exception as e:
                logger.error(f"Error in health monitor cycle: {type(e).__name__}: {e}", exc_info=True)
                state._health_monitor_running.wait(10)
    finally:
        state._health_monitor_running.clear()
