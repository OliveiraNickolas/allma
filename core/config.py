"""
Constants, logging setup, and model config loading.
"""
import json
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Any, Dict

try:
    from rich import box
    from rich.align import Align
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ==============================================================================
# FORMATTERS
# ==============================================================================
class JSONFormatter(logging.Formatter):
    """Structured JSON logging for production-friendly logs."""
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)
        extra_data = getattr(record, "extra_data", None)
        if extra_data is not None:
            log_entry["data"] = extra_data
        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Colored console output with emojis for human readability."""
    colors = {
        "DEBUG": "\033[36m ",
        "INFO": "\033[32m ",
        "WARNING": "\033[33m ",
        "ERROR": "\033[31m ",
        "CRITICAL": "\033[35m ",
    }
    reset = "\033[0m"
    datefmt = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        emoji = {
            logging.DEBUG: "🔍",
            logging.INFO: "ℹ️",
            logging.WARNING: "⚠️",
            logging.ERROR: "❌",
            logging.CRITICAL: "💥",
        }.get(record.levelno, "📝")
        color = self.colors.get(record.levelname, "")
        level_style = f"{color}{record.levelname}{self.reset}"
        return f"{self.formatTime(record)} - {emoji} {level_style} - {record.getMessage()}"


# ==============================================================================
# CONSTANTS (environment variables override defaults)
# ==============================================================================
ALLAMA_PORT = int(os.environ.get("ALLAMA_PORT", "9000"))
VLLM_BASE_PORT = int(os.environ.get("VLLM_BASE_PORT", "8000"))
LLAMA_BASE_PORT = int(os.environ.get("LLAMA_BASE_PORT", "9001"))
KEEP_ALIVE_SECONDS = int(os.environ.get("KEEP_ALIVE_SECONDS", "600"))
HEALTH_CHECK_INTERVAL = int(os.environ.get("HEALTH_CHECK_INTERVAL", "60"))
AUTO_SWAP_ENABLED = os.environ.get("AUTO_SWAP_ENABLED", "true").lower() == "true"
MAX_MESSAGES = int(os.environ.get("MAX_MESSAGES", "15"))

SCRIPT_DIR = Path(__file__).parent.parent  # allama/ root
ALLAMA_LOG_DIR = Path(os.environ.get("ALLAMA_LOG_DIR", str(SCRIPT_DIR / "logs")))
CONFIG_DIR = Path(os.environ.get("ALLAMA_CONFIG_DIR", str(SCRIPT_DIR / "configs")))
PATH_TO_ALLAMA = os.environ.get("PATH_TO_ALLAMA", str(SCRIPT_DIR.parent))
def _find_llama_server() -> str:
    """Find llama-server binary: env var → PATH → common locations."""
    from shutil import which
    # 1. Explicit env var
    if (env := os.environ.get("LLAMA_CPP_PATH")):
        return env
    # 2. llama-server on PATH
    if (found := which("llama-server")):
        return found
    # 3. Common build locations relative to HOME
    home = Path.home()
    candidates = [
        home / "AI" / "llama.cpp" / "build" / "bin" / "llama-server",
        home / "llama.cpp" / "build" / "bin" / "llama-server",
        Path("/usr/local/bin/llama-server"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # Fallback — will fail at runtime with a clear error
    return "llama-server"

LLAMA_CPP_PATH = _find_llama_server()

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
ALLAMA_LOG_DIR.mkdir(exist_ok=True)

file_handler = logging.handlers.RotatingFileHandler(
    str(ALLAMA_LOG_DIR / "allama.log"),
    maxBytes=10_485_760,
    backupCount=5,
    encoding="utf-8",
)
file_handler.setFormatter(JSONFormatter())

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColoredFormatter())

root_logger = logging.getLogger()
root_logger.handlers = []
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
root_logger.setLevel(logging.INFO)

logger = logging.getLogger("Allama")

for _uv in ["uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.info"]:
    _uv_log = logging.getLogger(_uv)
    _uv_log.handlers = []
    _uv_log.addHandler(console_handler)
    _uv_log.addHandler(file_handler)
    _uv_log.setLevel(logging.ERROR)

logging.getLogger("httpx").setLevel(logging.ERROR)


# ==============================================================================
# UTILITIES
# ==============================================================================
def format_user_agent(ua: str) -> str:
    """Simplify user agent for logging."""
    if not ua or ua == "unknown":
        return "unknown"
    ua_lower = ua.lower()
    if "claude" in ua_lower or "claude-code" in ua_lower:
        if "vscode" in ua_lower or "extension" in ua_lower:
            return "Claude - VSCode"
        return "Claude - Terminal"
    if "openwebui" in ua_lower:
        return "OpenWebUI"
    if "fastapi" in ua_lower or "uvicorn" in ua_lower:
        return "FastAPI"
    if "python" in ua_lower and "requests" in ua_lower:
        return "Python/requests"
    if ua.startswith("Mozilla/5.0") and "aiohttp" in ua_lower and "python" in ua_lower:
        return "OpenWebUI"
    if "python" in ua_lower and "aiohttp" in ua_lower:
        return "OpenWebUI"
    if "curl" in ua_lower:
        return "curl"
    if "wget" in ua_lower:
        return "wget"
    if "Mozilla/5.0" in ua and "Chrome/" in ua and "Safari/537.36" in ua and "Mobile" not in ua and "Tablet" not in ua:
        return "OpenWebUI Desktop"
    if "Mozilla/5.0" in ua and "CriOS/" in ua and "Mobile" in ua:
        return "OpenWebUI Mobile"
    if "Mozilla/5.0" in ua and "Version/26.3" in ua and "Safari/605.1" in ua:
        if "iPhone" in ua:
            return "OpenWebUI Mobile"
        if "iPad" in ua or "Tablet" in ua:
            return "OpenWebUI Tablet"
        return "OpenWebUI Desktop"
    if len(ua) > 50:
        return ua[:47] + "..."
    return ua


# ==============================================================================
# MODEL CONFIG LOADING
# ==============================================================================
def load_models_from_configs() -> tuple[dict, dict]:
    try:
        import sys as _sys
        _sys.path.insert(0, str(SCRIPT_DIR))
        from configs.loader import load_models_from_configs as _load
        return _load(str(CONFIG_DIR))
    except FileNotFoundError as e:
        logger.warning(f"Config directory not found: {e}")
        return {}, {}
    except Exception as e:
        logger.error(f"Failed to load configs: {e}")
        return {}, {}


PHYSICAL_MODELS, LOGICAL_MODELS = load_models_from_configs()
