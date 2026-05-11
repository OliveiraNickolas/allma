"""
Constants, logging setup, and model config loading.
"""
import json
import logging
import logging.handlers
import os
import sys
from pathlib import Path

try:
    from rich.console import Console as _Console  # noqa: F401 — presence check only
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ==============================================================================
# .env LOADER — must run before any os.environ reads
# ==============================================================================
def _load_dotenv() -> None:
    """Load .env from the project root into os.environ (existing vars take priority)."""
    env_file = Path(__file__).parent.parent / ".env"
    if not env_file.exists():
        return
    try:
        with env_file.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Strip inline comments (e.g. KEY=300 # comment)
                if not value.startswith(('"', "'")):
                    value = value.partition("#")[0].strip()
                value = value.strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass  # never crash on .env parse errors

_load_dotenv()


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
    """Minimal retro log formatter with Unicode symbols."""
    colors = {
        "DEBUG": "\033[36m",      # cyan
        "INFO": "\033[32m",       # green
        "WARNING": "\033[33m",    # yellow
        "ERROR": "\033[31m",      # red
        "CRITICAL": "\033[35m",   # magenta
    }
    reset = "\033[0m"
    datefmt = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        symbol = {
            logging.DEBUG: "○",
            logging.INFO: "◆",
            logging.WARNING: "⚠",
            logging.ERROR: "✕",
            logging.CRITICAL: "⚡",
        }.get(record.levelno, "·")
        color = self.colors.get(record.levelname, "")
        level_name = f"{color}{record.levelname:<8}{self.reset}"
        time_str = self.formatTime(record)
        message = record.getMessage()
        return f"{time_str} │ {level_name} {symbol} {message}"


# ==============================================================================
# CONSTANTS (environment variables override defaults)
# ==============================================================================
def _parse_int(key: str, default: int) -> int:
    """Parse env var as int with validation."""
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        print(f"ERROR: {key}={os.environ.get(key)} is not a valid integer. Using default: {default}")
        return default

ALLMA_PORT = _parse_int("ALLMA_PORT", 9000)
VLLM_BASE_PORT = _parse_int("VLLM_BASE_PORT", 8000)
LLAMA_BASE_PORT = _parse_int("LLAMA_BASE_PORT", 9001)
KEEP_ALIVE_SECONDS = _parse_int("KEEP_ALIVE_SECONDS", 600)
HEALTH_CHECK_INTERVAL = _parse_int("HEALTH_CHECK_INTERVAL", 60)
AUTO_SWAP_ENABLED = os.environ.get("AUTO_SWAP_ENABLED", "true").lower() == "true"
MAX_MESSAGES = _parse_int("MAX_MESSAGES", 50)

SCRIPT_DIR = Path(__file__).parent.parent  # allama/ root
ALLMA_LOG_DIR = Path(os.environ.get("ALLMA_LOG_DIR", str(SCRIPT_DIR / "logs")))
CONFIG_DIR = Path(os.environ.get("ALLMA_CONFIG_DIR", str(SCRIPT_DIR / "configs")))
PATH_TO_ALLMA = os.environ.get("PATH_TO_ALLMA", str(SCRIPT_DIR.parent))
def _find_llama_server() -> str:
    """Find llama-server binary: env var → PATH → common build locations.

    Returns the path/command to use. Two possible forms:
      - A path to the llama-server binary  (preferred, full features)
      - The sentinel "llama-cpp-python"    (fallback via pip package)
    Use LLAMA_CPP_BACKEND to check which form is active.
    """
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
        home / ".local" / "bin" / "llama-server",
        home / "AI" / "llama.cpp" / "build" / "bin" / "llama-server",
        home / "llama.cpp" / "build" / "bin" / "llama-server",
        Path("/usr/local/bin/llama-server"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # 4. llama-cpp-python Python package as fallback
    try:
        import importlib.util
        if importlib.util.find_spec("llama_cpp") is not None:
            return "llama-cpp-python"
    except Exception:
        pass
    # Fallback — will fail at runtime with a clear error
    return "llama-server"

LLAMA_CPP_PATH = _find_llama_server()
# True when using llama-cpp-python package instead of the native binary
LLAMA_CPP_PYTHON_BACKEND = LLAMA_CPP_PATH == "llama-cpp-python"


def _find_vllm() -> str:
    """Find vllm binary: env var → venv next to this package → PATH."""
    from shutil import which
    # 1. Explicit env var
    if (env := os.environ.get("VLLM_PATH")):
        return env
    # 2. venv sibling to the core/ package (most common allma setup)
    venv_bin = SCRIPT_DIR / "venv" / "bin" / "vllm"
    if venv_bin.exists():
        return str(venv_bin)
    # 3. vllm on PATH (activated venv or system install)
    if (found := which("vllm")):
        return found
    # Fallback — will fail at runtime with a clear error
    return "vllm"

VLLM_PATH = _find_vllm()

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
ALLMA_LOG_DIR.mkdir(exist_ok=True)

try:
    file_handler = logging.handlers.RotatingFileHandler(
        str(ALLMA_LOG_DIR / "allma.log"),
        maxBytes=10_485_760,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(JSONFormatter())
except Exception as _log_err:
    file_handler = logging.StreamHandler(sys.stderr)
    file_handler.setFormatter(JSONFormatter())
    print(f"WARNING: Could not open log file, falling back to stderr: {_log_err}", file=sys.stderr)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColoredFormatter())

root_logger = logging.getLogger()
root_logger.handlers = []
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
root_logger.setLevel(logging.INFO)

logger = logging.getLogger("Allma")

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
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            from configs.loader import load_models_from_configs as _load
        finally:
            if str(SCRIPT_DIR) in sys.path:
                sys.path.remove(str(SCRIPT_DIR))
        return _load(str(CONFIG_DIR))
    except FileNotFoundError as e:
        logger.warning(f"Config directory not found: {e}")
        return {}, {}
    except Exception as e:
        logger.error(f"Failed to load configs: {e}")
        return {}, {}


BASE_MODELS, PROFILE_MODELS = load_models_from_configs()
