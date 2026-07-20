import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ErrorAnalysis:
    """Structured analysis of a backend error."""
    error_type: str
    severity: str
    raw_message: str
    explanation: str
    suggestions: list[str]


class ErrorDetector:
    """Detects and analyses known error patterns in backend logs."""

    # Regex error patterns (priority order)
    ERROR_PATTERNS = {
        "cuda_out_of_memory": [
            r"CUDA out of memory",
            r"cuda runtime error.*out of memory",
            r"RuntimeError.*CUDA out of memory",
            r"torch\.cuda\.OutOfMemoryError",
            r"cuMemCreate.*out of memory",
            r"CUDA error.*out of memory",
            r"marlin_gemm",  # FP8/GPTQ fragmentation OOM during inference
        ],
        "cuda_allocation_failed": [
            r"Failed to allocate",
            r"allocation failed",
            r"cannot allocate memory",
            r"memory allocation failed",
        ],
        "tensor_parallel_failed": [
            r"tensor.?parallel",
            r"cannot split.*GPU",
            r"not enough memory for tensor parallel",
            r"TP size.*exceeds",
        ],
        "model_too_large": [
            r"model.*too large",
            r"exceeds.*memory",
            r"model exceeds",
            r"insufficient VRAM",
        ],
        "context_too_large": [
            r"max_model_len.*too large",
            r"context.*exceeds",
            r"n_ctx.*too large",
            r"context length.*exceeds",
        ],
        "invalid_model_path": [
            r"No such file or directory.*model",
            r"cannot find.*model",
            r"path does not exist",
            r"model file not found",
        ],
        "tokenizer_load_failed": [
            r"Failed to load tokenizer",
            r"cannot load tokenizer",
            r"tokenizer.*not found",
        ],
    }

    SUGGESTIONS = {
        "cuda_out_of_memory": [
            "VRAM fragmentation (marlin_gemm/FP8): PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set — restart the server to apply it",
            "Reduce max_num_seqs or max_num_batched_tokens in the base config",
            "Reduce max_model_len to shrink KV cache (e.g. 262K → 131K)",
            "Use --kv-cache-dtype fp8 to halve KV cache VRAM",
            "Reduce gpu_memory_utilization in the base config (e.g. 0.92 → 0.88)",
            "Use tensor-parallel across multiple GPUs",
        ],
        "cuda_allocation_failed": [
            "GPU memory may be fragmented — restart the server",
            "Check for other processes using VRAM (nvidia-smi)",
            "Reduce gpu_memory_utilization in the base config",
            "Restart NVIDIA persistence daemon: sudo systemctl restart nvidia-persistenced",
        ],
        "tensor_parallel_failed": [
            "Check that all GPUs have enough free VRAM",
            "Reduce tensor-parallel-size",
            "Set GPU_MEMORY_THRESHOLD_GB higher to ensure VRAM is clear before loading",
            "Verify CUDA_VISIBLE_DEVICES is set correctly",
        ],
        "model_too_large": [
            "Model does not fit in available VRAM",
            "Options: (1) add more GPUs, (2) use a quantized version, (3) use a smaller model",
            "Reduce max_model_len to save VRAM on the KV cache",
        ],
        "context_too_large": [
            "Context length is too large for available VRAM",
            "Reduce n_ctx in the base config",
            "Use --yarn-scale-factor to extend context with less VRAM via YaRN",
        ],
        "invalid_model_path": [
            "Check the model path in your base config (.allm file)",
            "Verify the file exists: ls -la /path/to/model",
            "Check file permissions",
        ],
        "tokenizer_load_failed": [
            "Check the tokenizer path in your base config",
            "Verify tokenizer.model or tokenizer.json exists in the model directory",
            "Make sure the correct --chat-template-file is set",
        ],
    }

    @staticmethod
    def analyze_log(log_content: str) -> Optional[ErrorAnalysis]:
        """
        Scan log content for known error patterns.
        Returns an ErrorAnalysis, or None if no pattern matched.
        """
        if not log_content:
            return None

        for error_type, patterns in ErrorDetector.ERROR_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, log_content, re.IGNORECASE | re.MULTILINE)
                if match:
                    raw_msg = match.group(0)
                    explanation = ErrorDetector._get_explanation(error_type, log_content)
                    suggestions = ErrorDetector.SUGGESTIONS.get(error_type, [])

                    return ErrorAnalysis(
                        error_type=error_type,
                        severity="ERROR",
                        raw_message=raw_msg,
                        explanation=explanation,
                        suggestions=suggestions,
                    )

        return None

    @staticmethod
    def _get_explanation(error_type: str, log_content: str) -> str:
        """Return a plain-English explanation for each error type."""
        explanations = {
            "cuda_out_of_memory": (
                "CUDA ran out of memory during allocation or inference. "
                "The GPU does not have enough VRAM for this model with the current settings."
            ),
            "cuda_allocation_failed": (
                "Failed to allocate CUDA memory. The GPU may be fragmented or out of space."
            ),
            "tensor_parallel_failed": (
                "Failed to split the model across GPUs (tensor-parallel). "
                "One or more GPUs may not have enough free VRAM."
            ),
            "model_too_large": (
                "The model is larger than available GPU memory. "
                "You need more GPUs, a quantized version, or a smaller model."
            ),
            "context_too_large": (
                "The configured context length (n_ctx) is too large for available VRAM. "
                "Reduce n_ctx or limit parallel requests."
            ),
            "invalid_model_path": (
                "Model file not found at the configured path. "
                "Check that the file exists and the path in your base config is correct."
            ),
            "tokenizer_load_failed": (
                "Tokenizer not found or failed to load. "
                "Check the tokenizer path in your base config."
            ),
        }
        return explanations.get(error_type, "Unknown error")

    @staticmethod
    def analyze_exit_code(exit_code: int, backend: str) -> Optional[str]:
        """
        Interpret a process exit code to infer the cause of failure.
        Returns a human-readable description, or None if the code is unknown.
        """
        if exit_code == 0:
            return None  # clean exit

        if exit_code == 1:
            return "Generic failure (startup error or fatal exception)"

        if exit_code == 127:
            return f"Command not found — check the {backend} binary path"

        if exit_code == -9 or exit_code == 137:
            return "Process was killed (SIGKILL — likely OOM killer)"

        if exit_code == -15 or exit_code == 143:
            return "Process terminated (SIGTERM — normal shutdown)"

        if exit_code > 128:
            signal_num = exit_code - 128
            return f"Terminated by signal {signal_num} (abnormal exit)"

        return f"Exit code {exit_code} (unknown error)"


# ==============================================================================
# Failure log — one JSONL row per backend crash, for `allma errors` and for
# the auto-degrade retry policy to consult crash history.
# ==============================================================================
_FAILURE_LOG_LOCK = threading.Lock()
_FAILURE_LOG_MAX_BYTES = 5 * 1024 * 1024   # rotate at 5 MB; keeps `allma errors` snappy


def _failure_log_path() -> Path:
    # Late import — core.config pulls in logging setup that may not want to
    # boot for a lightweight read (e.g. `allma errors` invoked cold).
    from core.config import ALLMA_LOG_DIR
    return Path(ALLMA_LOG_DIR) / "errors.jsonl"


def _rotate_failure_log(path: Path) -> None:
    """Same 20MB→.1→.2 pattern as backend logs, smaller ceiling."""
    try:
        if not path.exists() or path.stat().st_size < _FAILURE_LOG_MAX_BYTES:
            return
        older = path.with_suffix(path.suffix + ".1")
        if older.exists():
            older.unlink()
        path.rename(older)
    except Exception:
        pass


def record_failure(
    base_name: str,
    analysis: Optional[ErrorAnalysis],
    *,
    exit_code: Optional[int] = None,
    log_tail: str = "",
    context: Optional[dict[str, Any]] = None,
) -> None:
    """Append one JSONL row describing a backend crash.

    Called from health_monitor (backend died mid-life) and loader (backend
    failed to start). Thread-safe via _FAILURE_LOG_LOCK. The row is stable
    schema so `allma errors` and the auto-degrade policy can rely on it.
    """
    path = _failure_log_path()
    row: dict[str, Any] = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": base_name,
        "error_type": analysis.error_type if analysis else "unknown",
        "severity": analysis.severity if analysis else "error",
        "explanation": analysis.explanation if analysis else "",
        "suggestions": list(analysis.suggestions) if analysis else [],
        "raw_message": (analysis.raw_message if analysis else "").strip()[:400],
        "exit_code": exit_code,
        "log_tail": (log_tail or "")[-1200:],   # last ~1200 chars, no full dump
        "context": context or {},
    }
    try:
        with _FAILURE_LOG_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            _rotate_failure_log(path)
            with path.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        # Never let logging failure crash the caller. Silent by design —
        # the health monitor already logs the crash via `logger.error`.
        pass


def read_failures(limit: int = 50, model: Optional[str] = None) -> list[dict]:
    """Read the last `limit` failure rows, newest first. Optionally filter
    by model name (substring match, case-insensitive)."""
    path = _failure_log_path()
    if not path.exists():
        return []
    try:
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except Exception:
        return []
    if model:
        needle = model.lower()
        rows = [r for r in rows if needle in str(r.get("model", "")).lower()]
    return rows[-limit:][::-1]


def tail_file(file_path: str, lines: int = 50, max_bytes: int = 256 * 1024) -> str:
    """Read the last N lines of a file without loading the whole thing.

    Backend logs can grow to hundreds of MB (vLLM tracebacks + tensor dumps);
    the previous `f.readlines()` implementation loaded everything into RAM
    and blocked the caller (the health monitor) for seconds. Here we seek
    from the end and read at most `max_bytes` — enough for hundreds of log
    lines in practice, bounded regardless of file size.
    """
    try:
        with open(file_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            chunk = f.read().decode("utf-8", errors="replace")
        tail = chunk.splitlines(keepends=True)[-lines:]
        return "".join(tail)
    except Exception as e:
        return f"Error reading file: {e}"
