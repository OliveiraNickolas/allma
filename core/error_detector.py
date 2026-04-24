import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ErrorAnalysis:
    """Structured analysis of a backend error."""
    error_type: str
    severity: str
    raw_message: str
    explanation: str
    suggestions: list[str]
    auto_fix_available: bool = False
    auto_fix_action: Optional[str] = None  # e.g. "reduce_ubatch_size"


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
            "Reduce --ubatch-size (e.g. 1024 → 512 → 256)",
            "Use KV cache quantization (--cache-type-k q8_0 instead of fp16)",
            "Reduce max_model_len (e.g. 262K → 131K)",
            "Use tensor-parallel across multiple GPUs",
            "Use a quantized model (Q4, Q5 instead of fp16)",
            "Increase --defrag-thold to defragment VRAM",
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
                    auto_fix_action = ErrorDetector._get_auto_fix_action(error_type, log_content)

                    return ErrorAnalysis(
                        error_type=error_type,
                        severity="ERROR",
                        raw_message=raw_msg,
                        explanation=explanation,
                        suggestions=suggestions,
                        auto_fix_available=auto_fix_action is not None,
                        auto_fix_action=auto_fix_action,
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
    def _get_auto_fix_action(error_type: str, log_content: str) -> Optional[str]:
        """
        Return an auto-fix action identifier if one is available.
        Returns None if the error requires manual intervention.
        """
        if error_type == "cuda_out_of_memory":
            if re.search(r"ubatch|batch", log_content, re.IGNORECASE):
                return "reduce_ubatch_size"
            if re.search(r"max_model_len|n_ctx|context", log_content, re.IGNORECASE):
                return "reduce_context_length"
            return "reduce_batch_params"

        return None

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


def tail_file(file_path: str, lines: int = 50) -> str:
    """Read the last N lines of a file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])
    except Exception as e:
        return f"Error reading file: {e}"
