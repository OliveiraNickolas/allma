"""
Hardware detection and model auto-calibration.

Detects GPU capabilities at startup and calibrates each model's parameters
based on available VRAM, compute capability, and hardware profile.
"""
import asyncio
import json
import re
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

from core.config import logger, BASE_MODELS, ALLMA_LOG_DIR
from core.gpu import (
    get_all_gpus,
    _estimate_kv_cache_gb,
    _calc_model_size_gb,
    get_model_vram_need,
)


@dataclass
class GPUCapability:
    """Information about a detected GPU."""
    index: int
    name: str                    # "RTX 3090", "RTX 4090"
    compute_capability: str      # "8.9", "8.6"
    sm_count: int                # Streaming multiprocessors
    total_memory_gb: float
    free_memory_gb: float
    driver_supported: bool


@dataclass
class HardwareProfile:
    """Snapshot of detected hardware at startup."""
    driver_version: str          # "550.127"
    cuda_version: str            # "12.4"
    gpus: List[GPUCapability]
    total_vram_gb: float         # Sum of all GPUs
    available_vram_gb: float     # Free now
    max_contiguous_gb: float     # Largest contiguous block
    detected_at: str             # ISO timestamp
    detection_duration_ms: float


@dataclass
class CalibrationResult:
    """Calibration recommendations for a specific model."""
    model_name: str
    backend: str                 # "vllm" or "llama.cpp"
    recommended_tp: int          # Tensor Parallel size
    recommended_ubatch_size: int  # For vLLM
    recommended_n_batch: int      # For llama.cpp
    recommended_n_ctx: int        # Context length
    recommended_cache_dtype: str  # "auto", "fp16", "q8_0", "fp8"
    safety_margin_gb: float       # VRAM reserve (default 2GB)
    estimated_load_time_sec: float
    confidence: str              # "high", "medium", "low"
    warnings: List[str]          # Potential issues
    calibrated_at: str           # ISO timestamp
    estimated_vram_need_gb: float


class BootstrapDetector:
    """Hardware detection and model calibration system."""

    @staticmethod
    async def detect_hardware() -> HardwareProfile:
        """
        Detects hardware at startup.

        Returns:
            HardwareProfile with GPU info, driver version, CUDA version.

        Raises:
            RuntimeError: If no GPUs detected or detection fails.
        """
        start_time = time.time()

        try:
            # Get GPU info
            gpu_list = get_all_gpus()
            if not gpu_list:
                raise RuntimeError("No GPUs detected. Continuing with CPU-only mode.")

            # Parse driver and CUDA versions
            driver_version, cuda_version = BootstrapDetector._get_driver_cuda_versions()

            # Enhance each GPU with compute capability
            gpus = []
            for gpu_data in gpu_list:
                compute_cap = BootstrapDetector._get_compute_capability(gpu_data["index"])
                gpu = GPUCapability(
                    index=gpu_data["index"],
                    name=BootstrapDetector._get_gpu_name(gpu_data["index"]),
                    compute_capability=compute_cap,
                    sm_count=BootstrapDetector._get_sm_count(gpu_data["index"]),
                    total_memory_gb=gpu_data["total_gb"],
                    free_memory_gb=gpu_data["free_gb"],
                    driver_supported=True,  # If we got here, driver supports it
                )
                gpus.append(gpu)

            # Calculate total and max contiguous
            total_vram = sum(g.total_memory_gb for g in gpus)
            available_vram = sum(g.free_memory_gb for g in gpus)
            max_contiguous = max(g.free_memory_gb for g in gpus) if gpus else 0

            duration_ms = (time.time() - start_time) * 1000

            profile = HardwareProfile(
                driver_version=driver_version,
                cuda_version=cuda_version,
                gpus=gpus,
                total_vram_gb=total_vram,
                available_vram_gb=available_vram,
                max_contiguous_gb=max_contiguous,
                detected_at=datetime.now().isoformat(),
                detection_duration_ms=duration_ms,
            )

            return profile

        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")
            raise

    @staticmethod
    def _get_driver_cuda_versions() -> tuple[str, str]:
        """
        Extracts driver version and CUDA version from nvidia-smi.

        Returns:
            (driver_version, cuda_version) as strings, or ("unknown", "unknown")
        """
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            output = result.stdout
            driver_match = re.search(r"Driver Version:\s+([0-9.]+)", output)
            cuda_match = re.search(r"CUDA Version:\s+([0-9.]+)", output)

            driver = driver_match.group(1) if driver_match else "unknown"
            cuda = cuda_match.group(1) if cuda_match else "unknown"

            return driver, cuda
        except Exception as e:
            logger.debug(f"Could not parse driver/CUDA versions: {e}")
            return "unknown", "unknown"

    @staticmethod
    def _get_compute_capability(gpu_index: int) -> str:
        """
        Gets compute capability of GPU (e.g., "8.9" for RTX 4090).

        Attempts nvidia-smi first, falls back to conservative estimate.
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Could not get compute capability for GPU {gpu_index}: {e}")

        # Fallback: default to conservative 7.5 (Turing)
        return "7.5"

    @staticmethod
    def _get_gpu_name(gpu_index: int) -> str:
        """Gets GPU name like 'RTX 3090' or 'RTX 4090'."""
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return f"GPU {gpu_index}"

    @staticmethod
    def _get_sm_count(gpu_index: int) -> int:
        """Gets streaming multiprocessor count (approximate)."""
        # This would require cuda-capable GPU query, fallback to 0
        return 0

    @staticmethod
    async def calibrate_for_model(
        base_name: str,
        model_size_gb: float,
        hardware_profile: HardwareProfile,
        config: dict,
    ) -> CalibrationResult:
        """
        Generates calibration recommendations for a model.

        Args:
            base_name: Model name (e.g., "Qwen3.5-27b")
            model_size_gb: Model weights + KV cache estimate
            hardware_profile: Result from detect_hardware()
            config: Base config dict

        Returns:
            CalibrationResult with recommendations
        """
        backend = config.get("backend", "vllm")
        max_model_len = int(config.get("max_model_len") or config.get("n_ctx") or "40960")
        safety_margin = 2.0  # GB

        # Estimate KV cache
        model_path = config.get("path", "")
        kv_cache_gb = 0
        try:
            kv_cache_gb = _estimate_kv_cache_gb(model_path, max_model_len, "auto")
        except Exception as e:
            logger.debug(f"Could not estimate KV cache: {e}")

        total_need_gb = model_size_gb + kv_cache_gb + safety_margin

        warnings: List[str] = []
        confidence = "high"

        # Decide TP (Tensor Parallel)
        max_single_gpu_gb = max(g.total_memory_gb for g in hardware_profile.gpus) if hardware_profile.gpus else 24
        util_factor = 0.95  # Use 95% of GPU memory for model

        if backend == "vllm":
            # vLLM can use multiple GPUs easily
            if total_need_gb <= max_single_gpu_gb * util_factor:
                recommended_tp = 1
            elif total_need_gb <= max_single_gpu_gb * util_factor * 2 and len(hardware_profile.gpus) >= 2:
                recommended_tp = 2
            elif total_need_gb <= max_single_gpu_gb * util_factor * 3 and len(hardware_profile.gpus) >= 3:
                recommended_tp = 3
            else:
                # Too large, reduce context or quantize
                recommended_tp = min(len(hardware_profile.gpus), 4)
                confidence = "low"
                warnings.append(
                    f"Model is very large ({model_size_gb:.1f}GB). "
                    f"Consider quantization (Q4/Q5) or reducing max_model_len."
                )
        else:
            # llama.cpp: TP not standard, just use single GPU best fit
            recommended_tp = 1

        if recommended_tp > 1:
            warnings.append(f"TP={recommended_tp} requires {recommended_tp} GPU slots")

        # Decide ubatch-size (vLLM only)
        recommended_ubatch_size = 1024
        if backend == "vllm":
            # Start with 1024, reduce if tight on VRAM
            available_for_batch = (max_single_gpu_gb * util_factor) - (model_size_gb / recommended_tp)
            if available_for_batch < 3:
                recommended_ubatch_size = 256
                warnings.append("Low VRAM for batching, reduced ubatch-size to 256")
            elif available_for_batch < 6:
                recommended_ubatch_size = 512
            elif total_need_gb > hardware_profile.available_vram_gb * 0.8:
                recommended_ubatch_size = 512
                warnings.append("VRAM utilization high, reduced ubatch-size for stability")

        if recommended_ubatch_size > 1024:
            warnings.append("High batch size, monitor inference latency")

        # Decide n_batch (llama.cpp)
        recommended_n_batch = 1024
        if backend == "llama.cpp":
            if hardware_profile.available_vram_gb < 10:
                recommended_n_batch = 256
            elif hardware_profile.available_vram_gb < 5:
                recommended_n_batch = 128

        # Decide cache dtype
        recommended_cache_dtype = "auto"
        compute_cap_float = float(hardware_profile.gpus[0].compute_capability) if hardware_profile.gpus else 7.5

        if compute_cap_float >= 8.9:
            # RTX 4090, A100, H100: use auto (fp16)
            recommended_cache_dtype = "auto"
        elif compute_cap_float >= 8.0:
            # RTX 3090, A30: use fp16 or q8_0
            recommended_cache_dtype = "fp16"
        else:
            # Older GPUs: use q8_0 for safety
            recommended_cache_dtype = "q8_0"
            warnings.append("Older GPU compute capability, using q8_0 cache (reduced precision)")

        # Decide n_ctx (context length)
        recommended_n_ctx = max_model_len
        if hardware_profile.available_vram_gb < 10:
            recommended_n_ctx = max_model_len // 2
            warnings.append(f"Low VRAM, reduced context to {recommended_n_ctx}K")
        elif hardware_profile.available_vram_gb < 5:
            recommended_n_ctx = max_model_len // 4
            warnings.append(f"Very low VRAM, reduced context to {recommended_n_ctx}K")

        # Estimate load time (vLLM ~10s, llama.cpp ~20s + TP overhead)
        estimated_load_time = 10 if backend == "vllm" else 20
        if recommended_tp > 1:
            estimated_load_time += 5

        return CalibrationResult(
            model_name=base_name,
            backend=backend,
            recommended_tp=recommended_tp,
            recommended_ubatch_size=recommended_ubatch_size,
            recommended_n_batch=recommended_n_batch,
            recommended_n_ctx=recommended_n_ctx,
            recommended_cache_dtype=recommended_cache_dtype,
            safety_margin_gb=safety_margin,
            estimated_load_time_sec=estimated_load_time,
            confidence=confidence,
            warnings=warnings,
            calibrated_at=datetime.now().isoformat(),
            estimated_vram_need_gb=total_need_gb,
        )

    @staticmethod
    def validate_calibration(calib: CalibrationResult) -> List[str]:
        """
        Validates calibration result.

        Returns:
            List of errors (empty = valid)
        """
        errors = []

        if calib.recommended_tp < 1:
            errors.append("TP must be >= 1")

        if calib.recommended_ubatch_size < 64 or calib.recommended_ubatch_size > 4096:
            errors.append(f"ubatch-size {calib.recommended_ubatch_size} out of range [64-4096]")

        if calib.recommended_cache_dtype not in ["auto", "fp16", "fp8", "q8_0"]:
            errors.append(f"Unknown cache_dtype: {calib.recommended_cache_dtype}")

        return errors

    @staticmethod
    def save_profile_to_file(profile: HardwareProfile, path: Optional[str] = None) -> str:
        """
        Saves hardware profile to JSON for debugging.

        Returns:
            Path to saved file
        """
        if path is None:
            path = str(ALLMA_LOG_DIR / "bootstrap-profile.json")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "detected_at": profile.detected_at,
            "detection_duration_ms": profile.detection_duration_ms,
            "driver_version": profile.driver_version,
            "cuda_version": profile.cuda_version,
            "total_vram_gb": profile.total_vram_gb,
            "available_vram_gb": profile.available_vram_gb,
            "max_contiguous_gb": profile.max_contiguous_gb,
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "compute_capability": g.compute_capability,
                    "total_memory_gb": g.total_memory_gb,
                    "free_memory_gb": g.free_memory_gb,
                }
                for g in profile.gpus
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return path

    @staticmethod
    def save_calibrations_to_file(calibrations: Dict[str, CalibrationResult], path: Optional[str] = None) -> str:
        """
        Saves calibration results to JSON.

        Returns:
            Path to saved file
        """
        if path is None:
            path = str(ALLMA_LOG_DIR / "bootstrap-calibrations.json")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {calib.model_name: asdict(calib) for calib in calibrations.values()}

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return path
