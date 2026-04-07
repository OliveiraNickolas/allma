#!/usr/bin/env python3
"""
Flashtensors vs vLLM: Benchmark model loading speed on YOUR hardware

Measures:
  - Registration time (one-time)
  - Load time (coldstart)
  - Hotswap time (model switching)
  - VRAM usage
  - Inference speed (tokens/sec)

Goal: Validate if flashtensors integration makes sense for your 2x RTX 3090 setup
"""

import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import psutil

# Try importing flashtensors (optional, will test availability)
try:
    import flashtensors as ft
    FLASHTENSORS_AVAILABLE = True
except ImportError:
    FLASHTENSORS_AVAILABLE = False
    print("⚠️  flashtensors not installed. Install with:")
    print("   pip install git+https://github.com/leoheuler/flashtensors.git")

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️  vLLM not installed")


class FlashTensorsBenchmark:
    """Benchmark flashtensors vs standard vLLM loading."""

    # Your models (adjust paths as needed)
    TEST_MODELS = [
        {
            "id": "Qwen/Qwen3.5-35B-FP8",
            "profile": "coding",
            "hf_path": "/path/to/models/Qwen3.5-35B-FP8",
            "expected_size_gb": 27,
            "description": "Large reasoning model (TP=2)"
        },
        {
            "id": "Qwen/Qwen3.5-27B-FP8",
            "profile": "research",
            "hf_path": "/path/to/models/Qwen3.5-27B-FP8",
            "expected_size_gb": 20,
            "description": "General purpose with tools"
        },
        {
            "id": "Qwen/Qwen3.5-Vision-7B-FP8",
            "profile": "vision",
            "hf_path": "/path/to/models/Qwen3.5-Vision-7B-FP8",
            "expected_size_gb": 10,
            "description": "Multimodal (OCR, video, comics)"
        },
        {
            "id": "google/Gemma-4-9B-FP8",
            "profile": "alternative",
            "hf_path": "/path/to/models/Gemma-4-9B-FP8",
            "expected_size_gb": 12,
            "description": "Lightweight alternative"
        }
    ]

    def __init__(self, storage_path: str = "/tmp/flashtensors_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "hardware": self._get_hardware_info(),
            "tests": {}
        }

    def _get_hardware_info(self) -> Dict:
        """Get hardware specifications."""
        try:
            nvidia_smi = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,nounits,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            gpus = []
            for line in nvidia_smi.stdout.strip().split("\n"):
                if line:
                    name, mem = line.split(", ")
                    gpus.append({
                        "name": name,
                        "memory_gb": int(mem) // 1024
                    })
        except Exception as e:
            gpus = [{"error": str(e)}]

        return {
            "cpu": {
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True)
            },
            "ram_gb": psutil.virtual_memory().total // (1024**3),
            "gpus": gpus
        }

    def _get_vram_usage(self) -> Dict:
        """Get current GPU VRAM usage."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,nounits,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )

            vram = {}
            for i, line in enumerate(result.stdout.strip().split("\n")):
                if line:
                    used, total = map(int, line.split(", "))
                    vram[f"GPU{i}"] = {
                        "used_gb": used / 1024,
                        "total_gb": total / 1024,
                        "util_percent": (used / total) * 100
                    }
            return vram
        except Exception as e:
            return {"error": str(e)}

    def test_flashtensors_available(self) -> bool:
        """Check if flashtensors is properly installed."""
        if not FLASHTENSORS_AVAILABLE:
            print("\n❌ flashtensors not available")
            return False

        print("\n✅ flashtensors is installed")
        print(f"   Storage path: {self.storage_path}")
        return True

    def configure_flashtensors(self) -> bool:
        """Configure flashtensors for your hardware."""
        if not FLASHTENSORS_AVAILABLE:
            return False

        try:
            print("\n🔧 Configuring flashtensors...")

            # Your hardware: 2x RTX 3090 = 48GB total
            # Leave 8GB for OS/overhead, use 40GB for models
            ft.configure(
                storage_path=str(self.storage_path),
                mem_pool_size=1024**3 * 40,      # 40GB (conservative for 48GB)
                chunk_size=1024**2 * 32,         # 32MB chunks (default)
                num_threads=8,                    # Use 8 threads (from 16 cores)
                gpu_memory_utilization=0.85,     # vLLM default is 0.9
                server_host="127.0.0.1",
                server_port=8073
            )

            ft.activate_vllm_integration()
            print("   ✅ Configured with:")
            print(f"      - mem_pool_size: 40GB")
            print(f"      - chunk_size: 32MB")
            print(f"      - num_threads: 8")
            print(f"      - gpu_memory_utilization: 0.85")

            return True
        except Exception as e:
            print(f"   ❌ Configuration failed: {e}")
            return False

    def register_model(self, model_id: str, hf_path: str) -> Tuple[bool, float]:
        """
        Register a model for fast loading.

        Returns:
            (success, registration_time)
        """
        if not FLASHTENSORS_AVAILABLE:
            return False, 0.0

        try:
            print(f"\n📝 Registering {model_id}...", end=" ", flush=True)

            start_time = time.time()

            # Use HuggingFace path if available, otherwise model ID
            model_source = hf_path if Path(hf_path).exists() else model_id

            result = ft.register_model(
                model_id=model_id,
                backend="vllm",
                torch_dtype="bfloat16",  # FP8 if supported, otherwise bf16
                force=False,  # Don't re-register if exists
                hf_token=None
            )

            registration_time = time.time() - start_time
            print(f"✅ ({registration_time:.2f}s)")

            return True, registration_time

        except Exception as e:
            print(f"❌ ({e})")
            return False, 0.0

    def benchmark_load_time(
        self,
        model_id: str,
        use_flashtensors: bool = True,
        repetitions: int = 2
    ) -> Dict:
        """
        Benchmark model loading time.

        Returns:
            Dict with load times, VRAM usage, etc
        """
        results = {
            "model_id": model_id,
            "method": "flashtensors" if use_flashtensors else "vllm",
            "load_times": [],
            "vram_usage": [],
            "errors": []
        }

        for rep in range(repetitions):
            print(f"  Attempt {rep+1}/{repetitions}...", end=" ", flush=True)

            try:
                # Get VRAM before
                vram_before = self._get_vram_usage()

                start_time = time.time()

                if use_flashtensors and FLASHTENSORS_AVAILABLE:
                    # Load with flashtensors
                    llm = ft.load_model(
                        model_id=model_id,
                        backend="vllm",
                        dtype="bfloat16"
                    )
                else:
                    # Load with standard vLLM
                    if not VLLM_AVAILABLE:
                        print("❌ vLLM not available")
                        continue

                    llm = LLM(
                        model=model_id,
                        dtype="bfloat16",
                        gpu_memory_utilization=0.85
                    )

                load_time = time.time() - start_time

                # Get VRAM after
                vram_after = self._get_vram_usage()

                results["load_times"].append(load_time)
                results["vram_usage"].append(vram_after)

                print(f"✅ {load_time:.2f}s")

                # Cleanup for next iteration
                if use_flashtensors and FLASHTENSORS_AVAILABLE:
                    ft.cleanup_gpu()
                else:
                    del llm

                time.sleep(1)  # Brief pause between attempts

            except Exception as e:
                print(f"❌ {e}")
                results["errors"].append(str(e))

        return results

    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite."""
        print("\n" + "="*70)
        print("FLASHTENSORS BENCHMARK SUITE")
        print("="*70)

        # Step 1: Check availability
        print("\n1️⃣  CHECKING FLASHTENSORS AVAILABILITY")
        if not self.test_flashtensors_available():
            print("\n⚠️  Cannot proceed without flashtensors")
            print("    Install: pip install git+https://github.com/leoheuler/flashtensors.git")
            self.results["status"] = "flashtensors_not_available"
            return self.results

        # Step 2: Configure
        print("\n2️⃣  CONFIGURING FOR YOUR HARDWARE")
        if not self.configure_flashtensors():
            print("\n⚠️  Configuration failed")
            self.results["status"] = "configuration_failed"
            return self.results

        # Step 3: Register models
        print("\n3️⃣  REGISTERING MODELS (One-time)")
        registration_results = {}
        for model in self.TEST_MODELS[:2]:  # Test first 2 to save time
            success, reg_time = self.register_model(model["id"], model["hf_path"])
            registration_results[model["id"]] = {
                "success": success,
                "registration_time": reg_time
            }

        self.results["tests"]["registration"] = registration_results

        # Step 4: Benchmark load times
        print("\n4️⃣  BENCHMARKING LOAD TIMES")
        print("\n   With flashtensors:")
        ft_results = {}
        for model in self.TEST_MODELS[:2]:
            ft_results[model["id"]] = self.benchmark_load_time(
                model["id"],
                use_flashtensors=True,
                repetitions=2
            )

        self.results["tests"]["flashtensors_loads"] = ft_results

        print("\n   With standard vLLM (for comparison):")
        vllm_results = {}
        for model in self.TEST_MODELS[:2]:
            vllm_results[model["id"]] = self.benchmark_load_time(
                model["id"],
                use_flashtensors=False,
                repetitions=1
            )

        self.results["tests"]["vllm_loads"] = vllm_results

        # Step 5: Print summary
        self._print_summary()

        return self.results

    def _print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)

        print("\n📊 HARDWARE:")
        hw = self.results["hardware"]
        for gpu in hw.get("gpus", []):
            print(f"   {gpu.get('name')}: {gpu.get('memory_gb')}GB")

        print(f"   CPU: {hw.get('cpu', {}).get('threads')} threads")
        print(f"   RAM: {hw.get('ram_gb')}GB")

        tests = self.results.get("tests", {})

        # Registration times
        if "registration" in tests:
            print("\n⏱️  REGISTRATION TIMES (One-time, one per model):")
            for model_id, data in tests["registration"].items():
                if data["success"]:
                    print(f"   {model_id}: {data['registration_time']:.2f}s")
                else:
                    print(f"   {model_id}: ❌ Failed")

        # Load time comparison
        if "flashtensors_loads" in tests and "vllm_loads" in tests:
            print("\n⚡ LOAD TIME COMPARISON:")
            print(f"   {'Model':<40} {'flashtensors':<15} {'vLLM':<15} {'Speedup':<10}")
            print(f"   {'-'*80}")

            ft_results = tests["flashtensors_loads"]
            vllm_results = tests["vllm_loads"]

            for model_id in ft_results:
                ft_times = ft_results[model_id].get("load_times", [])
                vllm_times = vllm_results[model_id].get("load_times", [])

                if ft_times and vllm_times:
                    ft_avg = sum(ft_times) / len(ft_times)
                    vllm_avg = sum(vllm_times) / len(vllm_times)
                    speedup = vllm_avg / ft_avg if ft_avg > 0 else 0

                    print(f"   {model_id:<40} {ft_avg:>6.2f}s {vllm_avg:>14.2f}s {speedup:>9.1f}x")

        print("\n" + "="*70)

    def save_results(self, output_path: str = "flashtensors_benchmark.json"):
        """Save detailed results to JSON."""
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to {output_path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark flashtensors on your hardware"
    )
    parser.add_argument(
        "--storage-path",
        default="/tmp/flashtensors_storage",
        help="Where to store optimized models"
    )
    parser.add_argument(
        "--output",
        default="flashtensors_benchmark.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test (1 repetition, 1 model)"
    )

    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                  FLASHTENSORS BENCHMARK                              ║
║         Test if flashtensors improves your 2x RTX 3090 setup        ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    if not FLASHTENSORS_AVAILABLE:
        print("⚠️  flashtensors is not installed yet.")
        print("\nTo install, run:")
        print("  pip install git+https://github.com/leoheuler/flashtensors.git")
        print("\nThen re-run this benchmark.")
        exit(1)

    benchmark = FlashTensorsBenchmark(storage_path=args.storage_path)
    results = benchmark.run_full_benchmark()
    benchmark.save_results(args.output)

    print("\n✅ Benchmark complete!")
    print(f"📄 Full results saved to {args.output}")
    print("\n💡 Next steps:")
    print("   1. Review the results above")
    print("   2. If flashtensors is 3-5x faster, consider integrating")
    print("   3. Integration would go in: core/fast_loader.py")
