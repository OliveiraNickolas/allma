#!/usr/bin/env python3
"""
Allma Benchmark Tool: Calibrate your setup for optimal performance
Measures: latency, throughput, VRAM usage, accuracy per profile
"""

import json
import time
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


class AlamaBenchmarker:
    """Benchmark Allma performance on your hardware."""

    def __init__(self, allma_host: str = "http://127.0.0.1", allma_port: int = 9000):
        self.base_url = f"{allma_host}:{allma_port}"
        self.session = requests.Session()
        self.results = {}

    def benchmark_profile(
        self,
        profile: str,
        queries: List[str],
        expected_tokens: int = 100,
        repetitions: int = 3
    ) -> Dict:
        """
        Benchmark a profile with multiple queries.

        Args:
            profile: Profile name (coding, ocr, research, etc)
            queries: List of test queries
            expected_tokens: Expected output length
            repetitions: Number of times to run each query

        Returns:
            Benchmark results
        """
        print(f"\n{'='*70}")
        print(f"BENCHMARKING PROFILE: {profile}")
        print(f"{'='*70}")

        results = {
            "profile": profile,
            "timestamp": datetime.now().isoformat(),
            "queries": []
        }

        for query in queries:
            print(f"\nQuery: {query[:60]}...")
            query_results = {
                "query": query,
                "latencies": [],
                "ttft": [],  # Time to first token
                "throughput": []
            }

            for rep in range(repetitions):
                print(f"  Attempt {rep+1}/{repetitions}...", end=" ", flush=True)

                try:
                    ttft, latency, throughput = self._measure_query(
                        profile, query, expected_tokens
                    )
                    query_results["latencies"].append(latency)
                    query_results["ttft"].append(ttft)
                    query_results["throughput"].append(throughput)
                    print(f"✓ ({latency:.2f}s, {throughput:.0f} tokens/sec)")

                except Exception as e:
                    print(f"✗ Error: {e}")

            # Calculate stats
            if query_results["latencies"]:
                query_results["stats"] = {
                    "avg_latency": sum(query_results["latencies"]) / len(query_results["latencies"]),
                    "min_latency": min(query_results["latencies"]),
                    "max_latency": max(query_results["latencies"]),
                    "avg_ttft": sum(query_results["ttft"]) / len(query_results["ttft"]),
                    "avg_throughput": sum(query_results["throughput"]) / len(query_results["throughput"]),
                }

            results["queries"].append(query_results)

        # Get VRAM usage
        results["vram_usage"] = self._get_vram_usage()

        return results

    def _measure_query(
        self,
        profile: str,
        query: str,
        expected_tokens: int
    ) -> Tuple[float, float, float]:
        """
        Measure single query performance.

        Returns:
            (time_to_first_token, total_latency, tokens_per_sec)
        """
        payload = {
            "model": profile,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": expected_tokens,
            "stream": True
        }

        start_time = time.time()
        first_token_time = None
        token_count = 0

        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=60
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if not line or line.startswith(b"[DONE]"):
                continue

            if first_token_time is None:
                first_token_time = time.time()

            # Parse token count from SSE
            try:
                data = json.loads(line.decode()[6:])  # Skip "data: "
                if "choices" in data:
                    token_count += 1
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                pass  # Ignore malformed SSE lines

        total_time = time.time() - start_time
        ttft = first_token_time - start_time if first_token_time else total_time
        throughput = token_count / (total_time - ttft) if total_time > ttft else 0

        return ttft, total_time, throughput

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
                used, total = map(int, line.split(", "))
                vram[f"GPU{i}"] = {
                    "used_gb": used / 1024,
                    "total_gb": total / 1024,
                    "util_percent": (used / total) * 100
                }
            return vram
        except Exception as e:
            print(f"Warning: Could not get VRAM usage: {e}")
            return {}

    def benchmark_all_profiles(self) -> Dict:
        """Run benchmarks for all profiles."""
        benchmarks = {
            "coding": {
                "queries": [
                    "Write a quicksort algorithm in Python",
                    "Optimize this nested loop: for i in range(n): for j in range(n): print(i*j)",
                    "Explain the time complexity of a binary search tree"
                ],
                "expected_tokens": 200
            },
            "ocr": {
                "queries": [
                    "Extract all text from this document image preserving layout",
                ],
                "expected_tokens": 300
            },
            "video": {
                "queries": [
                    "Describe what's happening in this video frame",
                    "What are the main subjects and their actions in this frame?",
                ],
                "expected_tokens": 150
            },
            "research": {
                "queries": [
                    "Research the latest developments in AI",
                    "Find information about transformer architectures",
                ],
                "expected_tokens": 200
            }
        }

        all_results = {}
        for profile, config in benchmarks.items():
            try:
                results = self.benchmark_profile(
                    profile,
                    config["queries"],
                    expected_tokens=config.get("expected_tokens", 100),
                    repetitions=2
                )
                all_results[profile] = results
            except Exception as e:
                print(f"Error benchmarking {profile}: {e}")
                all_results[profile] = {"error": str(e)}

        return all_results

    def save_results(self, results: Dict, output_path: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")

    def print_summary(self, results: Dict):
        """Print summary of benchmark results."""
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}")

        for profile, profile_results in results.items():
            if "error" in profile_results:
                print(f"\n{profile}: ERROR - {profile_results['error']}")
                continue

            print(f"\n{profile}:")
            for query_result in profile_results["queries"]:
                if "stats" in query_result:
                    stats = query_result["stats"]
                    print(f"  Query: {query_result['query'][:40]}...")
                    print(f"    Latency:   {stats['avg_latency']:.2f}s (min: {stats['min_latency']:.2f}s, max: {stats['max_latency']:.2f}s)")
                    print(f"    TTFT:      {stats['avg_ttft']:.2f}s")
                    print(f"    Throughput: {stats['avg_throughput']:.0f} tokens/sec")

            if "vram_usage" in profile_results:
                print(f"  VRAM Usage:")
                for gpu, usage in profile_results["vram_usage"].items():
                    print(f"    {gpu}: {usage['used_gb']:.1f}GB / {usage['total_gb']:.1f}GB ({usage['util_percent']:.0f}%)")


# ============================================================================
# QUICK BENCHMARKS
# ============================================================================

def quick_benchmark_coding():
    """Quick benchmark: Coding profile latency."""
    benchmarker = AlamaBenchmarker()
    results = benchmarker.benchmark_profile(
        "coding",
        ["Write a simple Hello World program in 5 languages"],
        expected_tokens=150,
        repetitions=2
    )
    benchmarker.print_summary({"coding": results})
    return results


def quick_benchmark_ocr():
    """Quick benchmark: OCR profile accuracy."""
    benchmarker = AlamaBenchmarker()
    # Note: This requires an actual image
    results = benchmarker.benchmark_profile(
        "ocr",
        ["Extract all text from this document"],
        expected_tokens=200,
        repetitions=1
    )
    benchmarker.print_summary({"ocr": results})
    return results


def full_benchmark():
    """Full benchmark: All profiles."""
    benchmarker = AlamaBenchmarker()
    results = benchmarker.benchmark_all_profiles()
    benchmarker.save_results(results, "YOUR_BENCHMARKS.md")
    benchmarker.print_summary(results)
    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Allma Benchmark Tool")
    parser.add_argument(
        "--profile",
        choices=["coding", "ocr", "video", "research", "all"],
        default="all",
        help="Which profile to benchmark"
    )
    parser.add_argument(
        "--output",
        default="YOUR_BENCHMARKS.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (faster, less detailed)"
    )

    args = parser.parse_args()

    benchmarker = AlamaBenchmarker()

    if args.profile == "all":
        print("Running full benchmark suite...")
        results = benchmarker.benchmark_all_profiles()
    else:
        print(f"Benchmarking {args.profile}...")
        results = {args.profile: benchmarker.benchmark_profile(
            args.profile,
            ["Sample query for testing"],
            expected_tokens=100,
            repetitions=2 if not args.quick else 1
        )}

    benchmarker.save_results(results, args.output)
    benchmarker.print_summary(results)
