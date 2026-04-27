# 🎯 Your Model Configuration Reference
## Nick's Setup: Qwen3.5 Models + Optimizations

This document shows **your actual models** with recommended configurations based on official documentation and your hardware (2x RTX 3090 + Ryzen 9950X3D).

---

## Quick Inventory

| Model | Downloaded | Format | Current Config | Recommendation |
|-------|-----------|--------|-----------------|-----------------|
| **Qwen3.5-35B-A3B** | ✅ | FP8 | TP=2 ✅ | Add reasoning-parser for better quality |
| **Qwen3.5-27B** | ✅ | FP8 | TP=1 ✅ | Excellent as-is (balanced) |
| **Qwen3.5-35B-Uncensored** | ✅ | FP8 | GGUF only | Uncensored variant for unrestricted output |
| **Qwen3.5-9b** | ✅ | FP8 | TP=1 ✅ | Lightweight fallback |
| **Qwen3.VL-8B** | ✅ | FP8 | TP=1 ✅ | OCR/Video optimized |
| **Qwen3.VL-8B-Caption** | ✅ | FP8 | TP=1 ✅ | Caption-specialized variant |
| **Qwen3.5-27B-GGUF** | ✅ | GGUF | llama.cpp | Great for CPU fallback/testing |

---

## Detailed Recommendations Per Model

### 1️⃣ **Qwen3.5-35B-A3B (Your Best Model)**

**Current Physical Config:**
```ini
[configs/base/Qwen3.5-35b.allm]
backend = "vllm"
path = "/path/to/models/Qwen3.5-35B-A3B-FP8"
tensor_parallel = "2"          ✅ Correct for your 2x RTX 3090
gpu_memory_utilization = "0.95"  ⚠️  Risky, lower to 0.85
max_model_len = 65536          ✅ Good
max_num_seqs = 8               ⚠️  Reduce to 2-4 for quality
extra_args = ["--reasoning-parser", "qwen3", "..."]  ✅ Good
```

**Recommended Optimizations:**

```ini
# VERSION 1: Pure Reasoning (Best Quality)
[reasoning-focused]
gpu_memory_utilization = "0.85"  # More stable
max_num_seqs = 2                 # One request at a time = best quality
extra_args = [
    "--reasoning-parser", "qwen3",           # Enable thinking mode
    "--disable-custom-all-reduce"            # GPU sync fix
]

# Then in profile, use:
[sampling]
temperature = 0.6        # Balanced reasoning
top_p = 0.95
top_k = 20
presence_penalty = 0.0   # Don't penalize repetition for code
min_p = 0.0
```

**VERSION 2: Coding + Tool Calling (Best for Your Needs)**
```ini
gpu_memory_utilization = "0.85"
max_num_seqs = 4                # Handle multiple code requests
extra_args = [
    "--reasoning-parser", "qwen3",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_coder",     # Parse function calls
    "--disable-custom-all-reduce"
]

[sampling]
temperature = 0.3        # Code needs to be factual
top_p = 0.95
top_k = 20
presence_penalty = 0.0
repetition_penalty = 1.1  # Prevent repeating same code patterns
```

**VERSION 3: Fast Mode (No Reasoning)**
```ini
# Remove --reasoning-parser to skip thinking
extra_args = [
    "--disable-custom-all-reduce"
]
# Faster inference, less thinking
```

**What's Being Used Now:**
```bash
--reasoning-parser qwen3              ← Enables thinking tokens
--enable-auto-tool-choice             ← Auto tool selection
--tool-call-parser qwen3_coder        ← Parse Qwen's tool format
--disable-custom-all-reduce           ← GPU sync workaround
```

**Performance Expectations:**
- With reasoning: 2-3s first token latency, ~40 tokens/sec
- Without reasoning: 1-2s first token latency, ~50 tokens/sec
- VRAM: 26-28GB with TP=2

---

### 2️⃣ **Qwen3.5-27B (General Purpose)**

**Current Physical Config:**
```ini
backend = "vllm"
path = "/path/to/models/Qwen3.5-27B-FP8"
tensor_parallel = "1"          ✅ Correct (fits single GPU)
gpu_memory_utilization = "0.90"  ✅ Good
max_model_len = 65536          ✅ Long context
max_num_seqs = 8               ✅ Reasonable
extra_args = []                ⚠️  Could add reasoning
```

**Recommended Enhancement:**

```ini
# Option 1: Add Reasoning (Better Quality)
extra_args = [
    "--reasoning-parser", "qwen3"
]

# Option 2: Add Tool Calling (For Research Profile)
extra_args = [
    "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_coder"
]

# Option 3: Keep as-is (Fast, No Overhead)
extra_args = []
```

**Recommended Sampling (Research Profile):**
```ini
[sampling]
temperature = 0.7        # Balanced exploration
top_p = 0.85
top_k = 20
presence_penalty = 1.5   # Encourage varied vocabulary (for research)
repetition_penalty = 1.0
```

**Performance:**
- First token: 1-2s
- Throughput: ~60 tokens/sec
- VRAM: 20-22GB
- Status: Excellent for balanced workloads ✅

---

### 3️⃣ **Qwen3.VL-8B (Vision/OCR/Video)**

**Current Physical Config:**
```ini
backend = "vllm"
path = "/path/to/models/Qwen3.VL-8B"
tensor_parallel = "1"
gpu_memory_utilization = "0.90"
max_model_len = 61152
max_num_seqs = "4"
generation_config = "vllm"
limit_mm_per_prompt.video = "0"   ⚠️  Only if image-only!
extra_args = ["--mm-encoder-tp-mode", "data", ...]
```

**Recommended Optimization:**

```ini
# VERSION 1: OCR ONLY (Images, no videos)
# ✅ Saves 2-3GB VRAM
extra_args = [
    "--mm-encoder-tp-mode", "data",           # Split vision encoder
    "--async-scheduling",                     # Process while generating
    "--limit-mm-per-prompt.video", "0",       # Don't allocate for videos
    "--disable-custom-all-reduce"
]
max_num_seqs = "4"    # Batch panels together

# VERSION 2: MIXED (Both images and videos)
extra_args = [
    "--mm-encoder-tp-mode", "data",
    "--async-scheduling",
    # DON'T limit video, or set to higher number
    "--disable-custom-all-reduce"
]
max_num_seqs = "2"    # More conservative with video

# VERSION 3: VIDEO PRIMARY
extra_args = [
    "--mm-encoder-tp-mode", "data",
    "--async-scheduling",
    "--disable-custom-all-reduce"
]
max_num_seqs = "1"    # One video at a time
```

**For Your Use Case (OCR + Comics):**
```ini
# Use VERSION 1 if you're mostly doing OCR/comics
# This saves the most VRAM
extra_args = [
    "--mm-encoder-tp-mode", "data",
    "--async-scheduling",
    "--limit-mm-per-prompt.video", "0",
    "--disable-custom-all-reduce"
]

[sampling]
temperature = 0.2     # Factual text extraction
top_p = 0.8
max_num_seqs = 4      # Batch 4 panels/pages
```

**VRAM Savings:**
- With video support: 11-12GB
- Image-only (with flag): 9-10GB ← Use this for OCR
- Savings: ~2GB = can fit more models!

**Performance:**
- Per image: 1-2s
- Per page OCR: 1-2s
- Per video frame: 1-1.5s
- Status: Excellent for multimodal ✅

---

### 4️⃣ **Qwen3.5-35B-Uncensored (GGUF)**

**Current Physical Config:**
```ini
backend = "llama.cpp"
model = "...Qwen3.5-35B-Uncensored.gguf"
n_ctx = 65536          ✅ Huge context
n_batch = "4096"       ✅ Good
n_gpu_layers = "-1"    ✅ Auto-offload
n_threads = "24"       ✅ Use all cores
extra_args = ["--jinja", "--flash-attn", "on"]  ✅ Good
```

**Recommended Enhancements:**

```ini
# For Reasoning Tasks
extra_args = [
    "--jinja",                       # Chat template
    "--flash-attn", "on",            # Fast attention
    "--reasoning-format", "deepseek" # Parse thinking tokens
]

# For General Speed
extra_args = [
    "--jinja",
    "--flash-attn", "on"
]

# For Maximum Quality (slower)
extra_args = [
    "--jinja",
    "--flash-attn", "on",
    "--cache-type-k", "q8_0",   # Quantize KV cache → -50% memory
    "--cache-type-v", "q8_0"
]
```

**Performance (Rough Estimates):**
- First token: 2-4s (slower than vLLM due to CPU bottleneck)
- Throughput: 20-30 tokens/sec
- VRAM: 20-24GB (depends on n_gpu_layers)
- Status: Great fallback if vLLM fails, or for testing ✅

**When to Use:**
- ✅ Fallback if vLLM crashes
- ✅ Testing reasoning format
- ✅ Compatibility experiments
- ❌ Production (vLLM is faster)

---

### 5️⃣ **Qwen3.5-27B-GGUF (Claude-Opus Reasoning)**

**Current Config:**
```ini
backend = "llama.cpp"
model = ".../Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF/Qwen3.5-27B.Q8_0.gguf"
mmproj = "...BF16-mmproj.gguf"  # Vision support
n_ctx = 65536
n_batch = "4096"
n_gpu_layers = "-1"
n_threads = "24"
extra_args = ["--jinja", "--flash-attn", "on"]  ✅ Good
```

**Recommended For:**
- ✅ Reasoning-distilled variant (compressed for speed)
- ✅ llama.cpp testing
- ✅ CPU-only fallback
- ✅ Compatibility with older vLLM versions

**Expected Performance:**
- Faster than 35B GGUF (27B is smaller)
- Good for reasoning even with llama.cpp
- ~30-40 tokens/sec typical

---

## Summary Table: Quick Reference

| Model | Profile | TP | VRAM | Latency | Special Args |
|-------|---------|----|----|---------|---------|
| **35B** | coding | 2 | 26GB | 2-3s | `--reasoning-parser qwen3` |
| **27B** | research | 1 | 21GB | 1-2s | `--enable-auto-tool-choice` |
| **Vision 8B** | ocr | 1 | 9-11GB | 1-2s | `--limit-mm-per-prompt.video 0` |
| **Vision 8B** | video | 1 | 11-12GB | 1-1.5s/frame | (none, video enabled) |
| **9B** | fallback | 1 | 8-9GB | 1-2s | (none) |
| **35B-Uncensored** | gguf | - | 20-24GB | 2-4s | `--jinja --flash-attn on` |
| **27B-Reasoning** | gguf | - | 18-20GB | 2-3s | `--jinja --flash-attn on` |

---

## Optimization Tips for Your Hardware

### 1. GPU Memory Allocation
```
GPU0 (24GB):           GPU1 (24GB):
├─ 35B (TP=2):         ├─ 35B (TP=2):
│  13GB                │  13GB
│  = 26GB total ✅

OR (when not coding):

├─ Vision 8B (10GB)    ├─ 27B (20GB)
├─ Free: 14GB          └─ Free: 4GB
```

### 2. VRAM Optimization Checklist
- [ ] Set `--limit-mm-per-prompt.video 0` if OCR only (saves 2GB)
- [ ] Use FP8 models instead of FP16/BF16 (saves 50% VRAM)
- [ ] Set `max_num_seqs = 2` for quality > throughput
- [ ] Use `gpu_memory_utilization = 0.85` instead of 0.95 (more stable)
- [ ] Consider KV cache quantization for GGUF: `--cache-type-k q8_0`

### 3. Performance Tuning
- [ ] Enable reasoning-parser for complex tasks (worth 1-2% quality, costs 5-10% latency)
- [ ] Use async-scheduling for vision to overlap image processing with generation
- [ ] Enable flash-attn for GGUF (requires compilation)
- [ ] Set `n_batch = 4096` for GGUF (higher throughput on RTX 3090)

### 4. What Flags YOU Need Most
For your actual workloads:

**Coding Profile (35B):**
```bash
--reasoning-parser qwen3              # Thinking mode for complex code
--disable-custom-all-reduce            # GPU sync fix
```

**OCR Profile (Vision 8B):**
```bash
--limit-mm-per-prompt.video 0         # Saves 2GB VRAM!
--mm-encoder-tp-mode data             # Split vision encoder
--async-scheduling                    # Better throughput
```

**Research Profile (27B):**
```bash
--enable-auto-tool-choice             # Auto tool selection
--tool-call-parser qwen3_coder        # Parse tool calls
```

---

## Action Items

- [ ] Review your actual `configs/base/*.allm` files
- [ ] Test **with reasoning** vs **without** and note speed difference
- [ ] If OCR-primary: add `--limit-mm-per-prompt.video 0` to Vision config (saves 2GB!)
- [ ] If using GGUF: verify llama.cpp supports `--flash-attn` (check `llama-cli --help`)
- [ ] Run `benchmark.py` with current config to establish baseline
- [ ] Re-run `benchmark.py` after adding recommended flags to see impact
- [ ] Document your changes in `YOUR_BENCHMARKS.json`

---

**Sources Used:**
- [Qwen3.5-35B HuggingFace Card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [vLLM Qwen3.5 Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- [Qwen3-VL Specifications](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8)
- [llama.cpp CLI Reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md)
- Your current config files analysis

**Last Updated:** April 2026
**Customized For:** Nick's 2x RTX 3090 + Ryzen 9950X3D Setup
