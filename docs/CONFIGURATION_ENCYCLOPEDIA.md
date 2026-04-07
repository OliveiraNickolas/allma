# 📚 Allama Configuration Encyclopedia
## Non-Technical Guide to Model Parameters

Welcome! This encyclopedia explains what each configuration parameter does in simple, everyday language. Think of these parameters as "knobs and dials" you can turn to adjust how your AI model behaves and performs.

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [vLLM Parameters](#vllm-parameters-how-your-ai-thinks)
3. [llama.cpp Parameters](#llamacpp-parameters-local-cpu-gpu-inference)
4. [Model-Specific Recommendations](#model-specific-recommendations)
5. [Decision Trees](#decision-trees-which-parameters-to-change)

---

## Quick Start

**New to configuration?** Ask yourself:
- ❓ "What problem am I having?" (answers at bottom: Decision Trees)
- ⚙️ "Which backend am I using?" (vLLM or llama.cpp?)
- 📊 "What's my hardware?" (how much GPU/CPU do I have?)
- ⏱️ "What matters most?" (speed, quality, or VRAM usage?)

---

## vLLM Parameters: How Your AI Thinks

These parameters control how vLLM (a high-performance inference engine) loads and runs your model.

### **Sampling Parameters** (What the Model Outputs)

#### `temperature`
**Simple definition:** How "creative" or "random" the model's answers are.

**Analogy:** Imagine asking a DJ to play music:
- Temperature = 0.1: The DJ always plays the same hit song (boring, predictable)
- Temperature = 0.7: The DJ picks from the top 10 songs (balanced variety)
- Temperature = 1.5: The DJ plays random songs from all genres (chaotic)

**Range:** 0.0 to 2.0+ (typically 0.1 to 1.5)
**Default:** 1.0
**Effect on Output:** Lower = factual, consistent. Higher = creative, diverse.
**Effect on Speed:** No change (purely logical)
**Effect on Memory:** No change

**When to change:**
- ✅ Lower (0.3-0.5) for: facts, math, coding, customer service
- ✅ Higher (0.8-1.2) for: creative writing, brainstorming, storytelling

⚠️ **What breaks:** Setting to 0 makes the model always pick the exact same response (boring). Setting too high (>2.0) makes it incoherent gibberish.

---

#### `top_p` (Nucleus Sampling)
**Simple definition:** The model only considers "popular enough" words/phrases.

**Analogy:** When texting, you usually only choose from the next 3-5 word suggestions, not all 1000 possible words. `top_p` is that filtering.

**Range:** 0.0 to 1.0
**Default:** 1.0 (consider all possibilities)
**Effect on Output:** Lower = more focused, higher = more diverse.
**Effect on Speed:** No change
**Effect on Memory:** No change

**When to change:**
- ✅ Lower (0.7-0.9) for: precise answers, technical content
- ✅ Keep at 1.0 for: general chat, flexibility

⚠️ **What breaks:** Setting too low (<0.3) makes responses feel stilted or incomplete.

---

#### `top_k`
**Simple definition:** Only consider the "top K most likely" next words.

**Analogy:** Google search autocomplete shows you the top 10 suggestions, not all 1 million possibilities. That's `top_k=10`.

**Range:** 1 to infinity (typically 5 to 50)
**Default:** -1 (disabled)
**Effect on Output:** Lower = more focused, higher = more diverse.
**Effect on Speed:** Slightly faster with lower values
**Effect on Memory:** No change

**When to change:**
- Usually leave this disabled (default: -1)
- Use only with `top_p` for advanced fine-tuning

---

#### `min_p`
**Simple definition:** Minimum "confidence score" a word needs to be considered.

**Analogy:** "Only suggest words that are at least 5% as likely as the most likely word."

**Range:** 0.0 to 1.0
**Default:** 0.0 (disabled)
**Effect on Output:** Removes unlikely suggestions
**Effect on Speed:** Slightly faster
**Effect on Memory:** No change

**When to change:**
- Leave at 0 for most use cases
- Use sparingly for very specific domains

---

#### `repetition_penalty`
**Simple definition:** Discourages the model from repeating the same phrases over and over.

**Analogy:** If you keep saying "blue," the model makes it slightly less likely to say "blue" again (gently pushing variety).

**Range:** 0.8 to 2.0 (1.0 = no penalty)
**Default:** 1.0
**Effect on Output:** Prevents repetitive loops in text generation
**Effect on Speed:** No change
**Effect on Memory:** No change

**When to change:**
- ✅ Increase to 1.1-1.2 if: the model keeps repeating the same phrase
- Leave at 1.0 for: most general use

⚠️ **What breaks:** Values > 1.5 can make the model avoid common words it should use.

---

#### `presence_penalty`
**Simple definition:** Penalizes the model for using words it has used before in the conversation.

**Analogy:** "You already mentioned 'neural network' once; it's less likely now (but not impossible)."

**Range:** -2.0 to 2.0 (0 = no change)
**Default:** 0.0
**Effect on Output:** Encourages vocabulary diversity
**Effect on Speed:** No change
**Effect on Memory:** No change

**When to change:**
- ✅ Positive values (0.5-1.0) for: long conversations where you want varied vocabulary
- Leave at 0 for: short responses

---

### **Performance Parameters** (How Fast It Runs)

#### `tensor_parallel` (TP Size)
**Simple definition:** Split the model across multiple GPUs to run it faster.

**Analogy:** If cooking for 100 people, one chef is slow. Three chefs working together is faster!

**Range:** 1, 2, 4, 8, etc. (must divide into your GPU count)
**Default:** 1 (single GPU)
**Effect on Output:** No change (same answers, just faster)
**Effect on Speed:** Faster (proportional to number of GPUs, ~70% efficient)
**Effect on Memory:** Spreads VRAM across GPUs (better utilization)

**When to change:**
- ✅ Set to 2 if: you have 2+ GPUs and want 50% faster inference
- ✅ Set to 4 if: you have 4+ GPUs and want 2-3x faster inference
- ❌ Don't set higher than your GPU count

⚠️ **What breaks:** Setting to 3 when you have 4 GPUs wastes resources. Must match your hardware.

---

#### `gpu_memory_utilization`
**Simple definition:** What percentage of GPU memory vLLM can use.

**Analogy:** If your GPU has 24GB, a 0.90 setting means vLLM reserves 21.6GB for the model (and keeps 2.4GB free for overhead).

**Range:** 0.5 to 1.0 (50% to 100%)
**Default:** 0.90
**Effect on Output:** No change
**Effect on Speed:** Higher % = more concurrent requests, BUT higher risk of out-of-memory
**Effect on Memory:** Direct (higher % uses more VRAM)

**When to change:**
- ✅ Increase to 0.95 if: you have extra VRAM and want more throughput
- ✅ Decrease to 0.75 if: you get "out of memory" errors
- ❌ Never set to 1.0 (too risky)

⚠️ **What breaks:** Setting too high causes CUDA out-of-memory crashes. Setting too low wastes your expensive GPU.

---

#### `max_num_seqs`
**Simple definition:** How many conversations can the model handle at the same time.

**Analogy:** A restaurant can seat 10 tables (conversations) simultaneously. More tables = busier kitchen.

**Range:** 1 to 1024+ (default: 8-16)
**Default:** 8
**Effect on Output:** No change
**Effect on Speed:** Higher = more concurrent users, each gets slower response
**Effect on Memory:** Higher uses more VRAM (for "waiting room" cache)

**When to change:**
- ✅ Increase to 32-64 if: you want to handle many users simultaneously
- ✅ Decrease to 1-2 if: you have very limited VRAM
- Common values: 8 (default), 16 (2x users), 32 (4x users)

⚠️ **What breaks:** Setting too high causes out-of-memory. The model slows down if you max it out.

---

#### `max_model_len` (Context Window)
**Simple definition:** The maximum length of conversation the model can "remember."

**Analogy:** If `max_model_len=4096`, the model can remember about 4000 words back. Beyond that, it forgets earlier parts of the conversation.

**Range:** 512 to 131,072 (depends on model)
**Default:** Model-dependent (often 2048-4096)
**Effect on Output:** Larger = model can reference older messages in long conversations
**Effect on Speed:** Larger context = slower (quadratic relationship)
**Effect on Memory:** Larger context = exponentially more VRAM (KV cache grows)

**When to change:**
- ✅ Increase to 8192 if: you have 50GB+ VRAM and need long conversations
- ✅ Decrease to 2048 if: you have limited VRAM
- Common for Qwen3.5: 65536 (but uses massive VRAM)

⚠️ **What breaks:** Setting higher than your VRAM allows causes out-of-memory. Most users set this correctly; only advanced users tweak it.

---

## llama.cpp Parameters: Local CPU/GPU Inference

These control how llama.cpp (optimized for local machines) loads and runs your model.

### **Context & Batch Parameters**

#### `n_ctx` (Context Size)
**Simple definition:** Same as vLLM's `max_model_len` but for llama.cpp.

**Analogy:** Your short-term memory: how far back in a conversation can you remember?

**Range:** 128 to 131,072 (typically 512-40960)
**Default:** 512 (quite limited)
**Effect on Output:** Larger = longer memory
**Effect on Speed:** Larger context = slower
**Effect on Memory:** Exponentially more VRAM/RAM

**When to change:**
- ✅ Set to 8192 for: reasonable conversations without OOM
- ✅ Set to 16384 for: very long documents (if you have 24GB+ VRAM)
- ❌ Don't set below 512

⚠️ **What breaks:** Too high causes out-of-memory on small systems.

---

#### `n_batch` (Batch Size)
**Simple definition:** How many tokens llama.cpp processes in one go before waiting.

**Analogy:** Washing dishes: batch 1 = wash one dish, rinse, repeat. Batch 32 = collect 32 dishes, wash them all, rinse. Batch 32 is more efficient!

**Range:** 8 to 2048 (typically 256-1024)
**Default:** 512
**Effect on Output:** No change
**Effect on Speed:** Higher = faster (more efficient GPU use), BUT higher memory usage
**Effect on Memory:** Higher uses slightly more VRAM

**When to change:**
- ✅ Increase to 1024 if: you have 12GB+ VRAM (faster)
- ✅ Decrease to 256 if: you have limited VRAM
- Sweet spot: usually 256-512

⚠️ **What breaks:** Setting way too high (>2048) causes out-of-memory. Setting too low (<32) is slow.

---

#### `n_gpu_layers`
**Simple definition:** How many "layers" of the model to run on GPU (rest run on CPU).

**Analogy:** A 32-layer model. If `n_gpu_layers=24`, run 24 layers on GPU, 8 on CPU (CPU is much slower!).

**Range:** -1 (auto), 0 (CPU only), 1-33+ (number of layers)
**Default:** -1 (automatic)
**Effect on Output:** No change
**Effect on Speed:** More GPU layers = faster (GPU is ~100x faster than CPU)
**Effect on Memory:** More GPU layers = more VRAM needed

**When to change:**
- ✅ Set to -1 (auto) for: llama.cpp to figure it out (recommended)
- ✅ Set to 99 for: force all layers to GPU (if you have enough VRAM)
- ✅ Set to 10-20 for: balance speed/VRAM on limited cards

⚠️ **What breaks:** Setting too high causes out-of-memory. Setting to 0 makes it extremely slow (CPU-only).

---

#### `n_threads` (CPU Threads)
**Simple definition:** How many CPU cores llama.cpp uses for the CPU-side work.

**Analogy:** A 16-core CPU can do 16 tasks in parallel. More threads = more parallelism.

**Range:** 1 to your CPU core count
**Default:** Auto-detect (usually reasonable)
**Effect on Output:** No change
**Effect on Speed:** Usually little impact (GPU bottleneck dominates)
**Effect on Memory:** No change (just CPU utilization)

**When to change:**
- ✅ Set to CPU core count if: auto-detect fails (rare)
- ✅ Lower to avoid CPU heat if: your CPU is thermal-throttling
- Rarely needs adjustment

⚠️ **What breaks:** Nothing really; set it and forget it.

---

### **GPU Offloading** (Advanced)

#### `n_gpu_layers` with MoE (Mixture of Experts)
For **Nemotron** and other MoE models:

**Simple definition:** Offload selective layers to CPU, run expert layers on GPU.

**Analogy:** "Run the smart parts on GPU, let CPU handle the routing."

**Configuration:**
```ini
# Nemotron 3 Super (MoE variant)
n_gpu_layers = 40  # Main layers on GPU
# Automatically: expert layers distributed CPU/GPU
```

**Effect:** Faster than CPU-only, less VRAM than fully-GPU.

---

## Model-Specific Recommendations

### **Qwen3.5 (27B Dense)**

**What is it:** A general-purpose model from Alibaba with "thinking mode" for reasoning.

**Recommended Config for vLLM:**
```ini
[backend = vllm]
max_model_len = 65536      # Qwen supports huge context
max_num_seqs = 8           # Moderate concurrency (uses lots of VRAM)
gpu_memory_utilization = 0.90
tensor_parallel = 2        # If you have 2+ GPUs
temperature = 0.7          # Balanced creativity
top_p = 0.8                # Nucleus sampling for consistency
```

**Recommended Config for llama.cpp:**
```ini
[backend = llama.cpp]
n_ctx = 32768              # Use half the capability (still huge)
n_batch = 512              # Standard batch
n_gpu_layers = 40          # Most layers on GPU (if you have 24GB VRAM)
n_threads = 12             # Set to your CPU core count
```

**VRAM Requirements:**
- vLLM: 20-24GB (FP8 quantized)
- llama.cpp (GGUF 8-bit): 15-18GB

**Special Note:** Qwen3.5 has "thinking mode" enabled by default. Set `enable_thinking: false` in logical model config to disable it for faster inference.

---

### **Gemma4 (Google)**

**What is it:** Google's newest open-weight model, optimized for quality and speed.

**Recommended Config for vLLM:**
```ini
[backend = vllm]
max_model_len = 8192       # More modest context
max_num_seqs = 16          # Good throughput
gpu_memory_utilization = 0.85
tensor_parallel = 1        # Smaller, fits on single GPU easily
temperature = 0.6          # Slightly more factual than Qwen
top_p = 0.9
```

**Recommended Config for llama.cpp:**
```ini
[backend = llama.cpp]
n_ctx = 8192
n_batch = 1024             # Can handle larger batches
n_gpu_layers = 99          # Load everything to GPU (efficient)
n_threads = 8
```

**VRAM Requirements:**
- vLLM: 8-12GB (BF16)
- llama.cpp (GGUF 5-bit): 6-8GB

**Special Note:** Gemma4 is very memory-efficient. Great for consumer hardware (RTX 4090, RTX 4080).

---

### **LFM2 (On-Device Efficiency)**

**What is it:** Designed for on-device deployment; hybrid architecture with MoE.

**Recommended Config for vLLM:**
```ini
[backend = vllm]
max_model_len = 4096       # Modest context (on-device constraint)
max_num_seqs = 32          # Can handle many users efficiently
gpu_memory_utilization = 0.75
tensor_parallel = 1
temperature = 0.5          # Factual-leaning
```

**Recommended Config for llama.cpp:**
```ini
[backend = llama.cpp]
n_ctx = 4096
n_batch = 256              # Lower batch for low-power devices
n_gpu_layers = 30          # Offload selectively (MoE optimization)
n_threads = 4              # Can run on modest CPUs
```

**VRAM Requirements:**
- vLLM: 4-6GB
- llama.cpp: 3-4GB

**Special Note:** LFM2 excels on laptops and edge devices. Use it for mobile-friendly inference.

---

### **Nemotron (NVIDIA MoE)**

**What is it:** NVIDIA's optimized MoE (Mixture of Experts) model; huge but sparse.

**Recommended Config for vLLM:**
```ini
[backend = vllm]
max_model_len = 4096
max_num_seqs = 4           # MoE uses unpredictable VRAM; be conservative
gpu_memory_utilization = 0.80
tensor_parallel = 4        # Needs multi-GPU (120B is huge)
temperature = 0.8
```

**Recommended Config for llama.cpp:**
```ini
[backend = llama.cpp]
n_ctx = 4096
n_batch = 128              # Conservative for MoE stability
n_gpu_layers = 40          # Selectively offload
# Advanced: offload MoE layers to CPU
```

**VRAM Requirements:**
- vLLM: 40-80GB (depends on expert activation)
- llama.cpp: 30-50GB

**Special Note:** Nemotron is expert-level. Only for those with enterprise GPUs (H100, A100).

---

## Decision Trees: Which Parameters to Change?

### "My model is slow"

```
📊 Is throughput the issue (responses to many users)?
├─ YES:
│  ├─ Increase max_num_seqs (8 → 16 → 32)
│  ├─ Increase gpu_memory_utilization (0.90 → 0.95)
│  └─ Add tensor_parallel if you have 2+ GPUs
│
└─ NO (Single user slow):
   ├─ Decrease max_model_len (less context to process)
   ├─ Increase tensor_parallel
   └─ Check if you're running on CPU (n_gpu_layers should be high)
```

---

### "I'm running out of VRAM"

```
💾 VRAM OOM Error?
├─ Decrease max_num_seqs (8 → 4 → 1)
├─ Decrease max_model_len (4096 → 2048)
├─ Decrease gpu_memory_utilization (0.90 → 0.75)
├─ Decrease n_ctx (for llama.cpp)
├─ Decrease n_batch (for llama.cpp)
└─ Last resort: Switch to smaller model or lower precision (GGUF 4-bit)
```

---

### "Responses are too repetitive"

```
🔁 Model repeating itself?
├─ Increase repetition_penalty (1.0 → 1.1 → 1.2)
├─ Increase presence_penalty (0 → 0.5 → 1.0)
├─ Increase temperature (0.5 → 0.8)
└─ Increase top_p (0.7 → 0.9)
```

---

### "Responses are incoherent or off-topic"

```
🤔 Model answers are nonsensical?
├─ Decrease temperature (1.0 → 0.7 → 0.3)
├─ Decrease top_p (1.0 → 0.8 → 0.5)
├─ Decrease max_model_len (maybe context is too long)
└─ Check if you picked the right model (Qwen3.5 for thinking, Gemma4 for speed)
```

---

## FAQ

**Q: What's the difference between temperature and top_p?**
A: Temperature controls "randomness"; top_p controls "diversity." Use both for fine control. Start with temperature, add top_p only if needed.

**Q: Should I set context length to the maximum?**
A: No! Larger context = exponentially slower. Use only what you need. 4096 is good for most chats; 8192 for documents.

**Q: GPU layers = fast, CPU = slow. Why use CPU at all?**
A: Sometimes your model doesn't fit on GPU; llama.cpp lets you run it (slowly) on CPU. Use GPU layers whenever you can.

**Q: Tensor parallelism: how much speedup?**
A: Rough estimate: TP=2 → ~1.5x faster. TP=4 → ~2.5x faster. (Not perfectly linear.)

**Q: Can I change parameters while the server is running?**
A: No. Edit the config file and restart Allama: `allama stop && allama serve`

---

## Quick Reference Table

| Parameter | vLLM | llama.cpp | Range | Impact | Risk |
|-----------|------|-----------|-------|--------|------|
| temperature | ✅ | ✅ | 0.1-2.0 | Output quality | Low |
| top_p | ✅ | ✅ | 0.0-1.0 | Output diversity | Low |
| max_num_seqs | ✅ | ❌ | 1-1024 | Throughput | Medium (OOM) |
| gpu_memory_util | ✅ | ❌ | 0.5-1.0 | VRAM usage | High (OOM) |
| tensor_parallel | ✅ | ❌ | 1,2,4,8... | Speed | Low (must match GPUs) |
| max_model_len | ✅ | ❌ | 512-131K | Memory | High (OOM) |
| n_ctx | ❌ | ✅ | 128-131K | Memory | High (OOM) |
| n_batch | ❌ | ✅ | 8-2048 | Speed | Medium |
| n_gpu_layers | ❌ | ✅ | -1 to N | Speed | Medium (OOM) |

---

## Resources

- [vLLM Official Docs](https://docs.vllm.ai/) — Comprehensive technical reference
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp/) — Latest updates and examples
- [Qwen3.5 Model Card](https://huggingface.co/Qwen/Qwen3.5-7B) — Official specs
- [Gemma4 Announcement](https://blogs.nvidia.com/blog/rtx-ai-garage-open-models-google-gemma-4/) — Performance data

---

**Last Updated:** April 2026
**Version:** 1.0
**Audience:** All skill levels (beginner-friendly)
