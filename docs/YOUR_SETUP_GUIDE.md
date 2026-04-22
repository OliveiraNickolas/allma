# 🎬 Your Personalized Allma Setup Guide
## 2x RTX 3090 + Ryzen 9950X3D

This guide is tailored specifically to **your hardware and workloads**. Use this as your reference for optimal configuration, benchmarking, and decision-making.

---

## Quick Start: Your 5 Profiles

You have 5 pre-configured profiles optimized for your exact setup:

| Profile | Use Case | Model | VRAM | Latency | When |
|---------|----------|-------|------|---------|------|
| **coding** | Algorithm design, complex reasoning | Qwen 35B TP=2 | 26GB | 2-3s | When you need the best reasoning |
| **ocr** | Extract text from apostilas, scans | Vision 7B | 11GB | 1-2s/page | When processing documents |
| **video** | Analyze videos, extract frames | Vision 7B | 11GB | 1-1.5s/frame | When analyzing video content |
| **comics** | Extract text from quadrinhos | Vision 7B | 11GB | 0.8-1.5s/panel | When reading comics/graphics |
| **research** | Websearch, tool calling, MCP | Qwen 27B | 21GB | 1-2s | When you need tools + research |

### Using Profiles

```bash
# Auto-detect based on your query
allma run "optimize this algorithm"
# → Automatically uses 'coding' profile

# Or explicitly specify
allma run --profile ocr --file my_apostila.pdf
allma run --profile video --file movie.mp4

# Fallback to manual if unsure
allma run --profile research "search for latest AI news"
```

---

## Hardware Configuration

### Your Setup
```
GPU0: RTX 3090 (24GB)  │  GPU1: RTX 3090 (24GB)
CPU: Ryzen 9950X3D (16 cores, 3D V-Cache, excellent PCIe bandwidth)
RAM: 64GB DDR5
Storage: [Add your storage info]
```

### GPU Allocation Strategy

**Your two RTX 3090s work like this:**

#### Scenario 1: CODING (Best Reasoning)
```
GPU0: Qwen 35B         GPU1: Qwen 35B
     ↓                      ↓
Both GPUs share model via Tensor Parallel (TP=2)
Total VRAM used: 26-28GB
Latency: 2-3s per query
Quality: ⭐⭐⭐⭐⭐ (Excellent)
```
**When:** You're designing algorithms, debugging complex code, or need best reasoning.

#### Scenario 2: MULTIMODAL + RESEARCH (Flexible)
```
GPU0: Vision 7B (11GB)    GPU1: Qwen 27B (21GB)
OCR, Video, Comics        General reasoning, tools
Free: 13GB                Free: 3GB
```
**When:** You're doing general work — OCR, videos, research mixed together.

#### Scenario 3: OCR ONLY (Efficiency)
```
GPU0: Vision 7B (11GB)    GPU1: Empty
Document processing       Available for other tasks
Free: 13GB                Free: 24GB
```
**When:** Batch processing many documents. Can run another task on GPU1 if needed.

---

## Workload-Specific Decision Trees

### "I'm going to code for 2+ hours"

```
Step 1: Check if you need multimodal
   ├─ NO → Go to Step 2
   └─ YES → Use 'research' profile instead (Vision 7B understands code + images)

Step 2: Load Qwen 35B with TP=2
   allma serve --profile coding

Step 3: Keep Vision 7B unloaded
   → Don't use OCR during coding (GPU1 is busy)

Expected Performance:
   ├─ Startup time: ~5s (TP=2 initialization)
   ├─ Per-query latency: 2-3s (higher due to sync overhead)
   ├─ Quality: Excellent for complex algorithms
   └─ Token throughput: ~40 tokens/sec

Tips:
   ├─ Use long context (32K) for entire code files
   ├─ Set temperature=0.3 for deterministic output
   └─ Ctrl+C and reload if VRAM pressure detected
```

---

### "I need to OCR 100 pages of apostilas → Word format"

```
Step 1: Check image quality
   ├─ High quality (clear text) → Use 256px (faster)
   └─ Low quality (blurry) → Use 512px (slower, better)

Step 2: Load Vision 7B alone
   allma serve --profile ocr

Step 3: Batch your PDFs
   for file in *.pdf; do
     allma run --profile ocr --file "$file" > "$file.txt"
   done

Expected Performance:
   ├─ Per-page latency: 1-2s
   ├─ Accuracy: 95%+ on clear documents
   ├─ Preprocessing time: 0.2-0.5s per image
   └─ Total time: 100 pages ≈ 2-3 minutes

Tips:
   ├─ Preprocessing is 30% of latency (important!)
   ├─ 256px = fast + good quality (sweet spot)
   ├─ Increase temperature to 0.3 if text is ambiguous
   └─ Save to Word format with formatting preserved
```

---

### "Analyze a 1-hour video → create alternative versions"

```
Step 1: Prepare video
   ├─ Codec: MP4, AVI, MOV (vLLM supports these)
   ├─ Resolution: Any (will be downsampled to 360px)
   └─ FPS: Any (will be sampled at 1 FPS)

Step 2: Load Vision 7B with video config
   allma serve --profile video

Step 3: Process with chunking
   allma run --profile video \
     --file my_video.mp4 \
     --chunk-duration 300 \
     --output analysis.json

Chunking Strategy (from Reddit findings):
   ├─ 1h video → extracted as 1 FPS → 3600 frames
   ├─ Chunk into 5-min segments → 12 chunks
   ├─ 5s overlap between chunks (maintain context)
   ├─ Process each chunk: 5min × 60 frames/min = 300 frames
   └─ Total: ~300-400 API calls

Expected Performance:
   ├─ Per-frame latency: 1-1.5s
   ├─ Chunk latency: ~300 frames × 1.2s = ~6 minutes per chunk
   ├─ Total time: 12 chunks ≈ 1-1.5 hours
   └─ Quality: Maintains narrative context across chunks

Tips:
   ├─ Don't reduce FPS below 1 (loses important moments)
   ├─ Increase chunk overlap if scenes are complex
   ├─ Use temperature=0.7 for balanced summaries
   └─ Post-process summaries to create alternative formats
```

---

### "Extract text from quadrinhos/comics"

```
Step 1: Prepare images
   ├─ Resolution: Keep high (512px+ for small text)
   ├─ Format: PNG, JPG, WebP
   └─ Quality: Clear, good contrast

Step 2: Load Vision 7B with comics config
   allma serve --profile comics

Step 3: Process panels
   for image in *.jpg; do
     allma run --profile comics --file "$image" \
       --extract-speech-bubbles \
       --output "$image.txt"
   done

Expected Performance:
   ├─ Per-panel latency: 0.8-1.5s
   ├─ Speech accuracy: Excellent (trained on diverse text)
   ├─ Panel detection: Works well with clear panel boundaries
   └─ Batch 4 panels: ~3-4s total

Tips:
   ├─ Higher resolution (512px) helps with small text
   ├─ Preprocessing with "enhance_contrast=true" improves accuracy
   ├─ Use temperature=0.5 for balanced extraction
   ├─ Post-process output to clean up formatting
   └─ Consider manual review for complex layouts
```

---

### "Research + Tool Calling (WebSearch, MCP)"

```
Step 1: Setup tool integration
   ├─ WebSearch: Configure search engine credentials
   ├─ MCP: Start MCP servers (if using)
   └─ Tools: Enable tool calling in config

Step 2: Load Qwen 27B (balanced model)
   allma serve --profile research

Step 3: Ask questions with tools
   allma run --profile research \
     --enable-tools \
     "Find the latest trends in AI and summarize"

Expected Performance:
   ├─ Per-query latency: 1-2s
   ├─ Tool calls: 1-5 per query (configurable)
   ├─ Total latency: 2-5s (including tool execution)
   └─ Quality: Good reasoning + tool integration

Tips:
   ├─ Leave GPU0 free (Vision 7B can run if needed)
   ├─ Set temperature=0.7 for exploration
   ├─ Configure tool timeout (default: 30s)
   ├─ Use max_tool_calls=5 to prevent loops
   └─ Check MCP logs for tool execution issues
```

---

## Benchmark Your Setup

### Calibration Test 1: Measure Coding Latency

```bash
# Start coding server
allma serve --profile coding

# In another terminal, measure latency
time curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "coding",
    "messages": [{"role": "user", "content": "Write a quicksort algorithm"}],
    "max_tokens": 500
  }'

Expected output:
   ├─ Time to first token (TTFT): 2-3s
   ├─ Token generation speed: ~40 tokens/sec
   ├─ Total time (100 tokens): 5-6s
   └─ VRAM usage: 26-28GB
```

### Calibration Test 2: Measure OCR Performance

```bash
# Time a single page OCR
time allma run --profile ocr --file page.png

Expected output:
   ├─ Preprocessing: 0.2-0.5s
   ├─ Model inference: 0.8-1.5s
   ├─ Total latency: 1-2s per page
   └─ VRAM usage: 10-12GB
```

### Calibration Test 3: GPU Memory Profile

```bash
# Monitor VRAM during operation
watch -n 0.5 'nvidia-smi dmon -s pcum'

# In another terminal, run a task
allma run --profile coding "Complex algorithm"

Expected output:
   ├─ GPU0 + GPU1 TP=2: 26-28GB
   ├─ Vision 7B: 10-12GB
   ├─ Qwen 27B: 20-22GB
   └─ Peak memory: Never > 48GB (your total)
```

---

## When NOT to Use Your Setup

### ❌ Don't try to parallelize coding + OCR
```
WRONG:
  allma run --profile coding "optimize this"
  allma run --profile ocr --file doc.pdf  [waits forever]
         ↑
    Both GPUs busy with coding model

RIGHT:
  # Finish coding first
  # Then: allma run --profile ocr --file doc.pdf
```

### ❌ Don't use default TP=1 for 35B on single GPU
```
WRONG:
  tensor_parallel = 1  [on single 24GB GPU]
  → 35B needs 27GB minimum!

RIGHT:
  tensor_parallel = 2  [uses both GPUs]
  → 26GB total, balanced load
```

### ❌ Don't run multiple large models concurrently
```
WRONG:
  GPU0: Qwen 35B (16GB)
  GPU1: Qwen 27B (20GB)
  → Total: 36GB > 24GB per GPU! OOM!

RIGHT:
  Use profiles that fit:
  - Coding only (35B TP=2)
  - Or Vision 7B + Qwen 27B (11+20=31GB, won't fit!)
  - Or Vision 7B + Gemma4 (11+14=25GB ✓)
```

---

## Troubleshooting

### Problem: "CUDA out of memory"
```
Check which GPU:
  nvidia-smi dmon -s pcum

Solution:
  1. Reduce max_num_seqs (currently 2 for coding)
  2. Reduce gpu_memory_utilization (currently 0.85)
  3. Reduce batch size (max_num_batched_tokens)
  4. Use smaller model (Vision 7B instead of 27B)

Quick fix:
  allma stop
  # Edit config, reduce one parameter
  allma serve --profile coding
```

### Problem: "Model loads but is very slow"
```
Check:
  1. Is TP=1 on 35B model? → Should be TP=2
  2. Is max_model_len too large? → Reduce to 16K
  3. Are other processes using GPU? → nvidia-smi shows them

Diagnosis:
  nvidia-smi dmon  # Check GPU utilization
  # Should see 90%+ utilization when processing
```

### Problem: "Can't load both Vision 7B and Qwen 27B"
```
Math check:
  Vision 7B: ~11GB
  Qwen 27B: ~20GB
  Total: 31GB > 24GB per GPU ❌

Solutions:
  1. Use Gemma4 instead of Qwen 27B (14GB instead of 20GB)
  2. Unload Vision 7B, then load Qwen 27B
  3. Use smaller quant (int4 instead of FP8)
```

---

## Optimization Tips for Your Setup

### 1. Use Preprocessing Aggressively
```
# From Reddit findings:
1 FPS + 360px height reduces inference by 50%
256px for images reduces inference by 40-60%

Your setup:
  - Vision 7B on 360px: 1-1.5s per frame
  - Preprocessing: 0.2-0.5s overhead
  - Total: 1.2-2s (preprocessing is 15-40% of cost!)
```

### 2. Leverage Qwen 35B's Long Context
```
# Your Ryzen 9950X3D has excellent PCIe bandwidth
max_model_len = 32768 is practical
token throughput: ~40 tokens/sec

This means:
  - Load entire codebase (~5000 tokens)
  - Analyze in single context
  - Better reasoning than multiple chunks
```

### 3. Batch Vision Tasks
```
# Vision 7B can handle multiple images in a single batch
max_num_seqs = 4 (for comics profile)

This means:
  - Queue 4 panels
  - Process in ~1-2s total (not 4x slower)
  - Huge throughput boost for batch OCR
```

### 4. Monitor Your Ryzen's Temperature
```
# With 2x RTX 3090s, CPU gets hot
# Your 3D V-Cache design helps with this

Monitor:
  watch -n 1 'sensors | grep Tdie'

If > 90°C:
  - Reduce n_threads to 12 instead of 16
  - Add cooling / improve airflow
  - Consider underclocking
```

---

## ComfyUI Integration

Your setup is perfect for **generative content pipelines**:

```python
# Example: Generate image → Analyze with Vision → Refine

1. ComfyUI generates image (on GPU0)
2. Allma Vision 7B analyzes on GPU1 (11GB free)
3. Send feedback to ComfyUI
4. Iterate for best version

# API Endpoint
curl http://localhost:9000/v1/chat/completions \
  -d '{
    "model": "research",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Analyze this image"},
        {"type": "image_url", "image_url": "http://localhost/generated.png"}
      ]
    }]
  }'
```

---

## Final Checklist Before Use

- [ ] Download all 4 models (35B, 27B, Vision 7B, Gemma4)
- [ ] Edit profile files with your actual model paths
- [ ] Run `allma serve --profile coding` (warmup)
- [ ] Test `allma run --profile ocr --file test.png`
- [ ] Calibrate benchmarks (measure your actual latencies)
- [ ] Monitor VRAM (nvidia-smi dmon) during first run
- [ ] Setup ComfyUI webhook (if using)
- [ ] Document your benchmarks in `YOUR_BENCHMARKS.md`

---

## Your Next Steps

1. ✅ Read this guide (you're here!)
2. ⬜ Download models to `/path/to/models/`
3. ⬜ Edit profile files with correct paths
4. ⬜ Test each profile: `allma run --profile <name> "test query"`
5. ⬜ Calibrate benchmarks
6. ⬜ Integrate with your workflow (ComfyUI, scripts, etc)

---

**Happy inferencing! 🚀**

This setup is optimized specifically for YOU. Monitor your benchmarks, adjust if needed, and enjoy fast, flexible inference!
