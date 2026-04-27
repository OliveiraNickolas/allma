# 🚀 Quick Start: Your Personalized Allma Setup

This document gets you running in 5 minutes. For deep understanding, read `YOUR_SETUP_GUIDE.md`.

## What Was Created For You

### 📁 Profiles (5 Optimized Configurations)
```
configs/profiles/
├── coding.allm      → Qwen 35B TP=2  (best reasoning, 2-3s latency)
├── ocr.allm         → Vision 7B      (document OCR, 1-2s per page)
├── video.allm       → Vision 7B      (video analysis, 1-1.5s per frame)
├── comics.allm      → Vision 7B      (extract from graphics, 0.8-1.5s)
└── research.allm    → Qwen 27B       (tool calling + research, 1-2s)
```

### 🤖 Task Classifier
`core/classifier.py` - Automatically picks the right profile for your task.

### 📊 Integration Examples
```
integration/
└── comfyui_example.py  → Control Allma from ComfyUI workflows
```

### 🔧 Tools
```
scripts/
└── benchmark.py     → Measure YOUR setup's performance

docs/
├── YOUR_SETUP_GUIDE.md          → Detailed guide (READ THIS)
└── CONFIGURATION_ENCYCLOPEDIA.md → Parameter reference
```

---

## Installation (5 Steps)

### Step 1: Download Models
```bash
# Create model directory
mkdir -p /path/to/models

# Download the 4 models you need
# (Use your preferred model source: HuggingFace, ollama, etc)
# Suggested locations:
#   /path/to/models/Qwen3.5-35B-FP8
#   /path/to/models/Qwen3.5-27B-FP8
#   /path/to/models/Qwen3.5-Vision-7B-FP8
#   /path/to/models/Gemma-4-9B-FP8
```

### Step 2: Update Profile Paths
Edit each profile file and replace `/path/to/models/`:
```bash
cd ~/allma/configs/profile/

# Edit all .allm files to point to your actual model paths
sed -i 's|/path/to/models/|/your/actual/path/|g' *.allm
```

### Step 3: Start Server
```bash
cd ~/allma

# Start Allma server
allma serve
```

### Step 4: Test Each Profile
```bash
# In another terminal

# Test coding profile
allma run --profile coding "Write a quicksort algorithm"

# Test OCR profile (with an image)
allma run --profile ocr --file document.png

# Test research profile
allma run --profile research "What is retrieval augmented generation?"
```

### Step 5: Calibrate Your Setup
```bash
# Run benchmarks to measure YOUR hardware performance
python3 scripts/benchmark.py --profile coding

# Full benchmark (takes 10-15 min)
python3 scripts/benchmark.py --profile all --output YOUR_BENCHMARKS.json

# Results saved to YOUR_BENCHMARKS.json
```

---

## Usage Examples

### Auto-Detect Profile (Recommended)
```bash
# Allma automatically picks the right profile
allma run "optimize this code"
# → Detects "optimize" + "code" → uses 'coding' profile

allma run "extract text from this document"
# → Detects "extract" + "document" → uses 'ocr' profile

allma run "analyze this video"
# → Detects "analyze" + "video" → uses 'video' profile
```

### Explicit Profile
```bash
# Force a specific profile
allma run --profile coding "your query here"
allma run --profile ocr --file document.pdf
allma run --profile video --file movie.mp4
```

### With ComfyUI
```bash
# Use Allma to analyze ComfyUI-generated images
python3 integration/comfyui_example.py analyze /path/to/image.png

# Auto-refine generated images
python3 integration/comfyui_example.py refine workflow.json /output/dir
```

---

## Expected Performance (On Your Hardware)

Based on 2x RTX 3090 + Ryzen 9950X3D:

| Profile | Model | Latency | Throughput | VRAM | Quality |
|---------|-------|---------|-----------|------|---------|
| **coding** | Qwen 35B TP=2 | 2-3s | 40 tokens/s | 26GB | ⭐⭐⭐⭐⭐ |
| **ocr** | Vision 7B | 1-2s | - | 11GB | ⭐⭐⭐⭐⭐ |
| **video** | Vision 7B | 1-1.5s/frame | - | 11GB | ⭐⭐⭐⭐ |
| **comics** | Vision 7B | 0.8-1.5s | - | 11GB | ⭐⭐⭐⭐⭐ |
| **research** | Qwen 27B | 1-2s | 60 tokens/s | 21GB | ⭐⭐⭐⭐ |

---

## Decision Tree: Which Profile to Use?

```
What's your task?

├─ "Code something" or "Debug this"?
│  └─ → coding (best reasoning, 2-3s latency)
│
├─ "Extract text" or "Read this document"?
│  └─ → ocr (fast, 1-2s per page)
│
├─ "Analyze this video" or "Extract frames"?
│  └─ → video (specialized for motion, 1-1.5s/frame)
│
├─ "Read quadrinhos" or "Extract from graphic"?
│  └─ → comics (handles speech bubbles, 0.8-1.5s)
│
└─ "Research" or "Use tools/websearch"?
   └─ → research (balanced, good for tool calling, 1-2s)

Unsure? → Just ask, Allma will auto-detect!
```

---

## Troubleshooting

### "CUDA out of memory"
```bash
# Your GPU is using more memory than available
# Solution 1: Reduce model size (use smaller profile)
allma run --profile research (instead of coding)

# Solution 2: Reduce context length
# Edit the .allm file, change max_model_len

# Solution 3: Check GPU usage
nvidia-smi  # See what's using VRAM
```

### "Model loads but is slow"
```bash
# Check 1: Is Tensor Parallel configured correctly?
nvidia-smi  # Should show both GPUs active for coding profile

# Check 2: Is wrong model loading?
allma ps  # See which model is active

# Check 3: CPU bottleneck?
top -p $(pgrep vllm)  # Check CPU usage
```

### "Auto-detect didn't pick the right profile"
```bash
# Use explicit profile:
allma run --profile coding "your query"

# File extension detection:
allma run --file code.py  # Auto-detects coding
allma run --file doc.pdf  # Auto-detects ocr
allma run --file video.mp4  # Auto-detects video
```

---

## Next: Read the Full Guide

This was the quick start. For detailed info:

1. **YOUR_SETUP_GUIDE.md** — Everything about YOUR specific setup
2. **CONFIGURATION_ENCYCLOPEDIA.md** — Parameter reference
3. **comfyui_example.py** — ComfyUI integration code

---

## Support

- 📚 Read `docs/YOUR_SETUP_GUIDE.md` for detailed guidance
- 🧪 Run `scripts/benchmark.py` to calibrate your setup
- 📊 Check `YOUR_BENCHMARKS.json` for your actual latencies
- 🐛 Monitor with `nvidia-smi dmon` while running tasks

---

**You're ready to go! Start with:**
```bash
allma serve
# Then in another terminal:
allma run "optimize this algorithm"
```

Enjoy! 🚀
