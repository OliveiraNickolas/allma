# Allma

A personal LLM manager that sits in front of **vLLM** and **llama.cpp**, giving them a unified OpenAI + Anthropic-compatible API with automatic model loading, VRAM management, and multi-GPU support.

```
Claude Code / OpenAI client / curl
              │
              ▼
         Allma :9000          ← unified API (OpenAI + Anthropic)
         /v1/chat/completions
         /v1/messages
              │
     ┌────────┴────────┐
     ▼                 ▼
  vLLM :8000     llama-server :9001
  (safetensors)  (GGUF / multimodal)
```

**What it does:**
- Loads models on-demand when a request comes in, unloads them when idle
- Picks the GPU with the most free VRAM automatically
- Translates Anthropic Messages API ↔ OpenAI format so llama.cpp models work with Claude Code and other Anthropic-native tools
- Lets you define multiple "profiles" (different sampling settings) over the same model without loading it twice

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Linux | Ubuntu 20.04+ or equivalent |
| Python | 3.10+ |
| NVIDIA GPU | Any with CUDA support |
| NVIDIA driver | 520+ recommended |
| CUDA toolkit | 11.8+ (for vLLM) |

> **macOS / Windows:** Not supported. vLLM requires Linux + CUDA.

---

## Installation

```bash
git clone https://github.com/yourusername/allma
cd allma
bash install.sh
```

The installer:
1. Creates a Python virtualenv at `allma/venv/`
2. Installs Python dependencies (`fastapi`, `uvicorn`, `rich`, `textual`, …)
3. Creates an `allma` command at `~/.local/bin/allma`
4. Copies `.env.example` → `.env`
5. Reports whether vLLM and llama-server are found

Make sure `~/.local/bin` is on your PATH. If the installer says it isn't, add this to `~/.bashrc` or `~/.zshrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

---

## Backend Setup

Allma is a lightweight proxy. The heavy backends — vLLM and llama.cpp — are separate. **You only need to install the backend(s) you plan to use.**

### vLLM (safetensors / FP8 / BF16 models)

```bash
venv/bin/pip install vllm
```

That's it. vLLM requires a CUDA GPU. Installation downloads several GB and takes 5–15 minutes.

> If you already have vLLM installed in a different virtualenv or conda environment, allma will find it automatically via `which vllm`. No need to reinstall.

### llama.cpp (GGUF models)

**Option A — automated build script (recommended, full features)**

```bash
bash scripts/install-llama-cpp.sh
```

This clones llama.cpp, builds `llama-server` with CUDA support, and installs it to `~/.local/bin/`. Prerequisites: `cmake` and `build-essential` (standard on most Linux distros).

```bash
# If cmake is missing:
sudo apt install -y cmake build-essential
```

**Option B — pip install (no compilation, limited features)**

```bash
# Replace cu121 with your CUDA major version (cu118, cu121, cu124, cu128…)
# Find yours: nvidia-smi | grep "CUDA Version"
venv/bin/pip install "llama-cpp-python[server]" \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

This works without any compilation. Limitations vs. the native binary:
- No vision / multimodal (`mmproj`) support
- No KV cache quantization (`--cache-type-k`)
- No flash attention
- No jinja chat templates

Use Option B for quick testing or if you only need basic GGUF inference.

**Auto-detection order**

Allma finds `llama-server` from: `LLAMA_CPP_PATH` env var → `~/.local/bin/llama-server` → `~/llama.cpp/build/bin/llama-server` → `PATH` → `llama-cpp-python` Python module (fallback).

To override explicitly:
```ini
# .env
LLAMA_CPP_PATH=/custom/path/to/llama-server
```

---

## Configuration

### How configs work

Allma uses two layers of configuration files in `configs/`:

```
configs/
├── base/        ← one file per model installation (path, backend, hardware settings)
└── profile/     ← one file per "personality" (which base, sampling overrides)
```

**Base config** — describes a model on disk:

```ini
# configs/base/MyModel.allm

backend = "vllm"
path = "/path/to/Models/Qwen3.5-27B-FP8"
tokenizer = "/path/to/Models/Qwen3.5-27B-FP8"
gpu_memory_utilization = "0.92"
max_model_len = 131072
max_num_seqs = 4

extra_args = [
    "--reasoning-parser", "qwen3",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_coder"
]
```

```ini
# configs/base/MyGGUF.allm  (llama.cpp)

backend = "llama.cpp"
model = "/path/to/Models/model-Q4_K_M.gguf"
# mmproj = "/path/to/Models/mmproj-f16.gguf"  # optional: enables vision

n_ctx = 65536
n_gpu_layers = "-1"
n_threads = "8"
n_batch = "1024"

extra_args = [
    "--jinja",
    "--flash-attn", "on",
    "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0",
    "--cont-batching"
]
```

**Profile config** — describes how to talk to a base model:

```ini
# configs/profile/MyModel-Instruct.allm

name = "MyModel:Instruct"
base = "MyModel"          # matches the base config filename (without .allm)
enable_thinking = false   # disable chain-of-thought

[sampling]
temperature = 0.7
top_p = 0.9
top_k = 40
min_p = 0.0
presence_penalty = 0.0
repetition_penalty = 1.0
```

### Creating your first config

Example files are provided for every supported model family. Copy one and edit the paths:

```bash
# vLLM model
cp configs/base/Qwen3.6-27B-FP8.allm.example configs/base/Qwen3.6-27B-FP8.allm
# Edit the file and set the correct path

# GGUF model
cp configs/base/Qwen3.6-35B-A3B-Uncensored-GGUF_Q4.allm.example configs/base/MyGGUF.allm
# Edit the file and set the correct path
```

Or use the interactive wizard:

```bash
allma wizard
```

The wizard detects your model family, calculates VRAM requirements, and generates both the base config and matching profiles.

### Environment variables

Copy `.env.example` to `.env` and adjust:

```bash
cp .env.example .env
```

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLMA_PORT` | `9000` | Port for the Allma API |
| `VLLM_BASE_PORT` | `8000` | First port for vLLM backends |
| `LLAMA_BASE_PORT` | `9001` | First port for llama.cpp backends |
| `LLAMA_CPP_PATH` | auto | Path to `llama-server` binary |
| `KEEP_ALIVE_SECONDS` | `600` | Seconds before an idle model is unloaded |
| `GPU_MEMORY_THRESHOLD_GB` | `1.0` | Min free VRAM to load a model |
| `AUTO_SWAP_ENABLED` | `true` | Unload idle models when VRAM is needed |
| `ALLMA_VISIBLE_DEVICES` | all GPUs | Restrict to specific GPUs, e.g. `"0,1"` |
| `MAX_MESSAGES` | `0` | Truncate conversation history (0 = off) |

---

## CLI

```bash
# Server lifecycle
allma serve              # start daemon in background
allma serve -v           # start in foreground with live logs
allma restart            # stop + start
allma stop               # stop server and all backends

# Status
allma status             # is the server running?
allma list               # show available profiles
allma ps                 # show currently loaded models + GPU usage

# Models
allma run <profile>      # load model and open interactive chat
allma unload <model>     # unload a model immediately (free VRAM)

# Logs
allma logs               # show recent allma logs
allma logs -f            # follow allma logs live
allma backend logs       # tail the running backend log

# Integrations
allma launch claude <profile>   # load model, launch Claude Code pointed at it

# Setup tools
allma wizard             # interactive TUI wizard to create configs
allma hardware-detect    # show detected GPUs and VRAM
allma download <hf-repo> # download a HuggingFace model and create configs
```

---

## Downloading Models

Allma has a built-in HuggingFace downloader that handles model selection, download progress, and automatic config generation — all in one command.

```bash
allma download <hf-repo-or-url>
```

**Examples:**

```bash
# Paste a HuggingFace URL directly
allma download https://huggingface.co/bartowski/Qwen3-8B-GGUF

# Or use the repo ID shorthand
allma download Qwen/Qwen3-8B

# Works with any model type: GGUF, safetensors, FP8, BF16
allma download Qwen/Qwen3-8B-FP8
```

### What happens

**For GGUF repos** — an interactive file picker lists every `.gguf` variant sorted by quantization quality (BF16 → Q8 → Q6 → Q4 → …), with file sizes shown. Pick one or several:

```
╔══════════════════════════════════════════════════════════════╗
║ ┌──────────────────────────────────────────────────────────┐ ║
║ │ [ Download ]                                             │ ║
║ │   repo      ▸  bartowski/Qwen3-8B-GGUF                  │ ║
║ │   dest      ▸  ~/AI/Models/Qwen3-8B-GGUF                │ ║
║ └──────────────────────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║ ┌──────────────────────────────────────────────────────────┐ ║
║ │ [ Files ]                                                │ ║
║ │  #   File                                     Size       │ ║
║ │  1   Qwen3-8B-BF16.gguf                       15.7 GB    │ ║
║ │  2   Qwen3-8B-Q8_0.gguf                        8.6 GB    │ ║
║ │  3   Qwen3-8B-Q6_K_L.gguf                      6.6 GB    │ ║
║ │  4   Qwen3-8B-Q4_K_M.gguf                      5.0 GB    │ ║
║ │  5   Qwen3-8B-Q4_K_S.gguf                      4.7 GB    │ ║
║ │  6   Qwen3-8B-IQ4_XS.gguf                      4.5 GB    │ ║
║ │                                                          │ ║
║ │  Enter numbers to download  1  or  1 3  or  1,3         │ ║
║ │  · Enter to cancel                                       │ ║
║ └──────────────────────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════╝

  ▸ 4
```

**For safetensors repos** — shows total size and asks for confirmation before downloading the full repo (config files, tokenizer, weights). Skips `.bin`, `.pt`, and other formats automatically.

**After download** — Allma reads the model's `config.json` (or GGUF metadata) to detect the model family and auto-generates ready-to-use configs:

```
╔══════════════════════════════════════════════════════════════╗
║ ┌──────────────────────────────────────────────────────────┐ ║
║ │ [ Ready ]                                                │ ║
║ │   base      ▸  configs/base/Qwen3-8B-GGUF.allm          │ ║
║ │   profile   ▸  configs/profile/Qwen3-8B-GGUF-Instruct   │ ║
║ │   profile   ▸  configs/profile/Qwen3-8B-GGUF-Reasoning  │ ║
║ │                                                          │ ║
║ │   allma run Qwen3-8B-GGUF-Instruct                       │ ║
║ └──────────────────────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════╝
```

Run the suggested command and the model is ready to use.

### Requirements

`huggingface_hub` is included in Allma's dependencies and installed automatically. For gated models (Llama, Gemma, etc.), authenticate first:

```bash
venv/bin/pip install huggingface_hub   # already in requirements.txt
huggingface-cli login                  # paste your HF token
```

Or set the env var:

```bash
export HUGGING_FACE_HUB_TOKEN=hf_...
```

### Models directory

By default models are downloaded to `~/AI/Models/<repo-name>`. Override with:

```ini
# .env
ALLMA_MODELS_DIR=/data/models
```

---

## API

Allma exposes two compatible APIs on the same port.

### OpenAI-compatible

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:9000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="MyModel:Instruct",   # profile name from configs/profile/
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Anthropic-compatible

```python
import anthropic

client = anthropic.Anthropic(
    base_url="http://127.0.0.1:9000",
    api_key="dummy",
)
message = client.messages.create(
    model="MyModel:Instruct",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(message.content[0].text)
```

### Admin endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Server health + loaded model count |
| `GET /v1/models` | List available profiles |
| `GET /v1/ps` | Active backend processes |
| `POST /v1/load` | Pre-load a model: `{"model": "Profile:Name"}` |
| `POST /v1/unload` | Unload a model: `{"model": "base-name"}` |
| `POST /v1/shutdown` | Graceful shutdown |

---

## Integration with Claude Code

The most common use case: run Claude Code against a local model.

**Option 1 — `allma launch` (recommended)**

```bash
allma launch claude MyModel:Instruct
```

This loads the model, configures Claude Code to use it, and opens a Claude Code session. When you close the session, the model stays loaded until `KEEP_ALIVE_SECONDS` elapses.

**Option 2 — Manual configuration**

Point Claude Code to Allma permanently in `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:9000",
    "ANTHROPIC_AUTH_TOKEN": "dummy"
  }
}
```

Then set the default model:

```bash
export ANTHROPIC_DEFAULT_SONNET_MODEL="MyModel:Instruct"
export ANTHROPIC_DEFAULT_OPUS_MODEL="MyModel:Instruct"
claude
```

> **Context window:** Claude Code sessions can exceed 100 K tokens. Make sure your model's `n_ctx` (llama.cpp) or `max_model_len` (vLLM) is set accordingly. For long sessions with GGUF models, consider a separate config without `mmproj` to save VRAM for the KV cache.

---

## Multi-GPU setup

### Tensor parallelism (vLLM)

For models too large for a single GPU, set `tensor_parallel` in the base config:

```ini
backend = "vllm"
tensor_parallel = "2"   # split across 2 GPUs
```

Allma automatically selects consecutive GPUs (e.g. 0+1, 1+2) with enough free VRAM. If no consecutive pair fits, it reports a clear error.

### Pinning to a specific GPU

To pin a model to a specific GPU (e.g. GPU 1 while GPU 0 runs ComfyUI):

```ini
gpu_id = 1
```

This works for both vLLM and llama.cpp backends.

### Hiding GPUs from Allma

```ini
# .env
ALLMA_VISIBLE_DEVICES=1,2   # only use GPU 1 and 2
```

---

## VRAM budget

Allma estimates each model's VRAM requirement before loading and unloads other models if needed. The estimate includes:

- Model weights (from file sizes)
- KV cache (from model architecture — respects quantization, sliding window, hybrid architectures)
- Fixed overhead (~0.25 GB for llama.cpp, ~1 GB for vLLM)

For llama.cpp the KV cache size scales linearly with `n_ctx`. To reduce it, lower `n_ctx` or use KV quantization:

```ini
extra_args = ["--cache-type-k", "q4_0", "--cache-type-v", "q4_0"]
```

---

## How configs auto-detect arguments

If a base config has no `extra_args`, Allma reads the model's `config.json` (or GGUF metadata) to identify the model family (Qwen3, Gemma4, Llama, Phi, etc.) and applies appropriate defaults automatically. This means a minimal config like:

```ini
backend = "vllm"
path = "/path/to/Models/Qwen3-8B"
```

…will still get the right `--reasoning-parser`, `--tool-call-parser`, and other settings applied automatically.

---

## Profiles: thinking mode

Models that support chain-of-thought (e.g. Qwen3, DeepSeek-R1) can have thinking enabled or disabled per profile:

```ini
# configs/profile/MyModel-Reasoning.allm
name = "MyModel:Reasoning"
base = "MyModel"
enable_thinking = true   # default for non-instruct profiles

# configs/profile/MyModel-Instruct.allm
name = "MyModel:Instruct"
base = "MyModel"
enable_thinking = false  # fast, no <think> blocks
```

Profiles with "instruct" in the name disable thinking automatically.

---

## Troubleshooting

**Model won't load — VRAM error**
```
RuntimeError: Not enough VRAM: MyModel needs 22.1GB, only 18.3GB free
```
→ Another model is loaded. Run `allma ps` to see what's active, then `allma unload <model>`.

**Port already in use**
```
[Errno 98] Address already in use
```
→ A previous backend process is still running. Run `allma stop` which will kill orphaned processes.

**llama-server not found**
→ Build llama.cpp (see [Backend Setup](#backend-setup)) or set `LLAMA_CPP_PATH` in `.env`.

**vLLM not found**
→ `venv/bin/pip install vllm` — or `pip install vllm` if using system Python.

**Context window exceeded**
```
request (113158 tokens) exceeds context size (98304)
```
→ The conversation is too long for the model's configured context. Use `/compact` in Claude Code to summarize the session, or increase `n_ctx`/`max_model_len` (check VRAM budget first).

**Claude Code loads the wrong model**
→ Use `allma launch claude <profile>` instead of running `claude` directly — it pins the model for that session.

---

## Project structure

```
allma/
├── allma.py            # server entry point (uvicorn + signal handlers)
├── allma_cli.py        # CLI (allma serve / stop / run / launch / …)
├── allma_tui.py        # full TUI (model library + wizard launcher)
├── wizard.py           # interactive setup wizard (standalone + embedded in TUI)
├── create_config.py    # config generator (used by wizard and download command)
├── install.sh          # one-shot installer
│
├── core/
│   ├── config.py       # constants, .env loading, model config loading
│   ├── state.py        # shared runtime state
│   ├── server.py       # FastAPI app and all route handlers
│   ├── loader.py       # model loading, VRAM checks, readiness polling
│   ├── process.py      # build backend commands, kill, shutdown
│   ├── gpu.py          # VRAM detection, TP selection, VRAM estimation
│   ├── health.py       # idle timeout + crash detection monitor
│   ├── model_detect.py # auto-detect model family from config.json / GGUF
│   ├── bootstrap.py    # hardware detection at startup
│   ├── downloader.py   # HuggingFace model downloader
│   └── error_detector.py  # parse backend stderr for actionable errors
│
├── configs/
│   ├── base/           # *.allm (gitignored) + *.allm.example (committed)
│   ├── profile/        # *.allm profile configs (committed, no paths)
│   └── loader.py       # .allm file parser
│
├── docs/               # extended documentation
├── scripts/            # benchmark and utility scripts
└── tests/              # unit and integration tests
```

---

## License

MIT
