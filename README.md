# Allama

A personal LLM model manager with dynamic loading and support for both vLLM and llama.cpp backends.

> **Note:** Currently in early stages, designed for personal use.

## Overview

Allama manages multiple LLM models dynamically, automatically handling:

- **Automatic model loading** — Models load on-demand when first requested
- **Smart VRAM management** — Allocates models to best available GPU based on free memory
- **Idle model unloading** — Automatically unloads models after being idle to free up VRAM
- **Multi-backend support** — vLLM for high-performance serving, llama.cpp for flexibility
- **Unified API** — OpenAI-compatible (`/v1/chat/completions`) and Anthropic Messages API (`/v1/messages`) with full tool calling support
- **Logical model abstraction** — Multiple logical models can share the same physical model with different sampling parameters
- **CLI & Interactive REPL** — Manage the server and chat with models from the terminal
- **Watchdog daemon** — Auto-restarts on crashes

## Architecture

### Physical Models

Define actual model installations (model files + backend configuration).

Location: `configs/physical/*.allm`

```ini
# vLLM backend
backend = "vllm"
path = "/path/to/safetensors"
tokenizer = "/path/to/tokenizer"
tensor_parallel = "2"
gpu_memory_utilization = "0.90"
max_model_len = "131072"
max_num_seqs = "8"
extra_args = ["--reasoning-parser", "qwen3", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder"]
```

```ini
# llama.cpp backend
backend = "llama.cpp"
model = "/path/to/model.gguf"
mmproj = "/path/to/mmproj.gguf"
n_ctx = "196608"
n_batch = "1024"
n_gpu_layers = "-1"
n_threads = "8"
extra_args = ["--jinja", "--flash-attn", "on"]
```

### Logical Models

Define how to interact with a physical model — which physical model it uses plus optional sampling parameter overrides.

Location: `configs/logical/*.allm`

```ini
physical = "Qwen3.5-27b"

[sampling]
temperature = 0.7
top_p = 0.8
top_k = 20
min_p = 0.0
presence_penalty = 1.5
repetition_penalty = 1.0
```

An unlimited number of logical models can be created for each physical model. Models with "instruct" in the logical name automatically have thinking mode disabled.

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install fastapi httpx psutil uvicorn rich
   ```
3. vLLM and/or llama.cpp must be installed separately. The `llama-server` binary is found automatically via `LLAMA_CPP_PATH` env var, `PATH`, or common build locations.

## CLI

```bash
allama serve              # Start server as background daemon
allama serve -v           # Start in foreground with live logs
allama stop               # Stop server and all backends
allama status             # Show server status
allama list               # List available logical models
allama ps                 # Show loaded (running) models
allama logs -f            # Tail allama logs
allama backend logs       # Tail the running backend's logs
allama run <model>        # Load a model and open interactive chat
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLAMA_PORT` | `9000` | Port for the Allama API |
| `VLLM_BASE_PORT` | `8000` | Starting port range for vLLM backends |
| `LLAMA_BASE_PORT` | `9001` | Starting port range for llama.cpp backends |
| `LLAMA_CPP_PATH` | auto-detected | Path to `llama-server` binary |
| `KEEP_ALIVE_SECONDS` | `600` | Seconds to keep idle models loaded |
| `HEALTH_CHECK_INTERVAL` | `60` | Health check interval (seconds) |
| `GPU_MEMORY_THRESHOLD_GB` | `1.0` | Minimum free VRAM to load new models |
| `AUTO_SWAP_ENABLED` | `true` | Auto-unload idle models when VRAM is needed |
| `SWAP_IDLE_THRESHOLD` | `300` | Idle seconds before a model becomes swap-eligible |
| `MAX_MESSAGES` | `0` | Max messages per request (0 = unlimited) |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | OpenAI-compatible chat completions |
| `POST /v1/messages` | Anthropic Messages API (with tool calling translation for llama.cpp) |
| `GET /v1/models` | List available logical models |
| `POST /v1/load` | Pre-load a model without generating tokens |
| `GET /v1/ps` | Show active backend processes |
| `GET /health` | Health check |
| `POST /v1/shutdown` | Graceful shutdown |

### Anthropic Messages API (`/v1/messages`)

The `/v1/messages` endpoint provides full Anthropic Messages API compatibility, including:

- **Tool calling** — Tool definitions, `tool_use`, and `tool_result` blocks are translated between Anthropic and OpenAI formats for llama.cpp backends
- **Streaming** — SSE events translated to Anthropic format (`message_start`, `content_block_start/delta/stop`, `message_delta`, `message_stop`)
- **Thinking mode** — Enabled by default for non-Instruct models via the Qwen3.5 Jinja chat template
- **Multimodal** — Supported when the physical model includes an `mmproj` file

This allows using llama.cpp models as drop-in replacements with tools like Claude Code.

## Usage

### Using with Claude Code

```bash
# In Claude Code settings, point to Allama:
ANTHROPIC_BASE_URL=http://127.0.0.1:9000
ANTHROPIC_MODEL=Qwen3.5:27b-Claude-4.6
```

### Using with OpenAI-compatible Clients

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:9000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="Qwen3.5:27b-Instruct",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## License

Personal use only.
