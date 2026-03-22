# Allama

A personal LLM model manager with dynamic loading and support for both vLLM and llama.cpp backends.

> **Note:** Currently in early stages, designed for personal use.

## Overview

Allama manages multiple LLM models dynamically, automatically handling:

- **Automatic model loading** - Models load on-demand when first requested
- **Smart VRAM management** - Allocates models to best available GPU based on free memory
- **Idle model unloading** - Automatically unloads models after being idle to free up VRAM
- **Multi-backend support** - Use vLLM for high-performance serving or llama.cpp for flexibility
- **Unified API** - Compatible with OpenAI-compatible clients (`/v1/chat/completions`) and Anthropic Messages API (`/v1/messages`)
- **Logical model abstraction** - Multiple logical models can share the same physical model with different sampling parameters

## Architecture

### Physical Models
Define actual model installations (the set of model files and backend configuration).

Location: `configs/physical/*.allm`

### Logical Models
Define how to interact with a physical model - which physical model it uses plus optional sampling parameter overrides.

Location: `configs/logical/*.allm`

An unlimited number of logical models can be created for each physical model.

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install fastapi httpx psutil uvicorn rich
   ```
3. Installation and configuration of vLLM and llama.cpp is assumed to be present on the system

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLAMA_PORT` | 9000 | Port for the Allama server |
| `VLLM_BASE_PORT` | 8000 | Starting port for vLLM servers |
| `LLAMA_BASE_PORT` | 9001 | Starting port for llama.cpp servers |
| `KEEP_ALIVE_SECONDS` | 600 | How long to keep models loaded after last request (minutes) |
| `HEALTH_CHECK_INTERVAL` | 60 | Interval for health checks |
| `GPU_MEMORY_THRESHOLD_GB` | 1.0 | Minimum free VRAM to continue loading new models |
| `AUTO_SWAP_ENABLED` | true | Whether to automatically unload idle models |
| `SWAP_IDLE_THRESHOLD` | 300 | Seconds of idle time before unloading |

### Model Config Format

**Physical model config** (`configs/physical/name.allm`):

```ini
backend = "vllm"
path = "/path/to/model"
tokenizer = "/path/to/tokenizer"
tensor_parallel = "2"
max_model_len = "40960"
max_num_seqs = "8"
gpu-memory-utilization = "0.90"

[sampling]
temperature = 0.7
top_p = 0.95
```

**Logical model config** (`configs/logical/name.allm`):

```ini
physical = "qwen3.5-27b"

[sampling]
temperature = 0.1
top_p = 0.95
max_tokens = 4096
```

## API Endpoints

### OpenAI Compatible

- `POST /v1/chat/completions` - Standard chat completion endpoint
- `GET /v1/models` - List available models
- `POST /v1/messages` - Anthropic-compatible messages API
- `GET /health` - Health check endpoint

## Usage

### Start the Server

```bash
python allama.py
```

### Using with OpenAI-compatible Clients

Configure your client to use Allama as the base URL:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:9000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="qwen3.5-27b-instruct",  # Logical model name
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## License

Personal use only.
