# Allama — Contexto do Projeto

## O que é Allama?

**Allama** é um gerenciador de modelos LLM pessoal com suporte a múltiplos backends (vLLM + llama.cpp) e carga dinâmica.

### Funcionalidades Principais
- ✅ Carregamento automático de modelos sob demanda
- ✅ Gerenciamento inteligente de VRAM (aloca modelos ao melhor GPU disponível)
- ✅ Descarregamento automático de modelos ociosos
- ✅ API compatível com OpenAI (`/v1/chat/completions`) e Anthropic (`/v1/messages`)
- ✅ Suporte completo a tool calling (traduzido entre formatos)
- ✅ Múltiplos "logical models" podem compartilhar um "physical model" com sampling diferente
- ✅ CLI com REPL interativo
- ✅ Daemon watchdog (auto-restart em crashes)

---

## Estrutura do Projeto

```
allama/
├── allama.py                  # Entry point (uvicorn + signal handlers)
├── allama_cli.py              # CLI completo (serve, stop, run, etc)
├── allama_tui.py              # TUI Textual (interface interativa)
├── create_config.py           # Helper para criar configs .allm
│
├── core/                      # Núcleo da aplicação
│   ├── config.py              # Constantes, logging, load_models_from_configs()
│   ├── state.py               # Globals mutáveis (active_servers, port counters, etc)
│   ├── gpu.py                 # Gerenciamento GPU (free memory, TP selection, VRAM calc)
│   ├── process.py             # Build commands, kill process trees, shutdown
│   ├── loader.py              # LoadingSpinner, wait_for_model_ready(), ensure_physical_model()
│   ├── health.py              # Health monitor (idle timeout + crash detection)
│   └── server.py              # FastAPI app + todas as rotas
│
├── configs/
│   ├── physical/              # Configs de modelos físicos (*.allm)
│   │   ├── Qwen3.5-9b.allm
│   │   ├── Qwen3.5-27b.allm
│   │   ├── Qwen3.5-35b.allm
│   │   └── LFM2.5-VL-450M.allm
│   │
│   └── logical/               # Configs de modelos lógicos (*.allm)
│       ├── Qwen3.5-27b-Instruct.allm
│       ├── Qwen3.5-27b-Claude-4.6.allm
│       ├── Qwen3.5-9b-OCR.allm
│       └── LFM2.5-VL-450M-OCR.allm
│
├── integration/               # Integrações (Claude Code, etc)
├── docs/                      # Documentação
├── scripts/                   # Utilitários
└── logs/                      # Logs (allama.log, backends/)
```

---

## Conceitos: Physical vs Logical Models

### Physical Models (`configs/physical/*.allm`)
Define uma **instalação real** de modelo — files + backend configuration.

```ini
# vLLM backend (high-performance)
backend = "vllm"
path = "/home/nick/AI/Models/Qwen3.5-27b"
tokenizer = "/home/nick/AI/Models/Qwen3.5-27b"
gpu_memory_utilization = "0.90"
max_model_len = 262144
max_num_seqs = 12
max_num_batched_tokens = 32768
extra_args = [
    "--reasoning-parser", "qwen3",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_coder"
]
```

```ini
# llama.cpp backend (flexibility)
backend = "llama.cpp"
model = "/path/to/model.gguf"
mmproj = "/path/to/mmproj.gguf"
n_ctx = "196608"
n_batch = "1024"
n_gpu_layers = "-1"
extra_args = ["--jinja", "--flash-attn", "on"]
```

### Logical Models (`configs/logical/*.allm`)
Define **como interagir** com um physical model — qual model + sampling overrides.

```ini
name = "Qwen3.5:27b-Instruct"
physical = "Qwen3.5-27b"
enable_thinking = false

[sampling]
temperature = 0.7
top_p = 0.9
top_k = 40
min_p = 0.0
presence_penalty = 0.0
repetition_penalty = 1.0
```

**Regra**: Modelos com "instruct" no nome desativam thinking automaticamente.

---

## Configuração

### Environment Variables (em `.env`)
| Variável | Default | Uso |
|----------|---------|-----|
| `ALLAMA_PORT` | `9000` | Porta do API Allama |
| `VLLM_BASE_PORT` | `8000` | Primeira porta dos backends vLLM |
| `LLAMA_BASE_PORT` | `9001` | Primeira porta dos backends llama.cpp |
| `LLAMA_CPP_PATH` | auto-detected | Path para binary `llama-server` |
| `KEEP_ALIVE_SECONDS` | `600` | Tempo antes descarregar models ociosos |
| `HEALTH_CHECK_INTERVAL` | `60` | Intervalo de health check (seg) |
| `GPU_MEMORY_THRESHOLD_GB` | `1.0` | VRAM mínima livre para carregar models |
| `AUTO_SWAP_ENABLED` | `true` | Auto-unload models ociosos quando VRAM needed |

---

## Comandos CLI

```bash
# Servidor
allama serve              # Background daemon
allama serve -v           # Foreground com logs ao vivo
allama stop               # Para server + backends

# Status
allama status             # Server status
allama list               # Listar logical models disponíveis
allama ps                 # Modelos carregados (running)

# Logs
allama logs -f            # Tail allama logs
allama backend logs       # Tail backend logs

# Interativo
allama run <model>        # Carregar model e abrir chat interativo
```

### Workflow Típico
1. `allama serve` — inicia daemon
2. `allama run Qwen3.5:27b-Instruct` — abre chat
3. `allama stop` — para tudo

---

## Padrão de Desenvolvimento

### Após editar código:
```bash
allama stop && allama serve
```
Sempre reiniciar o server após mudanças no código (vide memory: feedback_restart_after_edit).

### Criando novo config (physical + logical):

**Physical** (`configs/physical/Model-Name.allm`):
```ini
backend = "vllm"
path = "/home/nick/AI/Models/Model-Name"
tokenizer = "/home/nick/AI/Models/Model-Name"
gpu_memory_utilization = "0.90"
max_model_len = 262144
max_num_seqs = 12
max_num_batched_tokens = 32768
```

**Logical** (`configs/logical/Model-Name-Variant.allm`):
```ini
name = "Model-Name:Variant"
physical = "Model-Name"

[sampling]
temperature = 0.7
top_p = 0.9
top_k = 40
min_p = 0.0
presence_penalty = 0.0
repetition_penalty = 1.0
```

### Para modelos -OCR:
- temperature = 0.0 ou 0.1 (determinístico)
- repetition_penalty = 1.05 (textos reais repetem)
- top_p = 1.0, top_k = -1 (sem truncation)

---

## APIs Principais

### OpenAI-compatible (`POST /v1/chat/completions`)
```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:9000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="Qwen3.5:27b-Instruct",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Anthropic Messages (`POST /v1/messages`)
Compatível com Anthropic API — tool calling é traduzido automaticamente para llama.cpp.

### Admin
- `GET /v1/models` — Lista logical models
- `POST /v1/load?model=<name>` — Pré-carregar model
- `GET /v1/ps` — Processos ativos
- `GET /health` — Health check

---

## Modelos Disponíveis

### Qwen3.5
- **9B** (dense, 9B params) — rápido, bom custo-benefício
- **27B** (dense, 27B params) — melhor qualidade, latência consistente
- **35B-A3B** (MoE sparse, 3B ativa) — rápido (111 tok/s RTX3090), qualidade altíssima
- **122B-A10B** (MoE sparse, 10B ativa) — flagship médio
- **397B-A17B** (MoE sparse, 17B ativa) — flagship

**Specs Qwen3.5**:
- Contexto nativo: 262K tokens (extensível a ~1M com YaRN)
- Multimodal nativo em todos (visão integrada ao treino)
- Tool calling nativo com auto-choice
- 201 idiomas
- Thinking mode (Qwen3) desativável

### Gemma4
- **26B-A4B** (MoE sparse, 4B ativa)
- **31B** (dense)

### LFM2.5-VL-450M
- **450M** (dense, vision) — modelo de visão compacto Liquid AI
- vLLM recomenda: temperature 0.1, min_p 0.15, max_model_len 1024

---

## Troubleshooting Rápido

**Modelo não carrega**
- Verificar `/home/nick/AI/Models/` — arquivo existe?
- `allama ps` — qual GPU está usando?
- `allama logs -f` — erros?

**VRAM alta**
- `allama ps` — qual model está rodando?
- `allama stop` — descarrega tudo
- Editar `KEEP_ALIVE_SECONDS` para descarregar mais rápido

**Erro "port already in use"**
- Algum backend crashed e ficou na porta
- Verificar: `lsof -i :8000` (ou porta vLLM)
- Matar: `kill -9 <PID>`

---

## Integração com Claude Code

No settings.json do Claude Code:
```json
{
  "anthropicBaseUrl": "http://127.0.0.1:9000",
  "anthropicModel": "Qwen3.5:27b-Claude-4.6"
}
```

---

## Notas para Futuras Conversas

- **Memory**: Consultar `/home/nick/.claude/projects/-home-nick-AI-allama/memory/MEMORY.md`
- **Restart rule**: Sempre `allama stop && allama serve` após editar code
- **Config pattern**: Physical tem backend/path; Logical tem sampling overrides
- **OCR configs**: temperature=0, repetition_penalty=1.05, sem top_k/top_p truncation
- **Models path**: `/home/nick/AI/Models/<ModelName>/`
