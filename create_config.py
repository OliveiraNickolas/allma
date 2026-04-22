#!/usr/bin/env python3
"""
Allma Config Creator — gera arquivos .allm para modelos baixados.

Uso:
    python create_config.py /path/to/model
    python create_config.py /path/to/model --name MeuModelo
    python create_config.py /path/to/model --yes   # aceita todos os defaults
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


# ==============================================================================
# Cores para terminal
# ==============================================================================
C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_DIM    = "\033[2m"
C_GREEN  = "\033[92m"
C_YELLOW = "\033[93m"
C_CYAN   = "\033[96m"
C_RED    = "\033[91m"
C_BLUE   = "\033[94m"


def bold(s):    return f"{C_BOLD}{s}{C_RESET}"
def green(s):   return f"{C_GREEN}{s}{C_RESET}"
def yellow(s):  return f"{C_YELLOW}{s}{C_RESET}"
def cyan(s):    return f"{C_CYAN}{s}{C_RESET}"
def red(s):     return f"{C_RED}{s}{C_RESET}"
def dim(s):     return f"{C_DIM}{s}{C_RESET}"


# ==============================================================================
# Presets por família de modelo
# Fontes: HuggingFace model cards + docs oficiais dos backends
# ==============================================================================
FAMILY_PRESETS = {
    # ── Qwen3.5 (texto) ──────────────────────────────────────────────────────
    "qwen3_5": {
        "label": "Qwen3.5 (texto)",
        "vllm_extra_args": [
            "--reasoning-parser", "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
        ],
        "llama_extra_args": ["--chat-template", "chatml", "--jinja"],
        "sampling": {
            "temperature": "0.7",
            "top_p": "0.8",
            "top_k": "20",
            "min_p": "0.0",
            "presence_penalty": "1.5",
        },
        "profile_variants": {
            "default":   {"temperature": "0.7", "top_p": "0.8",  "top_k": "20", "presence_penalty": "1.5"},
            "Instruct":  {"temperature": "0.7", "top_p": "0.8",  "top_k": "20"},
            "Reasoning": {"temperature": "0.6", "top_p": "0.95", "top_k": "20"},
            "Code":      {"temperature": "0.6", "top_p": "0.95", "top_k": "20"},
        },
    },
    # ── Qwen3.5-VL / Qwen3-VL (visão) ────────────────────────────────────────
    "qwen3_vl": {
        "label": "Qwen3-VL (visão)",
        "vllm_extra_args": [
            "--reasoning-parser", "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--mm-encoder-tp-mode", "data",
            "--generation-config", "vllm",
            "--limit-mm-per-prompt.video", "0",
            "--async-scheduling",
            "--disable-custom-all-reduce",
        ],
        "vllm_extra_fields": {
            "generation_config": "vllm",
            "limit_mm_per_prompt.video": "0",
        },
        "llama_extra_args": ["--chat-template", "chatml", "--jinja"],
        "sampling": {
            "temperature": "0.7",
            "top_p": "0.8",
            "top_k": "20",
            "min_p": "0.0",
            "presence_penalty": "1.5",
        },
        "profile_variants": {
            "default": {"temperature": "0.7", "top_p": "0.8",  "top_k": "20", "presence_penalty": "1.5"},
            "Caption": {"temperature": "0.3", "top_p": "0.9",  "top_k": "20"},
        },
    },
    # ── Qwen3 (MoE / A3B) ────────────────────────────────────────────────────
    "qwen3_moe": {
        "label": "Qwen3 MoE",
        "vllm_extra_args": [
            "--reasoning-parser", "qwen3",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
        ],
        "llama_extra_args": ["--chat-template", "chatml", "--jinja"],
        "sampling": {
            "temperature": "0.7",
            "top_p": "0.8",
            "top_k": "20",
            "min_p": "0.0",
        },
        "profile_variants": {
            "default":   {"temperature": "0.7", "top_p": "0.8",  "top_k": "20"},
            "Reasoning": {"temperature": "0.6", "top_p": "0.95", "top_k": "20"},
        },
    },
    # ── DeepSeek-R1 / V3 ──────────────────────────────────────────────────────
    "deepseek": {
        "label": "DeepSeek R1/V3",
        "vllm_extra_args": [
            "--reasoning-parser", "deepseek_r1",
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
        ],
        "llama_extra_args": ["--chat-template", "deepseek", "--jinja"],
        "sampling": {
            "temperature": "0.6",
            "top_p": "0.95",
            "top_k": "40",
            "min_p": "0.0",
        },
        "profile_variants": {
            "default":   {"temperature": "0.6", "top_p": "0.95", "top_k": "40"},
            "Reasoning": {"temperature": "0.5", "top_p": "0.95", "top_k": "20"},
        },
    },
    # ── Llama 3.x ────────────────────────────────────────────────────────────
    "llama": {
        "label": "Llama 3.x",
        "vllm_extra_args": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "llama3_json",
        ],
        "llama_extra_args": [],
        "sampling": {
            "temperature": "0.7",
            "top_p": "0.9",
            "top_k": "40",
            "min_p": "0.0",
        },
        "profile_variants": {
            "default": {"temperature": "0.7", "top_p": "0.9", "top_k": "40"},
            "Code":    {"temperature": "0.3", "top_p": "0.9", "top_k": "20"},
        },
    },
    # ── Mistral ───────────────────────────────────────────────────────────────
    "mistral": {
        "label": "Mistral / Mixtral",
        "vllm_extra_args": [
            "--enable-auto-tool-choice",
            "--tool-call-parser", "mistral",
        ],
        "llama_extra_args": [],
        "sampling": {
            "temperature": "0.7",
            "top_p": "0.9",
            "top_k": "40",
            "min_p": "0.0",
        },
        "profile_variants": {
            "default": {"temperature": "0.7", "top_p": "0.9", "top_k": "40"},
        },
    },
    # ── Gemma ─────────────────────────────────────────────────────────────────
    "gemma": {
        "label": "Gemma",
        "vllm_extra_args": [],
        "llama_extra_args": [],
        "sampling": {
            "temperature": "1.0",
            "top_p": "0.95",
            "top_k": "64",
            "min_p": "0.0",
        },
        "profile_variants": {
            "default": {"temperature": "1.0", "top_p": "0.95", "top_k": "64"},
        },
    },
    # ── Phi-4 / Phi-3 ─────────────────────────────────────────────────────────
    "phi": {
        "label": "Microsoft Phi",
        "vllm_extra_args": [
            "--trust-remote-code",
        ],
        "llama_extra_args": ["--jinja"],
        "sampling": {
            "temperature": "0.0",
            "top_p": "1.0",
            "top_k": "0",
            "min_p": "0.0",
        },
        "profile_variants": {
            "default": {"temperature": "0.0", "top_p": "1.0"},
        },
    },
    # ── Genérico (fallback) ───────────────────────────────────────────────────
    "generic": {
        "label": "Genérico",
        "vllm_extra_args": [],
        "llama_extra_args": [],
        "sampling": {
            "temperature": "0.7",
            "top_p": "0.9",
            "top_k": "40",
            "min_p": "0.0",
        },
        "profile_variants": {
            "default": {"temperature": "0.7", "top_p": "0.9", "top_k": "40"},
        },
    },
}

# Mapeamento: model_type / architecture → família
ARCH_TO_FAMILY = {
    # Qwen3.5
    "qwen3_5":                          "qwen3_5",
    "qwen3_5forcausallm":               "qwen3_5",
    "qwen3_5forconditionalgeneration":  "qwen3_5",
    # Qwen3 VL
    "qwen3_vl":                         "qwen3_vl",
    "qwen3vlforconditionalgeneration":  "qwen3_vl",
    # Qwen2 VL (compatível)
    "qwen2_vl":                         "qwen3_vl",
    "qwen2vlforconditionalgeneration":  "qwen3_vl",
    # Qwen3 / Qwen2 MoE
    "qwen3moeforconditionalgeneration": "qwen3_moe",
    "qwen3_moe":                        "qwen3_moe",
    "qwen2moeforconditionalgeneration": "qwen3_moe",
    # Qwen3 texto (sem MoE, sem VL)
    "qwen3forcausallm":                 "qwen3_5",
    "qwen3":                            "qwen3_5",
    # Qwen2
    "qwen2forcausallm":                 "qwen3_5",
    "qwen2":                            "qwen3_5",
    # DeepSeek
    "deepseekv3forcausallm":            "deepseek",
    "deepseek_v3":                      "deepseek",
    "deepseekr1":                       "deepseek",
    "deepseek_r1":                      "deepseek",
    "deepseekv2forcausallm":            "deepseek",
    # Llama
    "llamaforcausallm":                 "llama",
    "llama":                            "llama",
    "llama3":                           "llama",
    # Mistral
    "mistralformcausallm":              "mistral",
    "mistral":                          "mistral",
    "mixtralformcausallm":              "mistral",
    "mixtral":                          "mistral",
    # Gemma
    "gemmaforcausallm":                 "gemma",
    "gemma2forcausallm":                "gemma",
    "gemma3forcausallm":                "gemma",
    "gemma":                            "gemma",
    # Phi
    "phiforcausallm":                   "phi",
    "phi3forcausallm":                  "phi",
    "phi4forcausallm":                  "phi",
    "phi":                              "phi",
}


# ==============================================================================
# Detecção de GPU via nvidia-smi
# ==============================================================================
def get_gpus():
    """Retorna lista de GPUs com índice, total_gb e free_gb."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                gpus.append({
                    "index":    int(parts[0]),
                    "total_gb": int(parts[1]) / 1024,
                    "free_gb":  int(parts[2]) / 1024,
                })
        return gpus
    except Exception:
        return []


# ==============================================================================
# Detecção do modelo
# ==============================================================================
def detect_model(path: Path) -> dict:
    """Analisa a pasta do modelo e retorna um dict com as informações detectadas."""
    info = {
        "path":         str(path),
        "backend":      None,
        "family":       "generic",
        "model_type":   None,
        "architectures": [],
        "max_ctx":      None,
        "has_vision":   False,
        "gguf_files":   [],
        "mmproj_files": [],
        "size_gb":      0.0,
    }

    # ── Backend: GGUF → llama.cpp, safetensors → vLLM ──────────────────────
    gguf_files   = sorted(path.glob("*.gguf"))
    sf_files     = list(path.rglob("*.safetensors"))

    info["gguf_files"]   = [str(f) for f in gguf_files]
    info["mmproj_files"] = [str(f) for f in gguf_files if "mmproj" in f.name.lower()]

    if gguf_files:
        info["backend"] = "llama.cpp"
        # Tamanho: maior .gguf que não seja mmproj
        model_ggufs = [f for f in gguf_files if "mmproj" not in f.name.lower()]
        if model_ggufs:
            info["size_gb"] = model_ggufs[0].stat().st_size / (1024 ** 3)
        else:
            info["size_gb"] = gguf_files[0].stat().st_size / (1024 ** 3)
    elif sf_files:
        info["backend"] = "vllm"
        info["size_gb"] = sum(f.stat().st_size for f in sf_files) / (1024 ** 3)
    else:
        info["backend"] = "vllm"  # assume — sem arquivos detectáveis

    # ── config.json ─────────────────────────────────────────────────────────
    cfg_path = path / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            info["model_type"]    = cfg.get("model_type", "")
            info["architectures"] = cfg.get("architectures", [])
            info["has_vision"]    = "vision_config" in cfg

            # max_position_embeddings pode estar em text_config (modelos VL)
            tc = cfg.get("text_config", cfg)
            info["max_ctx"] = tc.get("max_position_embeddings")
        except Exception as e:
            print(yellow(f"⚠  Não foi possível ler config.json: {e}"))

    # ── Família — tenta por model_type, depois architectures, depois nome da pasta ──
    candidates = [info["model_type"] or ""] + [a.lower() for a in info["architectures"]]
    for c in candidates:
        key = c.lower().replace("-", "_").replace(" ", "_")
        if key in ARCH_TO_FAMILY:
            info["family"] = ARCH_TO_FAMILY[key]
            break

    # Fallback: deduz família pelo nome da pasta / arquivo .gguf
    if info["family"] == "generic":
        name_lower = path.name.lower()
        gguf_names = " ".join(Path(f).stem.lower() for f in info["gguf_files"])
        combined = name_lower + " " + gguf_names

        NAME_PATTERNS = [
            # Qwen3 VL (antes de qwen3 genérico)
            (("qwen3.vl", "qwen3vl", "qwen3-vl", "qwen3_vl",
              "qwen3.5vl", "qwen3.5-vl", "qwen3.5_vl"), "qwen3_vl"),
            # Qwen3.5 / Qwen3
            (("qwen3.5", "qwen3-5", "qwen3_5", "qwen3"), "qwen3_5"),
            # Qwen2 VL
            (("qwen2.vl", "qwen2vl", "qwen2-vl"), "qwen3_vl"),
            # Qwen2
            (("qwen2",), "qwen3_5"),
            # DeepSeek
            (("deepseek-r1", "deepseek_r1", "deepseek-v3", "deepseek_v3", "deepseek"), "deepseek"),
            # Llama
            (("llama-3", "llama3", "llama-2", "llama2", "llama"), "llama"),
            # Mistral / Mixtral
            (("mixtral", "mistral"), "mistral"),
            # Gemma
            (("gemma",), "gemma"),
            # Phi
            (("phi-4", "phi-3", "phi4", "phi3", "phi"), "phi"),
        ]
        for patterns, family in NAME_PATTERNS:
            if any(p in combined for p in patterns):
                info["family"] = family
                break

    return info


# ==============================================================================
# Sugestão de tensor_parallel e max_model_len
# ==============================================================================
def suggest_tp(size_gb: float, gpus: list) -> int:
    """Sugere tensor_parallel mínimo para caber nas GPUs disponíveis."""
    if not gpus:
        return 1
    # Estima VRAM necessária: tamanho * 1.15 (overhead de ativações + KV cache básico)
    need = size_gb * 1.15
    total_free = sum(g["free_gb"] for g in gpus)
    max_single  = max(g["free_gb"] for g in gpus)

    if max_single >= need:
        return 1
    # Quantas GPUs precisamos?
    n_gpus = len(gpus)
    for tp in [2, 4, 8]:
        if tp > n_gpus:
            break
        # Para vLLM TP, divide igualmente — estima que cada GPU recebe size/tp
        per_gpu = need / tp
        if max_single >= per_gpu:
            return tp
    return n_gpus  # usa tudo


def suggest_max_len(max_ctx: int | None, size_gb: float, tp: int, gpus: list) -> int:
    """Sugere max_model_len conservador baseado no contexto máximo e VRAM disponível."""
    native = max_ctx or 131072

    # Cap conservador: acima de 256k é raro ter VRAM suficiente para KV cache
    # Fórmula simplificada: KV cache ≈ layers * heads * ctx * 4 bytes
    # Para simplificar: limita a 128k se < 32GB total, 256k se >= 32GB
    total_free = sum(g["free_gb"] for g in gpus) if gpus else 0
    if total_free < 32:
        cap = 65536
    elif total_free < 64:
        cap = 131072
    else:
        cap = 262144

    return min(native, cap)


# ==============================================================================
# Geração dos arquivos .allm
# ==============================================================================
def render_extra_args(args: list) -> str:
    """Renderiza lista de extra_args no formato multi-linha do .allm."""
    if not args:
        return "[]"
    items = ",\n\t".join(f'"{a}"' for a in args)
    return f"[\n\t{items}\n]"


def generate_base_allm(
    name: str,
    info: dict,
    preset: dict,
    tp: int,
    max_len: int,
    gguf_path: str | None,
    mmproj_path: str | None,
) -> str:
    """Gera o conteúdo do arquivo .allm físico."""
    lines = [f"# Base model: {name} ({info['backend']} backend)\n"]

    if info["backend"] == "vllm":
        extra_args = render_extra_args(preset.get("vllm_extra_args", []))
        lines += [
            f'backend = "vllm"',
            f'path = "{info["path"]}"',
            f'tokenizer = "{info["path"]}"',
            f'tensor_parallel = "{tp}"',
            f'gpu_memory_utilization = "0.90"',
            f'max_model_len = "{max_len}"',
            f'max_num_seqs = "8"',
        ]
        for k, v in preset.get("vllm_extra_fields", {}).items():
            lines.append(f'{k} = "{v}"')
        lines += ["", f"extra_args = {extra_args}"]

    else:  # llama.cpp
        extra_args = render_extra_args(preset.get("llama_extra_args", []))
        lines += [
            f'backend = "llama.cpp"',
            f'model = "{gguf_path}"',
        ]
        if mmproj_path:
            lines.append(f'mmproj = "{mmproj_path}"')
        lines += [
            f'n_ctx = "{min(max_len, 40960)}"',
            f'n_batch = "1024"',
            f'n_gpu_layers = "-1"',
            f'n_threads = "16"',
            "",
            f"extra_args = {extra_args}",
        ]

    return "\n".join(lines) + "\n"


def generate_profile_allm(
    profile_name: str,
    base_name: str,
    sampling: dict,
) -> str:
    """Gera o conteúdo do arquivo .allm lógico."""
    lines = [
        f"# Profile model: {profile_name}",
        f'name = "{profile_name}"',
        f'base = "{base_name}"',
        "",
        "[sampling]",
    ]
    for k, v in sampling.items():
        lines.append(f"{k} = {v}")
    return "\n".join(lines) + "\n"


# ==============================================================================
# Utilitários de prompt interativo
# ==============================================================================
def ask(prompt: str, default: str, auto: bool) -> str:
    """Pergunta ao usuário com um default. Se auto=True, usa o default."""
    if auto:
        print(f"  {prompt}: {green(default)}")
        return default
    val = input(f"  {prompt} [{cyan(default)}]: ").strip()
    return val if val else default


def ask_int(prompt: str, default: int, auto: bool) -> int:
    """Pergunta ao usuário por um inteiro com validação."""
    default_str = str(default)
    if auto:
        print(f"  {prompt}: {green(default_str)}")
        return default
    while True:
        val = input(f"  {prompt} [{cyan(default_str)}]: ").strip()
        if not val:
            return default
        try:
            return int(val)
        except ValueError:
            print(f"    {red('✗ Invalid integer')} — try again")


def ask_list(prompt: str, default: list, auto: bool) -> list:
    """Pergunta por uma lista JSON. Se vazio, usa o default."""
    default_str = json.dumps(default)
    raw = ask(prompt + " (JSON)", default_str, auto)
    try:
        result = json.loads(raw)
        return result if isinstance(result, list) else default
    except json.JSONDecodeError:
        print(yellow("  ⚠  JSON inválido — usando default"))
        return default


def ask_yes(prompt: str, auto: bool) -> bool:
    if auto:
        print(f"  {prompt}: {green('s')}")
        return True
    val = input(f"  {prompt} [S/n]: ").strip().lower()
    return val not in ("n", "no", "não", "nao")


def pick_gguf(gguf_files: list, auto: bool) -> str | None:
    """Pede ao usuário para escolher o arquivo .gguf principal."""
    model_ggufs = [f for f in gguf_files if "mmproj" not in Path(f).name.lower()]
    if not model_ggufs:
        return None
    if len(model_ggufs) == 1:
        print(f"  GGUF detectado: {green(model_ggufs[0])}")
        return model_ggufs[0]

    print(bold("\n  Múltiplos GGUFs encontrados — escolha o modelo principal:"))
    for i, f in enumerate(model_ggufs):
        size = Path(f).stat().st_size / (1024 ** 3)
        print(f"    {cyan(str(i))} — {Path(f).name}  {dim(f'{size:.1f}GB')}")
    if auto:
        print(f"  → Usando {green(model_ggufs[0])}")
        return model_ggufs[0]
    idx = input(f"  Índice [0]: ").strip()
    try:
        return model_ggufs[int(idx)] if idx else model_ggufs[0]
    except (ValueError, IndexError):
        return model_ggufs[0]


# ==============================================================================
# Fluxo principal
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Gerador de configs .allm para modelos Allma",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
            Exemplos:
              python create_config.py /path/to/Qwen3.5-9b
              python create_config.py /path/to/model --name MeuModelo --yes
        """),
    )
    parser.add_argument("model_path", help="Pasta do modelo baixado")
    parser.add_argument("--name", "-n", help="Nome base para os configs (ex: Qwen3.5-9b)")
    parser.add_argument("--yes",  "-y", action="store_true", help="Aceitar todos os defaults sem perguntar")
    parser.add_argument("--config-dir", default="configs", help="Diretório de configs do Allma (default: configs)")
    args = parser.parse_args()

    auto = args.yes
    config_dir = Path(args.config_dir)
    model_path = Path(args.model_path).resolve()

    if not model_path.exists():
        print(red(f"❌  Pasta não encontrada: {model_path}"))
        sys.exit(1)

    print(bold(f"\n{'═'*60}"))
    print(bold(f"  Allma Config Creator"))
    print(bold(f"{'═'*60}"))
    print(f"  Pasta: {cyan(str(model_path))}")

    # ── Detectar modelo ──────────────────────────────────────────────────────
    info = detect_model(model_path)
    gpus = get_gpus()

    print(f"\n{bold('📦 Detecção automática:')}")
    print(f"  Backend    : {green(info['backend'])}")
    print(f"  Família    : {green(FAMILY_PRESETS[info['family']]['label'])}")
    print(f"  model_type : {dim(info['model_type'] or 'não detectado')}")
    print(f"  Contexto   : {dim(str(info['max_ctx']) if info['max_ctx'] else 'não detectado')}")
    size_str = f"{info['size_gb']:.1f}GB"
    print(f"  Tamanho    : {dim(size_str)}")
    print(f"  Visão      : {dim('sim' if info['has_vision'] else 'não')}")
    if gpus:
        gpu_str = ", ".join(f"GPU{g['index']} {g['free_gb']:.0f}GB livre" for g in gpus)
        print(f"  GPUs       : {dim(gpu_str)}")

    preset = FAMILY_PRESETS[info["family"]]
    tp_suggested   = suggest_tp(info["size_gb"], gpus) if info["backend"] == "vllm" else 1
    len_suggested  = suggest_max_len(info["max_ctx"], info["size_gb"], tp_suggested, gpus)

    # ── Nome físico ──────────────────────────────────────────────────────────
    default_name = args.name or model_path.name
    # Limpa caracteres problemáticos
    default_name = default_name.replace("/", "-").replace(" ", "-")

    print(f"\n{bold('⚙️  Configuração do modelo físico:')}")
    phys_name = ask("Nome físico (arquivo em configs/base/)", default_name, auto)

    # ── Backend ──────────────────────────────────────────────────────────────
    backend = ask("Backend (vllm / llama.cpp)", info["backend"], auto)

    # ── GGUF ─────────────────────────────────────────────────────────────────
    gguf_path   = None
    mmproj_path = None
    if backend == "llama.cpp":
        gguf_path = pick_gguf(info["gguf_files"], auto)
        if not gguf_path:
            gguf_path = ask("Caminho do arquivo .gguf", "", auto)
        if info["mmproj_files"]:
            print(f"  mmproj detectado: {green(info['mmproj_files'][0])}")
            mmproj_path = info["mmproj_files"][0]
            if not auto:
                custom = input(f"  mmproj [{cyan(mmproj_path)}]: ").strip()
                if custom:
                    mmproj_path = custom

    # ── Tensor parallel / max_model_len ──────────────────────────────────────
    if backend == "vllm":
        tp       = ask_int("tensor_parallel", tp_suggested, auto)
        max_len  = ask_int("max_model_len", len_suggested, auto)
        gpu_util = ask("gpu_memory_utilization", "0.90", auto)
        max_seqs = ask("max_num_seqs", "8", auto)

        extra_args_default = preset.get("vllm_extra_args", [])
        extra_args = ask_list("extra_args vLLM", extra_args_default, auto)
    else:
        tp = 1
        max_len = ask_int("n_ctx", min(len_suggested, 40960), auto)
        n_threads = ask("n_threads", "16", auto)
        extra_args_default = preset.get("llama_extra_args", [])
        extra_args = ask_list("extra_args llama.cpp", extra_args_default, auto)

    # Sobrescreve preset com valores do usuário
    preset_copy = dict(preset)
    if backend == "vllm":
        preset_copy["vllm_extra_args"] = extra_args
    else:
        preset_copy["llama_extra_args"] = extra_args

    # ── Gera base.allm ───────────────────────────────────────────────────
    phys_content = generate_base_allm(
        name=phys_name,
        info={**info, "backend": backend},
        preset=preset_copy,
        tp=tp,
        max_len=max_len,
        gguf_path=gguf_path,
        mmproj_path=mmproj_path,
    )
    if backend == "vllm":
        # Substituir valores custom
        phys_content = phys_content.replace(
            'gpu_memory_utilization = "0.90"',
            f'gpu_memory_utilization = "{gpu_util}"',
        ).replace(
            'max_num_seqs = "8"',
            f'max_num_seqs = "{max_seqs}"',
        )
    else:
        phys_content = phys_content.replace(
            'n_threads = "16"',
            f'n_threads = "{n_threads}"',
        )

    # ── Variantes lógicas ────────────────────────────────────────────────────
    print(f"\n{bold('🧩 Modelos lógicos (samplings):')}")
    variants = preset.get("profile_variants", {"default": preset["sampling"]})

    # Deriva nome base lógico do nome físico (ex: "Qwen3.5-9b" → "Qwen3.5:9b")
    import re
    m = re.search(r"-(\d+\.?\d*[bBmM])", phys_name)
    profile_base = phys_name[:m.start()] + ":" + phys_name[m.start() + 1:] if m else phys_name

    profile_configs = []
    for variant_key, variant_sampling in variants.items():
        if variant_key == "default":
            default_profile = profile_base
        else:
            default_profile = f"{profile_base}-{variant_key}"

        print(f"\n  {bold(f'Variante: {variant_key}')}")
        log_name = ask("  Nome do perfil", default_profile, auto)

        # Sampling
        sampling = dict(preset["sampling"])
        sampling.update(variant_sampling)
        print(f"  Sampling sugerido: {dim(str(sampling))}")
        if not auto:
            print("  Pressione Enter para aceitar, ou edite campo a campo:")
            for k, v in list(sampling.items()):
                new_v = input(f"    {k} [{cyan(str(v))}]: ").strip()
                if new_v:
                    sampling[k] = new_v

        log_content = generate_profile_allm(log_name, phys_name, sampling)
        profile_configs.append((log_name, log_content))

    # ── Preview ───────────────────────────────────────────────────────────────
    print(f"\n{bold('📄 Preview — base:')}")
    for line in phys_content.splitlines():
        print(f"  {dim(line)}")

    for log_name, log_content in profile_configs:
        print(f"\n{bold(f'📄 Preview — profile ({log_name}):')}")
        for line in log_content.splitlines():
            print(f"  {dim(line)}")

    # ── Confirmar e gravar ─────────────────────────────────────────────────────
    print()
    if not ask_yes("Gravar os arquivos?", auto):
        print(yellow("  Cancelado."))
        sys.exit(0)

    phys_dir = config_dir / "base"
    log_dir  = config_dir / "profile"
    phys_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    phys_file = phys_dir / f"{phys_name}.allm"
    if phys_file.exists() and not auto:
        if not ask_yes(f"  {phys_file} já existe — sobrescrever?", auto):
            print(yellow("  Físico não sobrescrito."))
        else:
            phys_file.write_text(phys_content)
            print(green(f"  ✔ {phys_file}"))
    else:
        phys_file.write_text(phys_content)
        print(green(f"  ✔ {phys_file}"))

    for log_name, log_content in profile_configs:
        log_file = log_dir / f"{log_name.replace(':', '-')}.allm"
        if log_file.exists() and not auto:
            if not ask_yes(f"  {log_file} já existe — sobrescrever?", auto):
                print(yellow(f"  Lógico '{log_name}' não sobrescrito."))
                continue
        log_file.write_text(log_content)
        print(green(f"  ✔ {log_file}"))

    print(f"\n{green(bold('✅ Concluído!'))} Reinicie o Allma para carregar os novos configs.\n")


if __name__ == "__main__":
    main()
