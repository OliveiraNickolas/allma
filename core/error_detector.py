import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ErrorAnalysis:
    """Análise estruturada de um erro."""
    error_type: str  # ex: "cuda_out_of_memory"
    severity: str    # "ERROR" ou "WARNING"
    raw_message: str  # Mensagem original detectada
    explanation: str  # Explicação em português
    suggestions: list[str]  # Opções de correção
    auto_fix_available: bool = False
    auto_fix_action: Optional[str] = None  # ex: "reduce_ubatch_size"


class ErrorDetector:
    """Detecta e analisa erros conhecidos em logs de backend."""

    # Padrões de erro em regex (priority order)
    ERROR_PATTERNS = {
        "cuda_out_of_memory": [
            r"CUDA out of memory",
            r"cuda runtime error.*out of memory",
            r"RuntimeError.*CUDA out of memory",
            r"torch\.cuda\.OutOfMemoryError",
            r"cuMemCreate.*out of memory",
            r"CUDA error.*out of memory",
        ],
        "cuda_allocation_failed": [
            r"Failed to allocate",
            r"allocation failed",
            r"cannot allocate memory",
            r"memory allocation failed",
        ],
        "tensor_parallel_failed": [
            r"tensor.?parallel",
            r"cannot split.*GPU",
            r"not enough memory for tensor parallel",
            r"TP size.*exceeds",
        ],
        "model_too_large": [
            r"model.*too large",
            r"exceeds.*memory",
            r"model exceeds",
            r"insufficient VRAM",
        ],
        "context_too_large": [
            r"max_model_len.*too large",
            r"context.*exceeds",
            r"n_ctx.*too large",
            r"context length.*exceeds",
        ],
        "vram_fragmentation": [
            r"fragmentation",
            r"defrag",
            r"memory fragmentation",
        ],
        "invalid_model_path": [
            r"No such file or directory.*model",
            r"cannot find.*model",
            r"path does not exist",
            r"model file not found",
        ],
        "tokenizer_load_failed": [
            r"Failed to load tokenizer",
            r"cannot load tokenizer",
            r"tokenizer.*not found",
        ],
        "compute_capability_mismatch": [
            r"compute capability",
            r"SM version",
            r"NVIDIA.*not supported",
        ],
    }

    # Sugestões por tipo de erro
    SUGGESTIONS = {
        "cuda_out_of_memory": [
            "Reduzir --ubatch-size (ex: 1024 → 512 → 256)",
            "Reduzir --cache-type (usar q8_0 em vez de fp16 para KV cache)",
            "Reduzir max_model_len (ex: 262K → 131K)",
            "Usar tensor-parallel com múltiplas GPUs",
            "Usar quantização (Q4, Q5 em vez de fp16)",
            "Aumentar --defrag-thold para desfraginizar memória",
        ],
        "cuda_allocation_failed": [
            "GPU pode estar fragilizada, reinicie o servidor",
            "Verificar se há outros processos usando GPU (nvidia-smi)",
            "Reduzir gpu_memory_utilization em base config",
            "Reiniciar NVIDIA driver: sudo systemctl restart nvidia-persistenced",
        ],
        "tensor_parallel_failed": [
            "Verificar se todas as GPUs têm memória suficiente",
            "Reduzir tensor-parallel-size",
            "Usar GPU_MEMORY_THRESHOLD_GB para limpar VRAM antes",
            "Verificar CUDA_VISIBLE_DEVICES está correto",
        ],
        "model_too_large": [
            "Modelo não cabe em GPU disponível",
            "Opções: (1) Aumentar GPUs, (2) Usar quantização, (3) Usar modelo menor",
            "Reduzir max_model_len para economizar VRAM",
        ],
        "context_too_large": [
            "Contexto muito grande para GPU",
            "Reduzir n_ctx em base config",
            "Usar --yarn-scale-factor para extensão com YaRN",
        ],
        "vram_fragmentation": [
            "Aumentar --defrag-thold (ex: 0.1 → 0.05)",
            "Reduzir max_num_seqs para menos requisições em paralelo",
            "Reiniciar server para limpar fragmentação",
        ],
        "invalid_model_path": [
            "Verificar path do modelo em base config",
            "Confirmar arquivo existe em /home/nick/AI/Models/",
            "Verificar permissões: ls -la /path/to/model",
        ],
        "tokenizer_load_failed": [
            "Verificar tokenizer path em base config",
            "Confirmar arquivo tokenizer.model ou tokenizer.json existe",
            "Usar chat_template-file correto",
        ],
        "compute_capability_mismatch": [
            "GPU não suporta compilação CUDA necessária",
            "Usar build pré-compilado ou reduzir capability requisito",
        ],
    }

    @staticmethod
    def analyze_log(log_content: str) -> Optional[ErrorAnalysis]:
        """
        Analisa conteúdo de log para detectar padrões de erro conhecidos.
        Retorna ErrorAnalysis ou None se nenhum padrão detectado.
        """
        if not log_content:
            return None

        # Procurar por cada tipo de erro em ordem de priority
        for error_type, patterns in ErrorDetector.ERROR_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, log_content, re.IGNORECASE | re.MULTILINE)
                if match:
                    raw_msg = match.group(0)
                    explanation = ErrorDetector._get_explanation(error_type, log_content)
                    suggestions = ErrorDetector.SUGGESTIONS.get(error_type, [])
                    auto_fix_action = ErrorDetector._get_auto_fix_action(error_type, log_content)

                    return ErrorAnalysis(
                        error_type=error_type,
                        severity="ERROR",
                        raw_message=raw_msg,
                        explanation=explanation,
                        suggestions=suggestions,
                        auto_fix_available=auto_fix_action is not None,
                        auto_fix_action=auto_fix_action,
                    )

        return None

    @staticmethod
    def _get_explanation(error_type: str, log_content: str) -> str:
        """Retorna explicação em português para cada tipo de erro."""
        explanations = {
            "cuda_out_of_memory": (
                "CUDA ficou sem memória durante alocação ou execução. "
                "A GPU não tem memória suficiente para o modelo com estas configurações."
            ),
            "cuda_allocation_failed": (
                "Falha ao alocar memória CUDA. GPU pode estar fragilizada ou sem espaço."
            ),
            "tensor_parallel_failed": (
                "Falha ao dividir modelo entre GPUs (tensor-parallel). "
                "Uma ou mais GPUs podem não ter memória suficiente."
            ),
            "model_too_large": (
                "Modelo é maior que memória disponível em GPU. "
                "Precisará de mais GPUs, quantização, ou modelo menor."
            ),
            "context_too_large": (
                "Contexto (n_ctx) configurado é muito grande para GPU. "
                "Reduzir n_ctx ou usar menos requisições em paralelo."
            ),
            "vram_fragmentation": (
                "VRAM is fragmented. Many small free spaces, no large contiguous block available."
            ),
            "invalid_model_path": (
                "Arquivo de modelo não encontrado em path configurado. "
                "Verificar se arquivo existe e path está correto."
            ),
            "tokenizer_load_failed": (
                "Tokenizer não encontrado ou falhou ao carregar. "
                "Verificar path do tokenizer em base config."
            ),
            "compute_capability_mismatch": (
                "GPU não suporta compilação CUDA necessária para este modelo."
            ),
        }
        return explanations.get(error_type, "Erro desconhecido")

    @staticmethod
    def _get_auto_fix_action(error_type: str, log_content: str) -> Optional[str]:
        """
        Retorna ação de auto-correção se disponível para este erro.
        Retorna None se erro requer intervenção manual.
        """
        if error_type == "cuda_out_of_memory":
            # Detectar se é ubatch-size que pode ser facilmente reduzido
            if re.search(r"ubatch|batch", log_content, re.IGNORECASE):
                return "reduce_ubatch_size"
            # Detectar se é context/max_model_len
            if re.search(r"max_model_len|n_ctx|context", log_content, re.IGNORECASE):
                return "reduce_context_length"
            # Caso genérico CUDA OOM
            return "reduce_batch_params"

        if error_type == "vram_fragmentation":
            return "increase_defrag_threshold"

        # Outros erros requerem ação manual
        return None

    @staticmethod
    def analyze_exit_code(exit_code: int, backend: str) -> Optional[str]:
        """
        Interpreta exit code do processo para inferir causa de falha.
        Retorna descrição legível ou None se desconhecido.
        """
        if exit_code == 0:
            return None  # Sucesso

        if exit_code == 1:
            return "Falha genérica (erro na inicialização ou fatal)"

        if exit_code == 127:
            return f"Comando não encontrado (verifique path do {backend})"

        if exit_code == -9 or exit_code == 137:
            return "Processo foi morto (SIGKILL, possivelmente OOM killer)"

        if exit_code == -15 or exit_code == 143:
            return "Processo terminado (SIGTERM, shutdown normal)"

        if exit_code > 128:
            signal_num = exit_code - 128
            return f"Terminado por signal {signal_num} (saída anormal)"

        return f"Exit code {exit_code} (erro desconhecido)"


def tail_file(file_path: str, lines: int = 50) -> str:
    """Lê as últimas N linhas de um arquivo."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            return "".join(all_lines[-lines:])
    except Exception as e:
        return f"Erro ao ler arquivo: {e}"
