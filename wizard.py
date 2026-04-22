#!/usr/bin/env python3
"""
Allma TUI — Model Import Wizard
Multi-step wizard for adding base + profile model configs.
"""
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Button, Input, Label, Select, Static

from create_config import (
    FAMILY_PRESETS,
    detect_model,
    generate_profile_allm,
    generate_base_allm,
    suggest_max_len,
    suggest_tp,
    get_gpus,
)

BASE_DIR   = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "configs"
STEP_TOTAL = 5

STEP_LABELS = [
    (1, "DIRETÓRIO",  "Localizar modelo"),
    (2, "ANÁLISE",    "Detectar arquivos"),
    (3, "FÍSICO",     "Configurar backend"),
    (4, "LÓGICO",     "Definir perfis"),
    (5, "REVISÃO",    "Salvar configs"),
]


@dataclass
class WizardState:
    model_path:   Optional[Path] = None
    info:         dict           = field(default_factory=dict)
    gpus:         list           = field(default_factory=list)
    backend:      str  = ""
    gguf_path:    str  = ""
    mmproj_path:  str  = ""
    phys_name:    str  = ""
    tp:           int  = 1
    max_len:      int  = 131072
    gpu_util:     str  = "0.90"
    max_num_seqs: str  = "8"
    n_ctx:        int  = 40960
    n_threads:    str  = "16"
    n_gpu_layers: str  = "-1"
    extra_args:   list = field(default_factory=list)
    profiles: list = field(default_factory=list)


def _nav_panel(current: int) -> str:
    """Render the step navigator sidebar text with Rich markup."""
    lines = ["", "  [bold #007878]ETAPAS[/bold #007878]",
             "  [#008888]──────────────────[/]", ""]
    for n, name, desc in STEP_LABELS:
        if n < current:
            lines.append(f"  [#6a5a48]✓ {n}  {name}[/]")
        elif n == current:
            lines.append(f"  [bold #007878]▶ {n}  {name}[/]")
            lines.append(f"  [#6a5a48]     {desc}[/]")
        else:
            lines.append(f"  [#a09080]  {n}  {name}[/]")
        lines.append("")
    return "\n".join(lines)


def _section(text: str) -> Static:
    return Static(f"  {text}", classes="wizard-section")


def _hint(text: str) -> Static:
    return Static(f"  {text}", classes="wizard-hint")


def _setup_tip(screen) -> None:
    """Set border title on the tip box."""
    try:
        screen.query_one("#tip-box").border_title = "  Tip:"
    except Exception:
        pass


# ─── Step 1: Diretório ───────────────────────────────────────────────────────
class WizardStep1Screen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back", show=False),
        Binding("f10",    "back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        with Container(classes="shadow-wrap"):
            with Container(id="main-window"):
                with Horizontal(id="wizard-layout"):
                    with Vertical(id="step-nav"):
                        yield Static(_nav_panel(1), id="nav-text")
                    with Vertical(id="step-content"):
                        with ScrollableContainer(id="wizard-container"):
                            yield _section("Onde está o modelo?")
                            with Vertical(classes="wizard-field"):
                                yield Label("  Diretório do modelo:")
                                yield Input(
                                    placeholder="/home/user/AI/Models/Qwen3.5-27b",
                                    id="path-input",
                                )
                                yield _hint(
                                    "Cole o caminho completo da pasta com os pesos do modelo.\n"
                                    "  Compatível com safetensors (HuggingFace) e GGUF."
                                )
                            yield Static("", id="detect-status")
                        with Container(id="tip-box"):
                            yield Static(
                                "  O Allma procurará arquivos de pesos nesta pasta (GGUF, SafeTensors).",
                                id="tip-text",
                            )
                        with Horizontal(id="wizard-nav"):
                            yield Button("◄ CANCELAR  [ESC]", id="cancel-btn")
                            yield Button("ESCANEAR & CONTINUAR  ►", id="next-btn", variant="primary")
                yield Static("<ESC: Cancelar>  <F10: Cancelar>", id="fkey-bar")

    def on_mount(self) -> None:
        self.query_one("#main-window").border_title = "[ ADICIONAR MODELO — PASSO 1/5 ]"
        _setup_tip(self)
        self.query_one("#path-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.action_back()
        elif event.button.id == "next-btn":
            self._do_scan()

    def _do_scan(self) -> None:
        raw    = self.query_one("#path-input", Input).value.strip()
        status = self.query_one("#detect-status", Static)

        if not raw:
            status.update("  [red]Informe um caminho.[/red]")
            return

        path = Path(raw).expanduser().resolve()
        if not path.exists():
            status.update(f"  [red]Caminho não encontrado:[/red] {path}")
            return
        if not path.is_dir():
            status.update(f"  [red]Não é um diretório:[/red] {path}")
            return

        status.update("  Escaneando arquivos do modelo...")
        try:
            info = detect_model(path)
            gpus = get_gpus()
        except Exception as e:
            status.update(f"  [red]Falha ao escanear:[/red] {e}")
            return

        preset = FAMILY_PRESETS[info["family"]]
        status.update(
            f"  [#007878]✔ Encontrado:[/#007878]  "
            f"[bold]{preset['label']}[/bold]  ·  "
            f"{info['backend'].upper()}  ·  "
            f"{info['size_gb']:.1f} GB  ·  "
            f"{'visão' if (info['has_vision'] or info['mmproj_files']) else 'somente texto'}"
        )
        state = WizardState(model_path=path, info=info, gpus=gpus)
        self.app.push_screen(WizardStep2Screen(state))

    def action_back(self) -> None:
        self.app.pop_screen()


# ─── Step 2: Análise Automática ──────────────────────────────────────────────
class WizardStep2Screen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back", show=False),
        Binding("f10",    "back", "Back"),
    ]

    def __init__(self, state: WizardState, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    def compose(self) -> ComposeResult:
        info   = self._state.info
        gpus   = self._state.gpus
        preset = FAMILY_PRESETS[info["family"]]

        with Container(classes="shadow-wrap"):
            with Container(id="main-window"):
                with Horizontal(id="wizard-layout"):
                    with Vertical(id="step-nav"):
                        yield Static(_nav_panel(2), id="nav-text")
                    with Vertical(id="step-content"):
                        with ScrollableContainer(id="wizard-container"):
                            yield _section("Allma analisou o diretório")
                            yield Static(self._summary_text(info, gpus, preset), id="detect-summary")

                            yield _section("Escolha o backend de inferência:")
                            with Vertical(classes="wizard-field"):
                                yield Label("  Backend:")
                                yield Select(
                                    [
                                        ("vLLM  —  safetensors HuggingFace, alto throughput", "vllm"),
                                        ("llama.cpp  —  formato GGUF, inferência local, portável", "llama.cpp"),
                                    ],
                                    id="backend-select",
                                    value=info["backend"],
                                )
                                yield _hint(
                                    "vLLM: melhor para safetensors e serving de alta demanda.\n"
                                    "  llama.cpp: melhor para GGUF e uso local."
                                )

                            model_ggufs = [f for f in info.get("gguf_files", [])
                                           if "mmproj" not in Path(f).name.lower()]
                            if len(model_ggufs) > 1:
                                yield _section("Múltiplos GGUFs — escolha o modelo principal:")
                                with Vertical(classes="wizard-field"):
                                    yield Label("  Arquivo GGUF:")
                                    yield Select(
                                        [(Path(f).name, f) for f in model_ggufs],
                                        id="gguf-select",
                                        value=model_ggufs[0],
                                    )
                            elif len(model_ggufs) == 1:
                                yield Static(
                                    f"\n  GGUF: [#007878]{Path(model_ggufs[0]).name}[/#007878]"
                                )

                        with Container(id="tip-box"):
                            yield Static(
                                "  Confirmaremos se o modelo pode ser carregado com as configurações detectadas.",
                                id="tip-text",
                            )
                        with Horizontal(id="wizard-nav"):
                            yield Button("◄ VOLTAR", id="back-btn")
                            yield Button("PRÓXIMO  ►", id="next-btn", variant="primary")
                yield Static("<ESC: Voltar>", id="fkey-bar")

    def _summary_text(self, info, gpus, preset) -> str:
        lines = [""]
        lines.append(f"  Família do modelo  :  [bold]{preset['label']}[/bold]")
        fmt = "GGUF" if info.get("gguf_files") else "safetensors (HuggingFace)"
        lines.append(f"  Formato            :  {fmt}")
        lines.append(f"  Backend sugerido   :  [#007878]{info['backend']}[/]")
        lines.append(f"  Tamanho            :  {info['size_gb']:.1f} GB")
        ctx = info.get("max_ctx")
        lines.append(f"  Contexto máximo    :  {ctx:,} tokens" if ctx else "  Contexto máximo    :  não detectado")
        has_vis = info.get("has_vision") or bool(info.get("mmproj_files"))
        lines.append(f"  Visão              :  {'[#007878]sim[/]' if has_vis else 'não'}")
        if gpus:
            for g in gpus:
                name = g.get("name", f"GPU {g['index']}")
                lines.append(f"  GPU {g['index']}              :  {name}  ·  {g['free_gb']:.0f} GB livres")
        else:
            lines.append("  GPUs               :  [#a09080]nenhuma detectada (CPU apenas)[/]")
        lines.append("")
        return "\n".join(lines)

    def on_mount(self) -> None:
        self.query_one("#main-window").border_title = "[ ADICIONAR MODELO — PASSO 2/5 ]"
        _setup_tip(self)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.action_back()
        elif event.button.id == "next-btn":
            self._do_next()

    def _do_next(self) -> None:
        sel = self.query_one("#backend-select", Select)
        if sel.value is Select.BLANK:
            return
        backend = str(sel.value)

        info        = self._state.info
        model_ggufs = [f for f in info.get("gguf_files", [])
                       if "mmproj" not in Path(f).name.lower()]
        mmproj_list = info.get("mmproj_files", [])

        gguf_path = ""
        if backend == "llama.cpp":
            if len(model_ggufs) > 1:
                gs = self.query_one("#gguf-select", Select)
                gguf_path = str(gs.value) if gs.value is not Select.BLANK else model_ggufs[0]
            elif len(model_ggufs) == 1:
                gguf_path = model_ggufs[0]

        state             = deepcopy(self._state)
        state.backend     = backend
        state.gguf_path   = gguf_path
        state.mmproj_path = mmproj_list[0] if mmproj_list else ""
        self.app.push_screen(WizardStep3Screen(state))

    def action_back(self) -> None:
        self.app.pop_screen()


# ─── Step 3: Configuração Física ─────────────────────────────────────────────
class WizardStep3Screen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back", show=False),
        Binding("f10",    "back", "Back"),
    ]

    def __init__(self, state: WizardState, **kwargs):
        super().__init__(**kwargs)
        self._state = state

    def _defaults(self) -> dict:
        info    = self._state.info
        gpus    = self._state.gpus
        backend = self._state.backend
        preset  = FAMILY_PRESETS[info["family"]]

        tp      = suggest_tp(info["size_gb"], gpus) if backend == "vllm" else 1
        max_len = suggest_max_len(info["max_ctx"], info["size_gb"], tp, gpus)

        if backend == "llama.cpp" and self._state.gguf_path:
            raw_name = Path(self._state.gguf_path).stem
        else:
            raw_name = self._state.model_path.name
        phys_name = raw_name.replace("/", "-").replace(" ", "-")

        extra = preset.get("vllm_extra_args" if backend == "vllm" else "llama_extra_args", [])
        return {
            "phys_name":    phys_name,
            "tp":           str(tp),
            "max_len":      str(max_len),
            "gpu_util":     "0.90",
            "max_num_seqs": "8",
            "n_ctx":        str(min(max_len, 40960)),
            "n_threads":    "16",
            "n_gpu_layers": "-1",
            "extra_args":   json.dumps(extra),
        }

    def compose(self) -> ComposeResult:
        d       = self._defaults()
        backend = self._state.backend
        info    = self._state.info
        gpus    = self._state.gpus

        gpu_hint = (
            "  ·  ".join(f"GPU{g['index']} {g['free_gb']:.0f}GB livres" for g in gpus)
            if gpus else "Nenhuma GPU detectada"
        )

        with Container(classes="shadow-wrap"):
            with Container(id="main-window"):
                with Horizontal(id="wizard-layout"):
                    with Vertical(id="step-nav"):
                        yield Static(_nav_panel(3), id="nav-text")
                    with Vertical(id="step-content"):
                        with ScrollableContainer(id="wizard-container"):
                            yield _section("Como o modelo físico será carregado?")

                            with Vertical(classes="wizard-field"):
                                yield Label("  Nome do modelo físico:")
                                yield Input(value=d["phys_name"], id="phys-name")
                                yield _hint(
                                    "Nome interno no Allma.\n"
                                    "  Vira o arquivo: configs/base/<nome>.allm"
                                )

                            if backend == "vllm":
                                with Horizontal(classes="wizard-row-2col"):
                                    with Vertical(classes="col-half"):
                                        yield Label(f"  Tensor parallel  (sugerido: {d['tp']}):")
                                        yield Input(value=d["tp"], id="tp")
                                        yield _hint("GPUs para dividir o modelo.  1 = 1 GPU.")
                                    with Vertical(classes="col-half"):
                                        yield Label("  Comprimento máximo de contexto:")
                                        yield Input(value=d["max_len"], id="max-len")
                                        max_ctx = info.get("max_ctx")
                                        yield _hint(f"Máximo: {max_ctx:,} tokens." if max_ctx else "Tokens máximos por vez.")
                                with Horizontal(classes="wizard-row-2col"):
                                    with Vertical(classes="col-half"):
                                        yield Label("  Utilização de VRAM  (0.0–1.0):")
                                        yield Input(value=d["gpu_util"], id="gpu-util")
                                        yield _hint("0.90 = 90% da VRAM — default seguro.")
                                    with Vertical(classes="col-half"):
                                        yield Label("  Máx. requisições simultâneas:")
                                        yield Input(value=d["max_num_seqs"], id="max-num-seqs")
                                        yield _hint("Throughput vs uso de VRAM.")
                            else:
                                with Horizontal(classes="wizard-row-2col"):
                                    with Vertical(classes="col-half"):
                                        yield Label("  Janela de contexto (tokens):")
                                        yield Input(value=d["n_ctx"], id="n-ctx")
                                        max_ctx = info.get("max_ctx")
                                        yield _hint(f"Suporta até {max_ctx:,} tokens." if max_ctx else "Comprimento máximo de conversa.")
                                    with Vertical(classes="col-half"):
                                        yield Label("  Threads de CPU:")
                                        yield Input(value=d["n_threads"], id="n-threads")
                                        yield _hint("Threads usadas para inferência.")
                                with Vertical(classes="wizard-field"):
                                    yield Label("  Camadas na GPU  (-1 = todas):")
                                    yield Input(value=d["n_gpu_layers"], id="n-gpu-layers")
                                    yield _hint("-1 = todas as camadas na GPU.  0 = somente CPU.")

                            with Vertical(classes="wizard-field"):
                                yield Label("  Argumentos extras (lista JSON):")
                                yield Input(value=d["extra_args"], id="extra-args")
                                yield _hint(
                                    "Flags avançadas para o backend.\n"
                                    "  Pré-definidas para esta família de modelos.\n"
                                    '  Formato: ["--flag", "valor", ...]'
                                )

                            yield Static(f"\n  [#6a5a48]Hardware detectado: {gpu_hint}[/]")

                        with Container(id="tip-box"):
                            yield Static(
                                "  Uma boa configuração física melhora o desempenho e a estabilidade.",
                                id="tip-text",
                            )
                        with Horizontal(id="wizard-nav"):
                            yield Button("◄ VOLTAR", id="back-btn")
                            yield Button("PRÓXIMO  ►", id="next-btn", variant="primary")
                yield Static("<ESC: Voltar>", id="fkey-bar")

    def on_mount(self) -> None:
        self.query_one("#main-window").border_title = "[ ADICIONAR MODELO — PASSO 3/5 ]"
        _setup_tip(self)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.action_back()
        elif event.button.id == "next-btn":
            self._do_next()

    def _do_next(self) -> None:
        def val(wid: str, fallback: str = "") -> str:
            try:
                return self.query_one(f"#{wid}", Input).value.strip() or fallback
            except Exception:
                return fallback

        state = deepcopy(self._state)
        state.phys_name = val("phys-name") or self._state.model_path.name

        raw_args = val("extra-args")
        try:
            parsed = json.loads(raw_args) if raw_args else []
            state.extra_args = parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            state.extra_args = []

        if state.backend == "vllm":
            try:   state.tp      = int(val("tp", "1"))
            except ValueError: state.tp = 1
            try:   state.max_len = int(val("max-len", "131072"))
            except ValueError: state.max_len = 131072
            state.gpu_util     = val("gpu-util",      "0.90")
            state.max_num_seqs = val("max-num-seqs",  "8")
        else:
            try:   state.n_ctx  = int(val("n-ctx", "40960"))
            except ValueError: state.n_ctx = 40960
            state.n_threads    = val("n-threads",    "16")
            state.n_gpu_layers = val("n-gpu-layers", "-1")

        self.app.push_screen(WizardStep4Screen(state))

    def action_back(self) -> None:
        self.app.pop_screen()


# ─── Step 4: Perfis Lógicos ───────────────────────────────────────────────────
class WizardStep4Screen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back", show=False),
        Binding("f10",    "back", "Back"),
    ]

    def __init__(self, state: WizardState, **kwargs):
        super().__init__(**kwargs)
        self._state    = state
        self._variants = self._build_variants()

    def _build_variants(self) -> list:
        info   = self._state.info
        preset = FAMILY_PRESETS[info["family"]]
        phys   = self._state.phys_name
        m    = re.search(r"-(\d+\.?\d*[bBmM])", phys)
        base = phys[:m.start()] + ":" + phys[m.start() + 1:] if m else phys

        variants = []
        for key, override in preset.get("profile_variants", {"default": {}}).items():
            sampling = {**preset["sampling"], **override}
            name     = base if key == "default" else f"{base}-{key}"
            variants.append({"key": key, "name": name, "sampling": sampling})
        return variants

    def compose(self) -> ComposeResult:
        with Container(classes="shadow-wrap"):
            with Container(id="main-window"):
                with Horizontal(id="wizard-layout"):
                    with Vertical(id="step-nav"):
                        yield Static(_nav_panel(4), id="nav-text")
                    with Vertical(id="step-content"):
                        with ScrollableContainer(id="wizard-container"):
                            yield _section("Perfis de uso sobre o modelo físico")
                            yield _hint(
                                "Cada perfil é uma camada de sampling sobre o mesmo modelo.\n"
                                "  Troque entre perfis sem recarregar.  Temp: 0 = preciso  ·  1 = criativo."
                            )

                            for i, v in enumerate(self._variants):
                                s = v["sampling"]
                                yield Static("")
                                yield _section(f"Perfil: {v['key'].upper()}")
                                with Horizontal(classes="wizard-row-2col"):
                                    with Vertical(classes="col-half"):
                                        yield Label("  Nome do perfil:")
                                        yield Input(value=v["name"], id=f"prof-{i}-name")
                                        yield _hint("Nome exibido na API e no OpenWebUI.")
                                    with Vertical(classes="col-half"):
                                        with Horizontal(classes="sampling-row-mini"):
                                            with Vertical(classes="sampling-mini"):
                                                yield Label("  Temp:")
                                                yield Input(value=s.get("temperature", "0.7"), id=f"prof-{i}-temp")
                                            with Vertical(classes="sampling-mini"):
                                                yield Label("  top_p:")
                                                yield Input(value=s.get("top_p", "0.9"), id=f"prof-{i}-topp")
                                            with Vertical(classes="sampling-mini"):
                                                yield Label("  top_k:")
                                                yield Input(value=s.get("top_k", "40"), id=f"prof-{i}-topk")
                                            with Vertical(classes="sampling-mini"):
                                                yield Label("  min_p:")
                                                yield Input(value=s.get("min_p", "0.0"), id=f"prof-{i}-minp")
                                if "presence_penalty" in s:
                                    with Vertical(classes="wizard-field"):
                                        yield Label("  Presence penalty:")
                                        yield Input(value=s.get("presence_penalty", "0.0"), id=f"prof-{i}-presence")
                                        yield _hint("Desincentiva repetição de tópicos.  0.0 = desligado.")

                        with Container(id="tip-box"):
                            yield Static(
                                "  Ajuste os parâmetros para criatividade (alta temp) ou precisão (baixa temp).",
                                id="tip-text",
                            )
                        with Horizontal(id="wizard-nav"):
                            yield Button("◄ VOLTAR", id="back-btn")
                            yield Button("REVISAR  ►", id="next-btn", variant="primary")
                yield Static("<ESC: Voltar>", id="fkey-bar")

    def on_mount(self) -> None:
        self.query_one("#main-window").border_title = "[ ADICIONAR MODELO — PASSO 4/5 ]"
        _setup_tip(self)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.action_back()
        elif event.button.id == "next-btn":
            self._do_next()

    def _do_next(self) -> None:
        profiles = []
        for i, v in enumerate(self._variants):
            def _v(suffix: str, fb: str = "0.0", _i: int = i) -> str:
                try:
                    return self.query_one(f"#prof-{_i}-{suffix}", Input).value.strip() or fb
                except Exception:
                    return fb

            sampling: dict = {
                "temperature": _v("temp", "0.7"),
                "top_p":       _v("topp", "0.9"),
                "top_k":       _v("topk", "40"),
                "min_p":       _v("minp", "0.0"),
            }
            if "presence_penalty" in v["sampling"]:
                sampling["presence_penalty"] = _v("presence", "0.0")

            try:
                name = self.query_one(f"#prof-{i}-name", Input).value.strip() or v["name"]
            except Exception:
                name = v["name"]
            profiles.append({"name": name, "sampling": sampling})

        state = deepcopy(self._state)
        state.profiles = profiles
        self.app.push_screen(WizardStep5Screen(state))

    def action_back(self) -> None:
        self.app.pop_screen()


# ─── Step 5: Revisão & Salvar ─────────────────────────────────────────────────
class WizardStep5Screen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Back", show=False),
        Binding("f10",    "back", "Back"),
    ]

    def __init__(self, state: WizardState, **kwargs):
        super().__init__(**kwargs)
        self._state = state
        self._saved = False

    def _phys_content(self) -> str:
        s      = self._state
        info   = s.info
        preset = dict(FAMILY_PRESETS[info["family"]])
        if s.backend == "vllm":
            preset["vllm_extra_args"] = s.extra_args
        else:
            preset["llama_extra_args"] = s.extra_args

        content = generate_base_allm(
            name        = s.phys_name,
            info        = {**info, "backend": s.backend},
            preset      = preset,
            tp          = s.tp,
            max_len     = s.max_len if s.backend == "vllm" else s.n_ctx,
            gguf_path   = s.gguf_path   or None,
            mmproj_path = s.mmproj_path or None,
        )
        if s.backend == "vllm":
            content = content.replace('gpu_memory_utilization = "0.90"',
                                      f'gpu_memory_utilization = "{s.gpu_util}"')
            content = content.replace('max_num_seqs = "8"',
                                      f'max_num_seqs = "{s.max_num_seqs}"')
        else:
            content = content.replace('n_threads = "16"',
                                      f'n_threads = "{s.n_threads}"')
            content = content.replace('n_gpu_layers = "-1"',
                                      f'n_gpu_layers = "{s.n_gpu_layers}"')
        return content

    def _preview_text(self) -> str:
        s     = self._state
        phys  = self._phys_content()
        lines = [
            "",
            "  ══ FÍSICO ════════════════════════════════════════",
            f"  configs/base/{s.phys_name}.allm",
            "  ──────────────────────────────────────────────────",
        ]
        for line in phys.splitlines():
            lines.append(f"  {line}")
        lines.append("")
        for p in s.profiles:
            lc = generate_profile_allm(p["name"], s.phys_name, p["sampling"])
            lines.append(f"  ══ LÓGICO: {p['name']} ══════════════════════════")
            lines.append(f"  configs/profile/{p['name'].replace(':', '-')}.allm")
            lines.append("  ──────────────────────────────────────────────────")
            for line in lc.splitlines():
                lines.append(f"  {line}")
            lines.append("")
        return "\n".join(lines)

    def compose(self) -> ComposeResult:
        with Container(classes="shadow-wrap"):
            with Container(id="main-window"):
                with Horizontal(id="wizard-layout"):
                    with Vertical(id="step-nav"):
                        yield Static(_nav_panel(5), id="nav-text")
                    with Vertical(id="step-content"):
                        with ScrollableContainer(id="wizard-container"):
                            yield _section("Verifique os arquivos que serão criados")
                            yield _hint(
                                "Após salvar, reinicie o Allma para carregar o novo modelo."
                            )
                            yield Static(self._preview_text(), id="preview-text", classes="wizard-preview")
                            yield Static("", id="save-status")
                        with Container(id="tip-box"):
                            yield Static(
                                "  Este modelo aparecerá na lista principal após salvar e reiniciar o Allma.",
                                id="tip-text",
                            )
                        with Horizontal(id="wizard-nav"):
                            yield Button("◄ VOLTAR", id="back-btn")
                            yield Button("SALVAR & ATIVAR  ✔", id="save-btn", variant="primary")
                yield Static("<ESC: Voltar>", id="fkey-bar")

    def on_mount(self) -> None:
        self.query_one("#main-window").border_title = "[ ADICIONAR MODELO — PASSO 5/5 ]"
        _setup_tip(self)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.action_back()
        elif event.button.id == "save-btn" and not self._saved:
            self._do_save()

    def _do_save(self) -> None:
        s        = self._state
        status   = self.query_one("#save-status", Static)
        phys_dir = CONFIG_DIR / "base"
        log_dir  = CONFIG_DIR / "profile"
        phys_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            phys_file = phys_dir / f"{s.phys_name}.allm"
            phys_file.write_text(self._phys_content())
            saved = [str(phys_file)]

            for p in s.profiles:
                content  = generate_profile_allm(p["name"], s.phys_name, p["sampling"])
                log_file = log_dir / f"{p['name'].replace(':', '-')}.allm"
                log_file.write_text(content)
                saved.append(str(log_file))

            self._saved = True
            msg = "  [#007878]✔ Arquivos salvos com sucesso:[/#007878]\n"
            msg += "\n".join(f"    {f}" for f in saved)
            msg += "\n\n  [#007878]Reinicie o Allma para carregar o novo modelo.[/#007878]"
            status.update(msg)
            btn = self.query_one("#save-btn", Button)
            btn.label    = "SALVO  ✔"
            btn.disabled = True
        except Exception as e:
            status.update(f"  [red]Erro ao salvar:[/red] {e}")

    def action_back(self) -> None:
        self.app.pop_screen()
