"""
FastAPI application, route handlers, HTTP client, and banner display.
"""
import httpx
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from core.config import (
    logger,
    LOGICAL_MODELS,
    PHYSICAL_MODELS,
    MAX_MESSAGES,
    KEEP_ALIVE_SECONDS,
    ALLAMA_PORT,
    RICH_AVAILABLE,
    format_user_agent,
)
import core.state as state
from core.loader import ensure_physical_model
from core.process import shutdown_server

# ==============================================================================
# HTTP CLIENT
# ==============================================================================
_httpx_client: httpx.AsyncClient | None = None


async def get_http_client() -> httpx.AsyncClient:
    global _httpx_client
    if _httpx_client is None or _httpx_client.is_closed:
        _httpx_client = httpx.AsyncClient(
            timeout=600.0,
            headers={"Authorization": "Bearer dummy"},
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
    return _httpx_client


async def close_http_client():
    global _httpx_client
    if _httpx_client is not None and not _httpx_client.is_closed:
        await _httpx_client.aclose()
        _httpx_client = None


# ==============================================================================
# FASTAPI APPLICATION
# ==============================================================================
async def lifespan(app: FastAPI):
    yield
    logger.info("🛑 Shutting down Allama...")
    with state.global_lock:
        names = list(state.active_servers.keys())
    for name in names:
        shutdown_server(name, reason="shutdown", fast=True)
    await close_http_client()


app = FastAPI(title="Allama - LLM API", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: dict = Body(...)):
    model_name = body.get("model", "")
    client_host = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    logger.info(
        f"📤 [HTTP] {request.method} {request.url.path} from {client_host} (🖥️  {format_user_agent(user_agent)})"
    )

    if model_name not in LOGICAL_MODELS:
        return JSONResponse(
            status_code=404,
            content={"error": f"Model '{model_name}' not found"},
        )

    if "messages" in body and MAX_MESSAGES > 0:
        body["messages"] = body["messages"][-MAX_MESSAGES:]

    logical_cfg = LOGICAL_MODELS[model_name]
    physical_name = logical_cfg["physical"]
    port = await ensure_physical_model(physical_name, model_name)

    cfg = PHYSICAL_MODELS[physical_name]
    backend = cfg.get("backend", "vllm")
    if backend == "vllm":
        body["model"] = cfg["path"]
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
    else:
        body["model"] = cfg["model"]
        url = f"http://127.0.0.1:{port}/chat/completions"

    sampling = logical_cfg.get("sampling", {})
    for key in ["temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"]:
        if key in sampling:
            if key not in body or body[key] is None:
                body[key] = sampling[key]
        else:
            body.pop(key, None)

    # Disable thinking for Instruct models or when explicitly set in config
    if logical_cfg.get("enable_thinking") is False or "instruct" in model_name.lower():
        body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False

    logger.debug(f"{model_name} -> {physical_name}:{port} ({backend})")

    if "messages" not in body or not body["messages"]:
        logger.warning("⚠️  Empty messages request")
        if body.get("stream", False):
            async def empty_stream():
                yield "data: [DONE]\n\n"
            return StreamingResponse(empty_stream(), media_type="text/event-stream")
        return JSONResponse(
            status_code=200,
            content={"choices": [{"message": {"content": ""}}]},
        )

    client = await get_http_client()
    try:
        if body.get("stream", False):
            async def generate():
                try:
                    async with client.stream("POST", url, json=body) as response:
                        if response.status_code != 200:
                            logger.error(f"Backend returned {response.status_code}")
                            yield 'data: {"error": "Backend error"}\n\n'
                            return
                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line:
                                continue
                            if line.startswith("data: "):
                                yield line + "\n\n"
                            elif line == "[DONE]":
                                yield "data: [DONE]\n\n"
                                break
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            resp = await client.post(url, json=body)
            if resp.status_code == 400:
                error_body = await resp.aread()
                error_detail = (
                    error_body.decode() if isinstance(error_body, bytes) else str(error_body)
                )
                logger.error(f"vLLM 400 Error for {model_name}: {error_detail}")
                logger.warning("vLLM 400 - returning empty response")
                return JSONResponse(
                    status_code=200,
                    content={"choices": [{"message": {"content": ""}}]},
                )
            logger.debug("Output sent")
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except httpx.ConnectError as e:
        logger.warning(f"Connection failed: {e}")
        return JSONResponse(status_code=503, content={"error": "Model server unavailable"})
    except Exception as e:
        logger.error(f"Request error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/v1/messages")
async def messages(request: Request, body: dict = Body(...)):
    model_name = body.get("model", "")
    client_host = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    logger.info(f"📤 [HTTP] {request.method} {request.url.path} from {client_host} (🖥️  {format_user_agent(user_agent)})")

    if model_name not in LOGICAL_MODELS:
        with state.global_lock:
            loaded_vllm = [
                name
                for name, srv in state.active_servers.items()
                if srv.get("backend") == "vllm" and srv.get("process") and srv["process"].poll() is None
            ]
        if loaded_vllm:
            physical = loaded_vllm[0]
            model_name = next(
                (k for k, v in LOGICAL_MODELS.items() if v["physical"] == physical),
                None,
            )
            if model_name and model_name != body.get("model"):
                logger.info(f"🔄 Auto-switch: {body.get('model')} -> {model_name} (using loaded {physical})")
        if not model_name or model_name not in LOGICAL_MODELS:
            return JSONResponse(
                status_code=404,
                content={"error": f"Model '{body.get('model')}' not found"},
            )

    logical_cfg = LOGICAL_MODELS[model_name]
    physical_name = logical_cfg["physical"]
    cfg = PHYSICAL_MODELS[physical_name]

    if cfg.get("backend", "vllm") != "vllm":
        return JSONResponse(
            status_code=400,
            content={"error": "Anthropic Messages API requires vllm backend"},
        )

    current_loaded = None
    with state.global_lock:
        for name, srv in state.active_servers.items():
            if srv.get("backend") == "vllm" and srv.get("process") and srv["process"].poll() is None:
                current_loaded = name
                break

    if current_loaded and current_loaded != physical_name:
        logger.info(f"🔄 Model switch: {current_loaded} -> {physical_name} ({model_name})")

    port = await ensure_physical_model(physical_name, model_name)

    max_model_len = int(cfg.get("max_model_len", "40960"))
    max_output_cap = max_model_len // 4
    requested = body.get("max_tokens", max_output_cap)
    if requested > max_output_cap:
        logger.info(f"⚠️  max_tokens {requested} -> {max_output_cap}")
        body["max_tokens"] = max_output_cap

    body["model"] = cfg["path"]
    url = f"http://127.0.0.1:{port}/v1/messages"

    logger.debug(f"{model_name} -> {physical_name}:{port}")

    client = await get_http_client()
    try:
        if body.get("stream", False):
            async def generate():
                try:
                    async with client.stream("POST", url, json=body) as response:
                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line:
                                continue
                            yield line + "\n\n"
                except Exception as e:
                    logger.error(f"Stream error: {e}")

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            resp = await client.post(url, json=body)
            return JSONResponse(content=resp.json(), status_code=resp.status_code)

    except httpx.ConnectError as e:
        logger.warning(f"Connection failed: {e}")
        return JSONResponse(status_code=503, content={"error": "Backend unavailable"})
    except Exception as e:
        logger.error(f"Claude Code error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/v1/models")
async def models_list():
    return {
        "object": "list",
        "data": [{"id": k, "object": "model"} for k in LOGICAL_MODELS],
    }


@app.get("/health")
async def health():
    with state.global_lock:
        active = len(state.active_servers)
    return {
        "status": "healthy",
        "active_servers": active,
        "running": state.running,
    }


# ==============================================================================
# BANNER
# ==============================================================================
def show_banner():
    if not RICH_AVAILABLE:
        logger.info("=" * 60)
        logger.info("Allama Started")
        logger.info("=" * 60)
        for name in PHYSICAL_MODELS:
            from core.config import ALLAMA_LOG_DIR
            logger.info(f"   - {name} - tail -f {ALLAMA_LOG_DIR}/{name}.log")
        logger.info("=" * 60)
        logger.info(f"Models configured: {len(PHYSICAL_MODELS)} physical, {len(LOGICAL_MODELS)} logical")
        logger.info(f"Keep-alive: {KEEP_ALIVE_SECONDS}s")
        logger.info(f"API: http://127.0.0.1:{ALLAMA_PORT}")
        logger.info("=" * 60)
        return

    from rich import box as _box
    from rich.align import Align
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text

    W      = 94
    C_RED  = "#ff4444"
    C_ORG  = "#ff8c00"
    C_GOLD = "#ffd700"
    C_CYAN = "#00e5ff"
    C_PURP = "#b388ff"
    C_WHT  = "#ffffff"
    C_DIM  = "#546e7a"
    C_SHD  = "#2a2a2a"

    console = Console(width=W)
    SHADOW  = Text("  " + "▄" * (W - 3), style=C_SHD)

    def make_table(panels, cols=3):
        tbl = Table.grid(expand=True, padding=0)
        for _ in range(cols):
            tbl.add_column(ratio=1)
        row = []
        for p in panels:
            row.append(p)
            if len(row) == cols:
                tbl.add_row(*row)
                row = []
        if row:
            row += [Text("")] * (cols - len(row))
            tbl.add_row(*row)
        return tbl

    def mac_panel(header, body, *, border_color, rule_color, expand=True):
        grid = Table.grid(expand=True, padding=0)
        grid.add_column()
        grid.add_row(Rule(title=f"[bold {C_WHT}]{header}[/]", style=rule_color, characters="≡"))
        grid.add_row(body)
        return Panel(grid, box=_box.HEAVY, border_style=border_color, padding=(0, 1), expand=expand)

    def build_logo() -> list[str]:
        CHARS = {
            'A': [" ███ ", "█   █", "█████", "█   █", "█   █"],
            'L': ["█    ", "█    ", "█    ", "█    ", "█████"],
            'M': ["█   █", "██ ██", "█ █ █", "█   █", "█   █"],
        }
        word   = list("ALLAMA")
        starts = [0, 6, 12, 18, 24, 30]
        height = 5
        width  = 36
        canvas = [[0] * width for _ in range(height)]
        for ch, col in zip(word, starts):
            for r, row in enumerate(CHARS[ch]):
                for c, px in enumerate(row):
                    if px == "█":
                        canvas[r][col + c] = 1
            if ch == "L":
                for r, row in enumerate(CHARS[ch]):
                    for c, px in enumerate(row):
                        if px == "█":
                            sc = col + c + 1
                            if sc < width and canvas[r][sc] == 0:
                                canvas[r][sc] = 2
        return [
            "".join("█" if v == 1 else "▒" if v == 2 else " " for v in row)
            for row in canvas
        ]

    LOGO_COLORS = [C_RED, C_ORG, C_GOLD, C_CYAN, C_PURP]
    logo_rows = build_logo()
    banner = Text(justify="center")
    for i, row in enumerate(logo_rows):
        color = LOGO_COLORS[i % len(LOGO_COLORS)]
        for ch in row:
            if ch == "█":
                banner.append(ch, style=f"bold {color}")
            elif ch == "▒":
                banner.append(ch, style=f"dim {color}")
            else:
                banner.append(ch)
        if i < len(logo_rows) - 1:
            banner.append("\n")
    banner.append(f"\n\nhttp://127.0.0.1:{ALLAMA_PORT}", style=C_DIM)

    console.print(Panel(Align(banner, align="center"), box=_box.HEAVY,
                        border_style=C_GOLD, padding=(1, 4), width=W))
    console.print(SHADOW)

    console.rule(f"[bold {C_GOLD}] CONFIGURATION [/]", style=C_DIM, characters="≡")
    cfg_panels = [
        mac_panel("PHYSICAL MODELS",
                  Align(Text(str(len(PHYSICAL_MODELS)), style=f"bold {C_WHT}", justify="center"), align="center"),
                  border_color=C_GOLD, rule_color=C_GOLD),
        mac_panel("LOGICAL MODELS",
                  Align(Text(str(len(LOGICAL_MODELS)), style=f"bold {C_WHT}", justify="center"), align="center"),
                  border_color=C_GOLD, rule_color=C_GOLD),
        mac_panel("KEEP ALIVE",
                  Align(Text(f"{KEEP_ALIVE_SECONDS}s", style=f"bold {C_WHT}", justify="center"), align="center"),
                  border_color=C_GOLD, rule_color=C_GOLD),
    ]
    console.print(make_table(cfg_panels, cols=3))
    console.print(SHADOW)

    console.rule(f"[bold {C_CYAN}] PHYSICAL MODELS [/]", style=C_DIM, characters="≡")
    vllm_panels  = []
    llama_panels = []
    for name, cfg in PHYSICAL_MODELS.items():
        backend = cfg.get("backend", "vllm")
        body = Align(Text(name, style=f"bold {C_WHT}", justify="center"), align="center", vertical="top")
        if backend == "vllm":
            vllm_panels.append(mac_panel("vLLM", body, border_color=C_CYAN, rule_color=C_CYAN))
        else:
            llama_panels.append(mac_panel("llama.cpp", body, border_color=C_ORG, rule_color=C_ORG))
    if vllm_panels:
        console.print(make_table(vllm_panels, cols=3))
    if vllm_panels and llama_panels:
        console.print(Text("─" * W, style=C_DIM))
    if llama_panels:
        console.print(make_table(llama_panels, cols=3))
    console.print(SHADOW)

    console.rule(f"[bold {C_PURP}] LOGICAL MODELS [/]", style=C_DIM, characters="≡")
    grouped: dict = {}
    for log_name, log_cfg in LOGICAL_MODELS.items():
        phys = log_cfg["physical"]
        grouped.setdefault(phys, []).append(log_name)

    log_panels = []
    for phys, names in grouped.items():
        body = Text()
        for i, n in enumerate(names):
            body.append(f"  {n}", style=C_WHT)
            if i < len(names) - 1:
                body.append("\n")
        log_panels.append(mac_panel(phys, body, border_color=C_PURP, rule_color=C_PURP))

    max_name_len = max(
        (len(f"  {n}") for names in grouped.values() for n in names),
        default=20,
    )
    log_cols = 2 if max_name_len > (W // 3 - 4) else 3
    console.print(make_table(log_panels, cols=log_cols))
    console.print(SHADOW)
    console.print()
