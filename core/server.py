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
            timeout=httpx.Timeout(30.0, read=600.0),
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
                            error_body = await response.aread()
                            error_detail = error_body.decode() if isinstance(error_body, bytes) else str(error_body)
                            logger.error(f"Backend returned {response.status_code} for {model_name}: {error_detail}")
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
            loaded = [
                name
                for name, srv in state.active_servers.items()
                if srv.get("process") and srv["process"].poll() is None
            ]
        if loaded:
            physical = loaded[0]
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
    backend = cfg.get("backend", "vllm")

    port = await ensure_physical_model(physical_name, model_name)

    max_model_len = int(cfg.get("max_model_len") or cfg.get("n_ctx") or "40960")
    max_output_cap = max_model_len // 4
    requested = body.get("max_tokens", max_output_cap)
    if requested > max_output_cap:
        logger.info(f"⚠️  max_tokens {requested} -> {max_output_cap}")
        body["max_tokens"] = max_output_cap

    logger.debug(f"{model_name} -> {physical_name}:{port} ({backend})")

    client = await get_http_client()

    # vLLM supports /v1/messages natively — pass through
    if backend == "vllm":
        body["model"] = cfg["path"]
        url = f"http://127.0.0.1:{port}/v1/messages"
        try:
            if body.get("stream", False):
                async def generate_vllm():
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
                    generate_vllm(),
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

    # llama.cpp — translate Anthropic Messages ↔ OpenAI Chat format
    import json as _json
    import uuid as _uuid

    # ── Convert tool definitions (Anthropic → OpenAI) ──
    oai_tools = None
    if body.get("tools"):
        oai_tools = []
        for tool in body["tools"]:
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            })

    # ── Convert messages ──
    oai_messages = []
    system_raw = body.get("system", "")
    if system_raw:
        if isinstance(system_raw, list):
            system_text = "".join(
                block.get("text", "") for block in system_raw if block.get("type") == "text"
            )
        else:
            system_text = system_raw
        if system_text:
            oai_messages.append({"role": "system", "content": system_text})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            oai_messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            oai_messages.append({"role": role, "content": str(content)})
            continue

        # Separate content block types
        text_parts, tool_uses, tool_results = [], [], []
        for block in content:
            btype = block.get("type", "")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_uses.append(block)
            elif btype == "tool_result":
                tool_results.append(block)

        if role == "assistant":
            oai_msg = {"role": "assistant"}
            oai_msg["content"] = "".join(text_parts) if text_parts else None
            if tool_uses:
                oai_msg["tool_calls"] = [
                    {
                        "id": tu["id"],
                        "type": "function",
                        "function": {
                            "name": tu["name"],
                            "arguments": _json.dumps(tu.get("input", {})),
                        },
                    }
                    for tu in tool_uses
                ]
            oai_messages.append(oai_msg)
        else:
            # User messages: text first, then tool_results as separate "tool" role messages
            if text_parts:
                oai_messages.append({"role": "user", "content": "".join(text_parts)})
            for tr in tool_results:
                tr_content = tr.get("content", "")
                if isinstance(tr_content, list):
                    tr_content = "".join(
                        b.get("text", "") for b in tr_content if b.get("type") == "text"
                    )
                oai_messages.append({
                    "role": "tool",
                    "tool_call_id": tr.get("tool_use_id", ""),
                    "content": str(tr_content),
                })

    oai_body = {
        "model": cfg["model"],
        "messages": oai_messages,
        "max_tokens": body.get("max_tokens", max_output_cap),
        "stream": body.get("stream", False),
    }
    if "temperature" in body:
        oai_body["temperature"] = body["temperature"]
    if oai_tools:
        oai_body["tools"] = oai_tools

    url = f"http://127.0.0.1:{port}/chat/completions"

    try:
        if oai_body["stream"]:
            msg_id = f"msg_{_uuid.uuid4().hex[:24]}"

            async def generate_llama():
                try:
                    # Anthropic SSE preamble
                    yield f"event: message_start\ndata: {_json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model_name, 'stop_reason': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

                    block_idx = 0
                    text_started = False
                    tool_calls_acc = {}  # tc_index -> {id, name, arguments, block_idx}
                    stop_reason = "end_turn"

                    async with client.stream("POST", url, json=oai_body) as response:
                        if response.status_code != 200:
                            error_body = await response.aread()
                            logger.error(f"Backend returned {response.status_code}: {error_body.decode()}")
                            yield f"event: content_block_start\ndata: {_json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                            yield f"event: content_block_stop\ndata: {_json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            return

                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line or not line.startswith("data: "):
                                continue
                            raw = line[6:]
                            if raw == "[DONE]":
                                break
                            try:
                                chunk = _json.loads(raw)
                                delta = chunk["choices"][0].get("delta", {})

                                # ── Text content ──
                                delta_text = delta.get("content", "")
                                if delta_text:
                                    if not text_started:
                                        yield f"event: content_block_start\ndata: {_json.dumps({'type': 'content_block_start', 'index': block_idx, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                        text_started = True
                                    yield f"event: content_block_delta\ndata: {_json.dumps({'type': 'content_block_delta', 'index': block_idx, 'delta': {'type': 'text_delta', 'text': delta_text}})}\n\n"

                                # ── Tool calls ──
                                for tc_delta in delta.get("tool_calls", []):
                                    tc_idx = tc_delta.get("index", 0)
                                    if tc_idx not in tool_calls_acc:
                                        # New tool call — close text block if open
                                        if text_started:
                                            yield f"event: content_block_stop\ndata: {_json.dumps({'type': 'content_block_stop', 'index': block_idx})}\n\n"
                                            block_idx += 1
                                            text_started = False

                                        tc_id = tc_delta.get("id", f"toolu_{_uuid.uuid4().hex[:24]}")
                                        tc_name = tc_delta.get("function", {}).get("name", "")
                                        tool_calls_acc[tc_idx] = {"id": tc_id, "name": tc_name, "block_idx": block_idx}
                                        stop_reason = "tool_use"

                                        yield f"event: content_block_start\ndata: {_json.dumps({'type': 'content_block_start', 'index': block_idx, 'content_block': {'type': 'tool_use', 'id': tc_id, 'name': tc_name, 'input': {}}})}\n\n"

                                    args_delta = tc_delta.get("function", {}).get("arguments", "")
                                    if args_delta:
                                        yield f"event: content_block_delta\ndata: {_json.dumps({'type': 'content_block_delta', 'index': tool_calls_acc[tc_idx]['block_idx'], 'delta': {'type': 'input_json_delta', 'partial_json': args_delta}})}\n\n"

                            except Exception:
                                pass

                except Exception as e:
                    logger.error(f"llama.cpp stream error: {e}")
                finally:
                    # Close any open content blocks
                    if text_started:
                        yield f"event: content_block_stop\ndata: {_json.dumps({'type': 'content_block_stop', 'index': block_idx})}\n\n"
                    for tc_info in tool_calls_acc.values():
                        yield f"event: content_block_stop\ndata: {_json.dumps({'type': 'content_block_stop', 'index': tc_info['block_idx']})}\n\n"

                    yield f"event: message_delta\ndata: {_json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
                    yield f"event: message_stop\ndata: {_json.dumps({'type': 'message_stop'})}\n\n"

            return StreamingResponse(
                generate_llama(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            resp = await client.post(url, json=oai_body)
            oai = resp.json()
            choice = oai["choices"][0]
            message = choice.get("message", {})
            msg_id = f"msg_{_uuid.uuid4().hex[:24]}"

            content_blocks = []
            if message.get("content"):
                content_blocks.append({"type": "text", "text": message["content"]})

            stop_reason = "end_turn"
            if message.get("tool_calls"):
                stop_reason = "tool_use"
                for tc in message["tool_calls"]:
                    try:
                        input_data = _json.loads(tc["function"]["arguments"])
                    except (ValueError, KeyError):
                        input_data = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", f"toolu_{_uuid.uuid4().hex[:24]}"),
                        "name": tc["function"]["name"],
                        "input": input_data,
                    })

            if not content_blocks:
                content_blocks.append({"type": "text", "text": ""})

            anthropic_resp = {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": content_blocks,
                "model": model_name,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": oai.get("usage", {}).get("prompt_tokens", 0),
                    "output_tokens": oai.get("usage", {}).get("completion_tokens", 0),
                },
            }
            return JSONResponse(content=anthropic_resp, status_code=200)

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


@app.get("/v1/ps")
async def ps():
    """Return active backend servers with their log file paths."""
    with state.global_lock:
        servers = [
            {
                "name": name,
                "backend": srv.get("backend", "unknown"),
                "port": srv.get("port"),
                "pid": srv.get("pid"),
                "logfile": str(srv.get("logfile", "")),
                "alive": srv["process"].poll() is None,
            }
            for name, srv in state.active_servers.items()
        ]
    return {"servers": servers}


@app.post("/v1/shutdown")
async def shutdown_endpoint():
    """Gracefully shut down the server and all backends."""
    import asyncio
    import os
    import signal as _signal

    async def _do_shutdown():
        await asyncio.sleep(0.1)
        os.kill(os.getpid(), _signal.SIGTERM)

    asyncio.create_task(_do_shutdown())
    return JSONResponse({"status": "shutting down"})


@app.post("/v1/load")
async def load_model(body: dict = Body(...)):
    """Pre-load a model without generating any tokens."""
    model_name = body.get("model", "")
    if model_name not in LOGICAL_MODELS:
        return JSONResponse(
            status_code=404,
            content={"error": f"Model '{model_name}' not found"},
        )
    logical_cfg = LOGICAL_MODELS[model_name]
    physical_name = logical_cfg["physical"]
    cfg = PHYSICAL_MODELS[physical_name]
    backend = cfg.get("backend", "vllm")
    await ensure_physical_model(physical_name, model_name)
    return JSONResponse({"status": "loaded", "model": model_name, "backend": backend})


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
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text

    _console_probe = Console()
    _term_w  = _console_probe.width
    W        = min(max(_term_w - 1, 40), 130)
    _narrow  = _term_w < 72   # mobile / narrow terminal
    # Logo brand colors (pixel art identity — unchanged)
    C_RED    = "#e52529"
    C_ORG    = "#f7941d"
    C_YELL   = "#f7d000"
    C_GRN    = "#43b047"
    C_BLUE   = "#009ddc"
    # Warm cream + teal palette (DOS wizard style)
    C_BG     = "#e8dfc8"   # window background (warm cream)
    C_SCREEN = "#d0c4a8"   # desktop / outer background (darker cream)
    C_FG     = "#1a1408"   # main text (near-black, warm)
    C_DIM    = "#6a5a48"   # secondary / dim (warm gray-brown)
    C_ACCENT = "#007878"   # section titles, accents (teal)
    C_BORDER = "#008888"   # panel border lines (teal)
    C_OK     = "#007878"   # READY / online (teal)

    console = Console(width=W)

    # ── helpers ──────────────────────────────────────────────────────────────
    def section(name: str) -> Text:
        """Render a [ Section Name ] label in wizard style."""
        t = Text()
        t.append("[ ", style=C_DIM)
        t.append(name, style=f"bold {C_ACCENT}")
        t.append(" ]", style=C_DIM)
        return t

    def build_logo() -> list[str]:
        CHARS = {
            'A': [" ███ ", "█   █", "█████", "█   █", "█   █"],
            'L': ["█    ", "█    ", "█    ", "█    ", "█████"],
            'M': ["█   █", "██ ██", "█ █ █", "█   █", "█   █"],
        }
        word, starts = list("ALLAMA"), [0, 6, 12, 18, 24, 30]
        canvas = [[0] * 36 for _ in range(5)]
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
                            if sc < 36 and canvas[r][sc] == 0:
                                canvas[r][sc] = 2
        return ["".join("█" if v == 1 else "▒" if v == 2 else " " for v in row)
                for row in canvas]

    # ── logo panel ────────────────────────────────────────────────────────────
    LOGO_COLORS = [C_RED, C_ORG, C_YELL, C_GRN, C_BLUE]
    logo_text   = Text(justify="center")
    for i, row in enumerate(build_logo()):
        color = LOGO_COLORS[i % len(LOGO_COLORS)]
        for ch in row:
            if ch == "█":   logo_text.append(ch, style=f"bold {color}")
            elif ch == "▒": logo_text.append(ch, style=f"dim {color}")
            else:            logo_text.append(ch)
        if i < 4:
            logo_text.append("\n")
    # ── configuration summary ─────────────────────────────────────────────────
    cfg_line = Text()
    cfg_line.append(f"  {len(PHYSICAL_MODELS)}", style=f"bold {C_FG}")
    cfg_line.append(" phys  ·  " if _narrow else " physical model(s)  ·  ", style=C_DIM)
    cfg_line.append(f"{len(LOGICAL_MODELS)}", style=f"bold {C_FG}")
    cfg_line.append(" logical  ·  keep-alive: " if _narrow else " logical model(s)  ·  keep-alive: ", style=C_DIM)
    cfg_line.append(f"{KEEP_ALIVE_SECONDS}s", style=f"bold {C_FG}")

    # ── physical models table — sorted by backend group then size desc ───────
    import re as _re

    def _param_b(model_name: str) -> float:
        m = _re.search(r'(\d+(?:\.\d+)?)\s*[bB]', model_name)
        return float(m.group(1)) if m else 0.0

    sorted_phys = sorted(
        PHYSICAL_MODELS.items(),
        key=lambda kv: (
            0 if kv[1].get("backend", "vllm") == "vllm" else 1,  # vllm first
            -_param_b(kv[0]),                                      # larger first
            kv[0].lower(),
        ),
    )

    # group by backend so we can add a separator between groups
    _phys_groups: dict[str, list] = {}
    for _n, _c in sorted_phys:
        _g = "vllm" if _c.get("backend", "vllm") == "vllm" else "gguf"
        _phys_groups.setdefault(_g, []).append((_n, _c))

    _tbl_pad  = (0, 1, 0, 0) if _narrow else (0, 2, 0, 0)
    _w_name_p = 16 if _narrow else 28
    _w_fmt    = 8  if _narrow else 12

    phys_tbl = Table(box=None, padding=_tbl_pad, show_header=True,
                     header_style=f"bold {C_DIM}", expand=True)
    phys_tbl.add_column("NAME",    style=C_FG,     no_wrap=True, min_width=_w_name_p)
    phys_tbl.add_column("FORMAT",  style=C_ACCENT, no_wrap=True, min_width=_w_fmt)
    phys_tbl.add_column("CTX",     style=C_DIM,    justify="right", min_width=6)
    phys_tbl.add_column("GPU",     style=C_DIM,    justify="right", min_width=3)

    _group_labels = {"vllm": "safetensors", "gguf": "gguf"}
    _group_list   = [g for g in ("vllm", "gguf") if g in _phys_groups]
    for _gi, _g in enumerate(_group_list):
        _items = _phys_groups[_g]
        _label = _group_labels[_g]
        _is_last_group = _gi == len(_group_list) - 1
        for _i, (_name, _cfg) in enumerate(_items):
            _ctx     = _cfg.get("n_ctx") or _cfg.get("max_model_len", "—")
            _gpu     = _cfg.get("n_gpu_layers", "—")
            _gpu_str = "all" if str(_gpu) == "-1" else str(_gpu)
            _end_sec = (_i == len(_items) - 1) and not _is_last_group
            phys_tbl.add_row(_name, _label, str(_ctx), _gpu_str,
                             end_section=_end_sec)

    # ── logical models table — grouped by physical backend, sorted by size ───
    def _log_sort_key(kv):
        _lname, _lcfg = kv
        _phys_name = _lcfg.get("physical", "")
        _phys_cfg  = PHYSICAL_MODELS.get(_phys_name, {})
        _backend   = _phys_cfg.get("backend", "vllm")
        _group     = 0 if _backend == "vllm" else 1
        return (_group, -_param_b(_phys_name), _phys_name.lower(), _lname.lower())

    sorted_log = sorted(LOGICAL_MODELS.items(), key=_log_sort_key)

    # group by physical backend for separator
    _log_groups: dict[str, list] = {}
    for _n, _c in sorted_log:
        _phys_cfg = PHYSICAL_MODELS.get(_c.get("physical", ""), {})
        _g = "vllm" if _phys_cfg.get("backend", "vllm") == "vllm" else "gguf"
        _log_groups.setdefault(_g, []).append((_n, _c))

    _w_name_l = 14 if _narrow else 24
    _w_phys   = 16 if _narrow else 28

    log_tbl = Table(box=None, padding=_tbl_pad, show_header=True,
                    header_style=f"bold {C_DIM}", expand=True)
    log_tbl.add_column("NAME",     style=C_FG,   no_wrap=True, min_width=_w_name_l)
    log_tbl.add_column("PHYSICAL", style=C_DIM,  no_wrap=True, min_width=_w_phys)
    log_tbl.add_column("TEMP",     style=C_DIM,  justify="right", min_width=4)
    log_tbl.add_column("TOP_P",    style=C_DIM,  justify="right", min_width=4)
    log_tbl.add_column("TOP_K",    style=C_DIM,  justify="right", min_width=4)

    _log_group_list = [g for g in ("vllm", "gguf") if g in _log_groups]
    for _gi, _g in enumerate(_log_group_list):
        _items = _log_groups[_g]
        _is_last_group = _gi == len(_log_group_list) - 1
        for _i, (_lname, _lcfg) in enumerate(_items):
            _sampling = _lcfg.get("sampling", {})
            _end_sec  = (_i == len(_items) - 1) and not _is_last_group
            log_tbl.add_row(
                _lname,
                _lcfg.get("physical", "—"),
                str(_sampling.get("temperature", "—")),
                str(_sampling.get("top_p",       "—")),
                str(_sampling.get("top_k",       "—")),
                end_section=_end_sec,
            )

    # ── sub-panels (DOS wizard style — bordered boxes inside the main window) ──
    _inner_pad = (0, 1) if _narrow else (0, 2)
    _sec_title = lambda name: f"[bold {C_ACCENT}][ {name} ][/]"

    _S = f"on {C_BG}"  # window background style

    logo_panel = Panel(
        Align(logo_text, align="center"),
        box=_box.SQUARE, border_style=C_BORDER,
        style=_S, padding=(1, 0),
    )

    cfg_panel = Panel(
        Align(cfg_line, align="center"),
        title=_sec_title("Configuration"), title_align="left",
        box=_box.SQUARE, border_style=C_DIM,
        style=_S, padding=_inner_pad,
    )

    phys_panel = Panel(
        phys_tbl,
        title=_sec_title("Physical Models"), title_align="left",
        box=_box.SQUARE, border_style=C_DIM,
        style=_S, padding=_inner_pad,
    )

    log_panel = Panel(
        log_tbl,
        title=_sec_title("Logical Models"), title_align="left",
        box=_box.SQUARE, border_style=C_DIM,
        style=_S, padding=_inner_pad,
    )

    # ── status bar ────────────────────────────────────────────────────────────
    status_line = Text(justify="center")
    status_line.append("● ", style=f"bold {C_OK}")
    status_line.append("READY", style=f"bold {C_OK}")
    status_line.append("  ·  ", style=C_DIM)
    if _narrow:
        status_line.append(f":{ALLAMA_PORT}", style=C_ACCENT)
    else:
        status_line.append(f"http://127.0.0.1:{ALLAMA_PORT}", style=C_ACCENT)
        status_line.append("  ·  Ctrl+C to stop", style=C_DIM)

    status_panel = Panel(
        Align(status_line, align="center"),
        box=_box.SQUARE, border_style=C_DIM,
        style=_S, padding=(0, 1),
    )

    # ── main floating window — light bg, floats on dark terminal desktop ──────
    import sys as _sys
    body     = Group(logo_panel, cfg_panel, phys_panel, log_panel, status_panel)
    main_win = Panel(body, box=_box.DOUBLE, border_style=C_BORDER,
                     style=f"on {C_SCREEN}", padding=(0, 0), width=W)

    # Capture panel with ANSI codes so we can add shadow on each line
    _cap = Console(width=W, force_terminal=True, color_system="truecolor",
                   highlight=False)
    with _cap.capture() as _capture:
        _cap.print(main_win)
    _lines = _capture.get().rstrip('\n').split('\n')

    import re as _re_ansi
    _ansi_re  = _re_ansi.compile(r'\x1b\[[0-9;]*m')
    _lpad     = " " * max(0, (_term_w - W) // 2)
    _shd_char = "\033[48;2;42;32;24m  \033[0m"   # 2 cols, warm dark = solid shadow
    _bshd     = (" " * (max(0, (_term_w - W) // 2) + 2)
                 + "\033[48;2;42;32;24m" + " " * W + "\033[0m")

    _sys.stdout.write('\n')
    for i, ln in enumerate(_lines):
        if i == 0:
            _sys.stdout.write(_lpad + ln + '\n')
        else:
            # pad to exactly W visible chars so shadow sits flush on the border
            vis = len(_ansi_re.sub('', ln))
            pad = ' ' * max(0, W - vis)
            _sys.stdout.write(_lpad + ln + pad + _shd_char + '\n')
    if not _narrow:
        _sys.stdout.write(_bshd + '\n')
    _sys.stdout.write('\n')
    _sys.stdout.flush()
