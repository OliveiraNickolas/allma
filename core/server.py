"""
FastAPI application, route handlers, HTTP client, and banner display.
"""
import asyncio
import json as _json
import re
from typing import Optional

import httpx
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from core.config import (
    logger,
    PROFILE_MODELS,
    BASE_MODELS,
    MAX_MESSAGES,
    KEEP_ALIVE_SECONDS,
    ALLMA_PORT,
    RICH_AVAILABLE,
    format_user_agent,
)
import core.state as state
from core.loader import ensure_base_model
from core.process import shutdown_server
from core.error_detector import ErrorDetector


# ==============================================================================
# TOOL SCHEMA SIMPLIFICATION (llama.cpp GBNF grammar guard)
# ==============================================================================
def _simplify_prop(schema: dict) -> dict:
    """
    Simplify a single parameter property schema for llama.cpp.

    llama.cpp auto-generates GBNF grammar from tool schemas.  Complex schemas
    (array/object types, anyOf with complex variants, numeric range constraints)
    produce large recursive grammar rules — for 30+ tools the grammar can
    exceed the parser's capacity, causing "failed to parse grammar" on every
    request and falling back to unreliable unconstrained generation.

    This function flattens complex property types to plain "string" so the
    resulting grammar is small and non-recursive.
    """
    if not isinstance(schema, dict):
        return schema

    # Resolve anyOf / oneOf → pick the first non-null variant and recurse.
    # Handles the common pattern: anyOf: [{type: "array", …}, {type: "null"}]
    for composite_key in ("anyOf", "oneOf"):
        if composite_key in schema:
            variants = schema[composite_key]
            if isinstance(variants, list):
                non_null = [v for v in variants if isinstance(v, dict) and v.get("type") != "null"]
                chosen = non_null[0] if non_null else (variants[0] if variants else {})
                if schema.get("description") and not chosen.get("description"):
                    chosen = {**chosen, "description": schema["description"]}
                return _simplify_prop(chosen)

    schema_type = schema.get("type")

    # Complex types → string (these produce recursive grammar rules in GBNF)
    if schema_type in ("object", "array"):
        return {"type": "string", "description": schema.get("description", "")}

    # Strip keys that generate complex / large grammar rules
    drop_keys = {
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        "minLength", "maxLength", "minItems", "maxItems",
        "$schema", "additionalProperties", "pattern",
        "items", "default", "examples", "$defs", "definitions",
        "unevaluatedProperties", "allOf", "if", "then", "else",
        "prefixItems", "contains", "propertyNames",
    }
    out = {}
    for k, v in schema.items():
        if k in drop_keys:
            continue
        if k == "enum" and isinstance(v, list):
            # Cap large enums to avoid bloated grammar rules
            out[k] = v[:12] if len(v) > 12 else v
        else:
            out[k] = v
    return out


def _simplify_tools_for_llama(tools: list) -> list:
    """
    Apply _simplify_prop to every parameter property in a list of OAI-format
    tools.  The top-level 'parameters' object (always type:object) is kept
    intact; only its individual properties are simplified.
    """
    result = []
    for tool in tools:
        fn = tool.get("function", {})
        params = fn.get("parameters", {})
        props = params.get("properties")
        if not isinstance(props, dict):
            result.append(tool)
            continue
        new_props = {name: _simplify_prop(pschema) for name, pschema in props.items()}
        # Also clean up top-level noise on the params container itself
        new_params = {k: v for k, v in params.items()
                      if k not in {"additionalProperties", "$schema"}}
        new_params["properties"] = new_props
        result.append({
            **tool,
            "function": {**fn, "parameters": new_params},
        })
    return result


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
    logger.info("Shutting down Allma...")
    with state.global_lock:
        names = list(state.active_servers.keys())
    for name in names:
        shutdown_server(name, reason="shutdown", fast=True)
    await close_http_client()


app = FastAPI(title="Allma - LLM API", lifespan=lifespan)


# OpenAI-spec roles the Qwen template doesn't accept, mapped to supported ones.
# The Qwen3.x template only handles system/user/assistant/tool and raises
# "Unexpected message role." on anything else (e.g. "developer").
_ROLE_ALIASES = {"developer": "system"}


def _content_to_text(content) -> str:
    """Flatten OpenAI message content to plain text, dropping non-text blocks.
    Used for system content, which the Qwen template forbids from holding
    images/videos."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and (item.get("type") == "text" or "text" in item)
        ]
        return "\n".join(p for p in parts if p)
    return ""


def _normalize_messages_for_qwen(messages: list) -> list:
    """Make a messages list safe for the strict Qwen3.x chat template.

    The template raises on: a non-leading system message, unknown roles, and
    images/videos inside a system message (both vLLM and llama.cpp --jinja
    enforce it). Agent clients such as hermes occasionally inject system/context
    messages mid-conversation or use the "developer" role. We:
      1. map known role aliases (developer -> system) to supported roles, and
      2. hoist + merge every system message into a single text-only leading one.
    No-op for already-valid input. Returns the original list unchanged when
    nothing needed fixing.
    """
    if not messages:
        return messages

    # 1) Map roles the template would reject.
    mapped = []
    role_changed = False
    for m in messages:
        if m.get("role") in _ROLE_ALIASES:
            m = dict(m, role=_ROLE_ALIASES[m["role"]])
            role_changed = True
        mapped.append(m)

    # 2) Hoist/merge system messages to a single text-only leading message.
    sys_idx = [i for i, m in enumerate(mapped) if m.get("role") == "system"]
    needs_hoist = bool(sys_idx) and not (len(sys_idx) == 1 and sys_idx[0] == 0)

    if not needs_hoist:
        return mapped if role_changed else messages

    sys_text = "\n\n".join(
        t for t in (_content_to_text(mapped[i].get("content", "")) for i in sys_idx) if t
    )
    others = [m for m in mapped if m.get("role") != "system"]
    logger.info(
        f"Normalized messages for Qwen template: hoisted {len(sys_idx)} system "
        f"message(s) to front{', mapped developer->system' if role_changed else ''}"
    )
    return [{"role": "system", "content": sys_text}] + others


def _estimate_prompt_tokens(messages: list) -> int:
    """Cheap token estimate for the prompt so we can pre-clamp max_tokens
    without pulling in a tokenizer. Overestimates (chars/3 + per-image cost)
    so we err on the safe side; the reactive refit below is the exact catch."""
    chars = 0
    images = 0
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    chars += len(part.get("text", ""))
                else:                       # image / audio / other modality
                    images += 1
    return chars // 3 + images * 1200 + 512


_CTX_RE = re.compile(r"maximum context length is (\d+)")
_INPUT_RE = re.compile(r"contains at least (\d+) input tokens")


def _ctx_overflow_fit(error_detail: str) -> Optional[int]:
    """If a backend 400 is the 'prompt + max_tokens > context' overflow, return
    a max_tokens value that fits; else None. vLLM states both numbers in the
    error. The margin is generous (256) because vLLM reports the input as a
    lower bound ('at least N') that can drift by a few tokens between calls."""
    if "maximum context length" not in error_detail:
        return None
    ctx = _CTX_RE.search(error_detail)
    inp = _INPUT_RE.search(error_detail)
    if not ctx or not inp:
        return None
    return max(256, int(ctx.group(1)) - int(inp.group(1)) - 256)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: dict = Body(...)):
    model_name = body.get("model", "")
    client_host = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")

    logger.info(
        f"[HTTP] {request.method} {request.url.path} from {client_host} ({format_user_agent(user_agent)})"
    )

    if model_name not in PROFILE_MODELS:
        return JSONResponse(
            status_code=404,
            content={"error": f"Model '{model_name}' not found"},
        )

    # Qwen template is strict (system must be first, only known roles, no media
    # in system) — normalize before truncation/injection so client quirks (e.g.
    # hermes injecting a mid-conversation system message) don't 400 the template.
    if "messages" in body and body["messages"]:
        body["messages"] = _normalize_messages_for_qwen(body["messages"])

    if "messages" in body and MAX_MESSAGES > 0:
        msgs = body["messages"]
        if len(msgs) > MAX_MESSAGES:
            # Separate pinned prefix (system + first user) from the rest
            pinned = []
            rest_start = 0
            if msgs and msgs[0].get("role") == "system":
                pinned.append(msgs[0])
                rest_start = 1
            # Always keep the first user message so the model has a query anchor
            for i in range(rest_start, len(msgs)):
                if msgs[i].get("role") == "user":
                    pinned.append(msgs[i])
                    rest_start = i + 1
                    break
            rest = msgs[rest_start:]
            # Fill remaining budget with the tail of the conversation
            budget = max(MAX_MESSAGES - len(pinned), 0)
            body["messages"] = pinned + (rest[-budget:] if budget > 0 else [])

    logical_cfg = PROFILE_MODELS[model_name]
    base_name = logical_cfg["base"]
    cfg = state.effective_base_cfg(base_name)
    backend = cfg.get("backend", "vllm")

    port = await ensure_base_model(base_name, model_name)

    if backend == "vllm":
        body["model"] = cfg["path"]
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
    else:
        body["model"] = cfg["model"]
        url = f"http://127.0.0.1:{port}/chat/completions"

    sampling = state.effective_sampling(model_name, logical_cfg)
    for key in ["temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"]:
        if key in sampling:
            if key not in body or body[key] is None:
                body[key] = sampling[key]
        else:
            body.pop(key, None)

    # Inject profile-level system prompt (prepended to the client's system message)
    profile_system_prompt = logical_cfg.get("system_prompt", "").strip()
    if profile_system_prompt and "messages" in body and body["messages"]:
        msgs = body["messages"]
        if msgs[0].get("role") == "system":
            existing = msgs[0].get("content", "")
            if isinstance(existing, str):
                msgs[0] = dict(msgs[0], content=profile_system_prompt + "\n\n" + existing)
            elif isinstance(existing, list):
                msgs[0] = dict(msgs[0], content=[{"type": "text", "text": profile_system_prompt + "\n\n"}] + existing)
        else:
            body["messages"] = [{"role": "system", "content": profile_system_prompt}] + msgs

    # Disable thinking when profile says so or when "instruct" is in the model name
    if logical_cfg.get("enable_thinking") is False or "instruct" in model_name.lower():
        body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    elif backend == "llama.cpp":
        # Inject reasoning_budget for llama.cpp to prevent infinite thinking loops.
        # llama.cpp defaults to INT_MAX (2147483647) when no budget is given, causing
        # the model to think for hundreds of tokens before every response.
        # Profile field: thinking_budget (int, tokens). Default: 2048.
        # llama.cpp defaults to INT_MAX when no budget is given — model thinks
        # for hundreds of tokens per response. 2048 ≈ 10–15 s at ~30 t/s,
        # which is enough for complex reasoning while keeping chat snappy.
        budget = logical_cfg.get("thinking_budget")
        if budget is None:
            budget = 2048
        try:
            budget = int(budget)
        except (TypeError, ValueError):
            budget = 2048
        if "reasoning_budget" not in body:
            body["reasoning_budget"] = budget

    # Simplify tool schemas for llama.cpp to prevent GBNF grammar parse failures
    if backend == "llama.cpp" and body.get("tools"):
        body["tools"] = _simplify_tools_for_llama(body["tools"])

    logger.debug(f"{model_name} -> {base_name}:{port} ({backend})")

    if "messages" not in body or not body["messages"]:
        logger.warning(" Empty messages request")
        if body.get("stream", False):
            async def empty_stream():
                yield "data: [DONE]\n\n"
            return StreamingResponse(empty_stream(), media_type="text/event-stream")
        return JSONResponse(
            status_code=200,
            content={"choices": [{"message": {"content": ""}}]},
        )

    # Clamp max_tokens to a sane OUTPUT budget. Clients often pin a huge
    # max_tokens (e.g. 65536); once the prompt grows — tool results, images —
    # prompt + max_tokens overflows the context window and vLLM 400s, which
    # streaming clients surface as "an error occurred during streaming" and
    # retry to death. Filling the whole remaining context is also what drives
    # the OOM (exit -9) crashes, since a near-full sequence explodes the KV
    # cache. So cap output at ctx//4 AND ensure it still fits behind the prompt.
    max_model_len = int(cfg.get("max_model_len") or cfg.get("n_ctx") or 0)
    if max_model_len > 0:
        est = _estimate_prompt_tokens(body.get("messages") or [])
        output_cap = max(512, max_model_len // 4)
        fit_cap = max(512, max_model_len - est - 256)
        ceiling = min(output_cap, fit_cap)
        req_max = body.get("max_tokens")
        if req_max is None or req_max > ceiling:
            logger.info(
                f" max_tokens {req_max} -> {ceiling} "
                f"(ctx {max_model_len}, ~{est} prompt, out cap {output_cap})"
            )
            body["max_tokens"] = ceiling

    # Ceiling the reactive backstop uses too, so a refit can never re-authorize
    # a context-filling (OOM-prone) generation.
    _out_cap = max(512, max_model_len // 4) if max_model_len > 0 else 0

    client = await get_http_client()
    try:
        req_body = body

        if req_body.get("stream", False):
            async def generate():
                try:
                    for attempt in (1, 2):
                        async with client.stream("POST", url, json=req_body) as response:
                            if response.status_code != 200:
                                error_body = await response.aread()
                                error_detail = error_body.decode() if isinstance(error_body, bytes) else str(error_body)
                                # Exact refit from the backend's own numbers, then
                                # retry once before giving up (no content sent yet).
                                fit = _ctx_overflow_fit(error_detail)
                                if fit and _out_cap:
                                    fit = min(fit, _out_cap)
                                if fit and attempt == 1:
                                    logger.warning(
                                        f"↻ context overflow — refit max_tokens -> {fit}, retrying"
                                    )
                                    req_body["max_tokens"] = fit
                                    continue
                                logger.error(f"Backend returned {response.status_code} for {model_name}: {error_detail}")

                                # Analyze backend error
                                error_analysis = ErrorDetector.analyze_log(error_detail)
                                if error_analysis:
                                    error_response = {
                                        "error": error_detail,
                                        "error_type": error_analysis.error_type,
                                        "explanation": error_analysis.explanation,
                                        "suggestions": error_analysis.suggestions,
                                    }
                                    yield f'data: {_json.dumps(error_response)}\n\n'
                                else:
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
                            return
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    yield f'data: {_json.dumps({"error": {"message": str(e), "type": "stream_error"}})}\n\n'
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            resp = await client.post(url, json=req_body)
            if resp.status_code == 400:
                error_body = await resp.aread()
                error_detail = (
                    error_body.decode() if isinstance(error_body, bytes) else str(error_body)
                )
                # Exact refit + one retry before surfacing the 400 to the client.
                fit = _ctx_overflow_fit(error_detail)
                if fit and _out_cap:
                    fit = min(fit, _out_cap)
                if fit:
                    logger.warning(f"↻ context overflow — refit max_tokens -> {fit}, retrying")
                    req_body["max_tokens"] = fit
                    resp = await client.post(url, json=req_body)
                    if resp.status_code == 200:
                        return JSONResponse(content=resp.json(), status_code=200)
                    error_body = await resp.aread()
                    error_detail = (
                        error_body.decode() if isinstance(error_body, bytes) else str(error_body)
                    )
                logger.error(f"vLLM 400 Error for {model_name}: {error_detail}")

                # Analyze backend error
                error_analysis = ErrorDetector.analyze_log(error_detail)
                if error_analysis:
                    logger.warning(
                        f"   Tipo: {error_analysis.error_type}\n"
                        f"   {error_analysis.explanation}"
                    )
                    for sugg in error_analysis.suggestions:
                        logger.warning(f"   → {sugg}")

                try:
                    error_json = resp.json()
                except Exception:
                    error_json = {"error": error_detail}
                return JSONResponse(status_code=400, content=error_json)
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

    logger.info(f"[HTTP] {request.method} {request.url.path} from {client_host} ({format_user_agent(user_agent)})")

    if model_name not in PROFILE_MODELS:
        orig_name = body.get("model")
        # Priority 1: use the profile pinned by `allma launch` / /v1/load
        if state.default_profile and state.default_profile in PROFILE_MODELS:
            model_name = state.default_profile
            logger.info(f"Auto-switch: {orig_name} -> {model_name} (pinned default)")
        else:
            # Priority 2: fall back to any currently loaded model
            with state.global_lock:
                loaded = [
                    name
                    for name, srv in state.active_servers.items()
                    if srv.get("process") and srv["process"].poll() is None
                ]
            if loaded:
                base = loaded[0]
                model_name = next(
                    (k for k, v in PROFILE_MODELS.items() if v["base"] == base),
                    None,
                )
                if model_name and model_name != orig_name:
                    logger.info(f"Auto-switch: {orig_name} -> {model_name} (using loaded {base})")
        if not model_name or model_name not in PROFILE_MODELS:
            return JSONResponse(
                status_code=404,
                content={"error": f"Model '{body.get('model')}' not found"},
            )

    logical_cfg = PROFILE_MODELS[model_name]
    base_name = logical_cfg["base"]
    cfg = state.effective_base_cfg(base_name)
    backend = cfg.get("backend", "vllm")

    port = await ensure_base_model(base_name, model_name)

    max_model_len = int(cfg.get("max_model_len") or cfg.get("n_ctx") or "40960")
    max_output_cap = max_model_len // 4
    requested = body.get("max_tokens", max_output_cap)
    if requested > max_output_cap:
        logger.info(f" max_tokens {requested} -> {max_output_cap}")
        body["max_tokens"] = max_output_cap

    logger.debug(f"{model_name} -> {base_name}:{port} ({backend})")

    client = await get_http_client()

    # Translate Anthropic Messages -> OpenAI Chat for both backends. vLLM does
    # NOT expose /v1/messages natively (only OpenAI's /v1/chat/completions), so
    # we route every request through the translator below.
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
        # Simplify schemas for llama.cpp to prevent GBNF grammar parse failures
        if backend == "llama.cpp":
            oai_tools = _simplify_tools_for_llama(oai_tools)

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
        text_parts, image_parts, tool_uses, tool_results, thinking_parts = [], [], [], [], []
        for block in content:
            btype = block.get("type", "")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "thinking":
                # Anthropic Extended Thinking block — preserve as <think>...</think>
                # in the assistant content so Qwen3 keeps reasoning context across turns.
                thinking_parts.append(block.get("thinking", ""))
            elif btype == "image":
                source = block.get("source", {})
                src_type = source.get("type", "")
                if src_type == "base64":
                    media = source.get("media_type", "image/jpeg")
                    data = source.get("data", "")
                    image_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{media};base64,{data}"},
                    })
                elif src_type == "url":
                    image_parts.append({
                        "type": "image_url",
                        "image_url": {"url": source.get("url", "")},
                    })
            elif btype == "tool_use":
                tool_uses.append(block)
            elif btype == "tool_result":
                tool_results.append(block)

        if role == "assistant":
            oai_msg = {"role": "assistant"}
            # Re-attach thinking as <think>...</think> prefix so the Qwen3 template
            # sees the model's prior reasoning when the conversation history is replayed.
            text_joined = "".join(text_parts)
            if thinking_parts:
                think_block = "<think>\n" + "\n\n".join(thinking_parts).strip() + "\n</think>\n\n"
                text_joined = think_block + text_joined
            oai_msg["content"] = text_joined if text_joined else None
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
            # User messages: build multipart content array if images present, else plain text
            if image_parts:
                oai_content = []
                if text_parts:
                    oai_content.append({"type": "text", "text": "".join(text_parts)})
                oai_content.extend(image_parts)
                oai_messages.append({"role": "user", "content": oai_content})
            elif text_parts:
                oai_messages.append({"role": "user", "content": "".join(text_parts)})
            for tr in tool_results:
                tr_content = tr.get("content", "")
                tr_images = []
                if isinstance(tr_content, list):
                    for b in tr_content:
                        if b.get("type") == "image":
                            source = b.get("source", {})
                            src_type = source.get("type", "")
                            if src_type == "base64":
                                media = source.get("media_type", "image/jpeg")
                                data = source.get("data", "")
                                tr_images.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{media};base64,{data}"},
                                })
                            elif src_type == "url":
                                tr_images.append({
                                    "type": "image_url",
                                    "image_url": {"url": source.get("url", "")},
                                })
                    tr_content = "".join(
                        b.get("text", "") for b in tr_content if b.get("type") == "text"
                    )
                oai_messages.append({
                    "role": "tool",
                    "tool_call_id": tr.get("tool_use_id", ""),
                    "content": str(tr_content),
                })
                # tool messages can't carry images in OpenAI format — inject as user message
                if tr_images:
                    oai_messages.append({"role": "user", "content": tr_images})

    # Same Qwen3-template safety net used in /v1/chat/completions: ensure system
    # is hoisted to the front and no unknown roles slip through (otherwise the
    # template raises 'System message must be at the beginning.').
    oai_messages = _normalize_messages_for_qwen(oai_messages)

    # Backend-specific naming: vLLM is the OpenAI server with /v1 prefix and
    # expects the full HF model path; llama-server uses /chat/completions and
    # takes the GGUF path. Pick both based on backend.
    backend_model = cfg["path"] if backend == "vllm" else cfg["model"]
    oai_body = {
        "model": backend_model,
        "messages": oai_messages,
        "max_tokens": body.get("max_tokens", max_output_cap),
        "stream": body.get("stream", False),
    }
    if "temperature" in body:
        oai_body["temperature"] = body["temperature"]
    # Apply profile sampling overrides (temperature, top_p, top_k, min_p) for
    # both backends — same behavior the /v1/chat/completions path provides.
    sampling = state.effective_sampling(model_name, logical_cfg)
    for key in ["temperature", "top_p", "top_k", "min_p"]:
        if key in sampling and key not in oai_body:
            oai_body[key] = sampling[key]
    if oai_tools:
        oai_body["tools"] = oai_tools

    # Thinking control:
    #  - llama.cpp uses chat_template_kwargs.enable_thinking + reasoning_budget
    #  - vLLM Qwen template uses chat_template_kwargs.enable_thinking only
    if logical_cfg.get("enable_thinking") is False or "instruct" in model_name.lower():
        oai_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    elif backend == "llama.cpp":
        # Same reasoning_budget guard as the /v1/chat/completions path
        budget = logical_cfg.get("thinking_budget")
        if budget is None:
            budget = 2048
        try:
            budget = int(budget)
        except (TypeError, ValueError):
            budget = 2048
        oai_body.setdefault("reasoning_budget", budget)

    url = (
        f"http://127.0.0.1:{port}/v1/chat/completions"
        if backend == "vllm"
        else f"http://127.0.0.1:{port}/chat/completions"
    )
    llama_req_body = oai_body

    try:
        if llama_req_body["stream"]:
            msg_id = f"msg_{_uuid.uuid4().hex[:24]}"

            async def generate_llama():
                try:
                    # Anthropic SSE preamble
                    yield f"event: message_start\ndata: {_json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model_name, 'stop_reason': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

                    block_idx = 0
                    thinking_started = False
                    text_started = False
                    tool_calls_acc = {}  # tc_index -> {id, name, arguments, block_idx}
                    stop_reason = "end_turn"

                    def _close_thinking():
                        nonlocal block_idx, thinking_started
                        if thinking_started:
                            ev = f"event: content_block_stop\ndata: {_json.dumps({'type': 'content_block_stop', 'index': block_idx})}\n\n"
                            block_idx += 1
                            thinking_started = False
                            return ev
                        return None

                    async with client.stream("POST", url, json=llama_req_body) as response:
                        if response.status_code != 200:
                            error_body = await response.aread()
                            error_detail = error_body.decode() if isinstance(error_body, bytes) else str(error_body)
                            logger.error(f"Backend returned {response.status_code}: {error_detail}")

                            # Analyze backend error
                            error_analysis = ErrorDetector.analyze_log(error_detail)
                            if error_analysis:
                                logger.error(f"   Tipo: {error_analysis.error_type}")
                                logger.error(f"   {error_analysis.explanation}")

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
                                delta_text = delta.get("content", "")
                                # vLLM emits 'reasoning'; llama.cpp emits 'reasoning_content'.
                                # Both map to the Anthropic 'thinking' content block.
                                delta_thinking = delta.get("reasoning_content") or delta.get("reasoning") or ""

                                # Track OpenAI finish_reason -> Anthropic stop_reason. Critical
                                # so Claude Code doesn't read end_turn when budget actually ran
                                # out mid-thinking ("length" -> "max_tokens", not "end_turn").
                                fr = chunk["choices"][0].get("finish_reason")
                                if fr == "length":
                                    stop_reason = "max_tokens"
                                elif fr == "tool_calls":
                                    stop_reason = "tool_use"
                                elif fr == "stop":
                                    if stop_reason == "end_turn":  # don't downgrade tool_use
                                        stop_reason = "end_turn"

                                # ── Thinking content (Anthropic Extended Thinking block) ──
                                if delta_thinking:
                                    if not thinking_started:
                                        yield f"event: content_block_start\ndata: {_json.dumps({'type': 'content_block_start', 'index': block_idx, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"
                                        thinking_started = True
                                    yield f"event: content_block_delta\ndata: {_json.dumps({'type': 'content_block_delta', 'index': block_idx, 'delta': {'type': 'thinking_delta', 'thinking': delta_thinking}})}\n\n"

                                # ── Text content (closes thinking if open) ──
                                if delta_text:
                                    ev = _close_thinking()
                                    if ev:
                                        yield ev
                                    if not text_started:
                                        yield f"event: content_block_start\ndata: {_json.dumps({'type': 'content_block_start', 'index': block_idx, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                                        text_started = True
                                    yield f"event: content_block_delta\ndata: {_json.dumps({'type': 'content_block_delta', 'index': block_idx, 'delta': {'type': 'text_delta', 'text': delta_text}})}\n\n"

                                # ── Tool calls ──
                                for tc_delta in delta.get("tool_calls", []):
                                        tc_idx = tc_delta.get("index", 0)
                                        if tc_idx not in tool_calls_acc:
                                            # New tool call — close any open thinking/text block first
                                            ev = _close_thinking()
                                            if ev:
                                                yield ev
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
                    if thinking_started:
                        yield f"event: content_block_stop\ndata: {_json.dumps({'type': 'content_block_stop', 'index': block_idx})}\n\n"
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
            resp = await client.post(url, json=llama_req_body)
            if resp.status_code != 200:
                error_detail = resp.text
                logger.error(f"Backend returned {resp.status_code} for {model_name}: {error_detail}")
                error_analysis = ErrorDetector.analyze_log(error_detail)
                if error_analysis:
                    logger.error(f"   {error_analysis.error_type}: {error_analysis.explanation}")
                return JSONResponse(
                    status_code=resp.status_code,
                    content={"error": {"type": "api_error", "message": error_detail}},
                )
            oai = resp.json()
            choices = oai.get("choices")
            if not choices:
                logger.error(f"Backend returned no choices for {model_name}: {oai}")
                return JSONResponse(
                    status_code=500,
                    content={"error": {"type": "api_error", "message": "Backend returned empty response"}},
                )
            choice = choices[0]
            message = choice.get("message", {})
            msg_id = f"msg_{_uuid.uuid4().hex[:24]}"

            content_blocks = []
            # Thinking block (vLLM emits 'reasoning'; llama.cpp emits 'reasoning_content')
            thinking_text = message.get("reasoning_content") or message.get("reasoning") or ""
            if thinking_text:
                content_blocks.append({"type": "thinking", "thinking": thinking_text})
            if message.get("content"):
                content_blocks.append({"type": "text", "text": message["content"]})

            # Map OpenAI finish_reason -> Anthropic stop_reason so Claude Code
            # knows when the budget actually ran out instead of "modelo terminou".
            fr = choice.get("finish_reason")
            if fr == "length":
                stop_reason = "max_tokens"
            elif fr == "tool_calls":
                stop_reason = "tool_use"
            else:
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


def _model_context_length(profile_cfg: dict) -> int | None:
    """Return n_ctx for a profile's base model, or None if not found."""
    base_cfg = BASE_MODELS.get(profile_cfg.get("base", ""), {})
    try:
        return int(base_cfg.get("n_ctx") or base_cfg.get("max_model_len") or 0) or None
    except (TypeError, ValueError):
        return None


@app.get("/v1/models")
async def models_list():
    data = []
    for k in sorted(PROFILE_MODELS.keys()):
        entry: dict = {"id": k, "object": "model"}
        ctx = _model_context_length(PROFILE_MODELS[k])
        if ctx:
            entry["context_length"] = ctx
        data.append(entry)
    return {"object": "list", "data": data}


@app.get("/v1/models/{model_id:path}")
async def model_retrieve(model_id: str):
    if model_id in PROFILE_MODELS:
        entry: dict = {"id": model_id, "object": "model"}
        ctx = _model_context_length(PROFILE_MODELS[model_id])
        if ctx:
            entry["context_length"] = ctx
        return entry
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


@app.get("/v1/hardware")
async def get_hardware_info():
    """
    Returns detected hardware profile and model calibrations.
    Useful for debugging and understanding environment configuration.
    """
    import time as _time

    result = {
        "detected_at": state.hardware_detected_at,
        "profile": None,
        "calibrations": {},
        "active_models": {},
        "uptime_seconds": _time.time() - state.startup_time if hasattr(state, 'startup_time') else 0,
    }

    if state.hardware_profile:
        result["profile"] = {
            "driver_version": state.hardware_profile.driver_version,
            "cuda_version": state.hardware_profile.cuda_version,
            "total_vram_gb": state.hardware_profile.total_vram_gb,
            "available_vram_gb": state.hardware_profile.available_vram_gb,
            "max_contiguous_gb": state.hardware_profile.max_contiguous_gb,
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "compute_capability": g.compute_capability,
                    "total_memory_gb": g.total_memory_gb,
                    "free_memory_gb": g.free_memory_gb,
                }
                for g in state.hardware_profile.gpus
            ],
        }

    # Calibrations
    for name, calib in state.bootstrap_calibrations.items():
        result["calibrations"][name] = {
            "backend": calib.backend,
            "tp": calib.recommended_tp,
            "ubatch_size": calib.recommended_ubatch_size,
            "n_batch": calib.recommended_n_batch,
            "n_ctx": calib.recommended_n_ctx,
            "cache_dtype": calib.recommended_cache_dtype,
            "confidence": calib.confidence,
            "warnings": calib.warnings,
            "estimated_vram_need_gb": calib.estimated_vram_need_gb,
            "calibrated_at": calib.calibrated_at,
        }

    # Active models
    with state.global_lock:
        for name, server in state.active_servers.items():
            idle_sec = _time.time() - state.server_idle_time.get(name, _time.time())
            result["active_models"][name] = {
                "backend": server.get("backend", "unknown"),
                "pid": server.get("pid"),
                "port": server.get("port"),
                "gpu": state.gpu_allocation.get(name),
                "idle_seconds": idle_sec,
            }

    return result


@app.head("/")
@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/health")
async def health():
    with state.global_lock:
        active = len(state.active_servers)
    return {
        "status": "healthy",
        "active_servers": active,
        "running": state.running,
    }


def _is_cpu_only(base_name: str) -> bool:
    """True when a base is configured to run on CPU (no GPU offload), so the UI
    can label it 'CPU' instead of a misleading GPU index."""
    cfg = BASE_MODELS.get(base_name, {})
    if str(cfg.get("n_gpu_layers", "")).strip() == "0":
        return True
    ea = cfg.get("extra_args", []) or []
    for flag in ("--device", "-dev"):
        if flag in ea:
            try:
                if str(ea[ea.index(flag) + 1]).strip().lower() == "none":
                    return True
            except (ValueError, IndexError):
                pass
    return False


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
                # CPU-only backends report gpu=None so the UI shows "CPU".
                "gpu": None if _is_cpu_only(name) else state.gpu_allocation.get(name),
                "logfile": str(srv.get("logfile", "")),
                "alive": bool(srv.get("process")) and srv["process"].poll() is None,
                # Running with ephemeral (one-time) overrides not saved to .allm
                "custom_load": name in state.session_load_overrides,
            }
            for name, srv in state.active_servers.items()
        ]
        # Include last known error analysis for any crashed model
        errors = {}
        for name, analysis in state.last_error_analysis.items():
            errors[name] = {
                "error_type": analysis.error_type,
                "explanation": analysis.explanation,
                "suggestions": analysis.suggestions,
            }
    return {"servers": servers, "errors": errors}


@app.post("/v1/shutdown")
async def shutdown_endpoint():
    """Gracefully shut down the server and all backends.

    SIGTERM triggers the graceful path in allma.py's signal handler (which
    already carries its own hard-exit watchdog). A second daemon thread here
    is belt-and-suspenders: it fires SIGKILL after 10s in case signals are
    being blocked by a native extension and the handler never runs.
    """
    import asyncio
    import os
    import signal as _signal
    import threading
    import time as _time

    def _nuclear_backup():
        _time.sleep(10)
        # Only fires if we're still alive — the signal handler's os._exit(0)
        # or its own hard-exit watchdog should have killed us long before this.
        os.kill(os.getpid(), _signal.SIGKILL)

    threading.Thread(target=_nuclear_backup, daemon=True).start()

    async def _do_shutdown():
        await asyncio.sleep(0.1)
        os.kill(os.getpid(), _signal.SIGTERM)

    asyncio.create_task(_do_shutdown())
    return JSONResponse({"status": "shutting down"})


@app.post("/v1/unload")
async def unload_model(body: dict = Body(...)):
    """Unload a running base model immediately, freeing its VRAM."""
    base_name = body.get("model", "")
    if not base_name:
        return JSONResponse(status_code=400, content={"error": "model required"})
    with state.global_lock:
        running = list(state.active_servers.keys())
    if base_name not in running:
        # Accept profile names too — resolve to base
        if base_name in PROFILE_MODELS:
            base_name = PROFILE_MODELS[base_name].get("base", base_name)
    if base_name not in running and base_name not in state.active_servers:
        return JSONResponse(
            status_code=404,
            content={"error": f"'{base_name}' is not loaded", "loaded": running},
        )
    import asyncio as _asyncio
    loop = _asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: shutdown_server(base_name, reason="manual unload"))
    # Clear default_profile if it pointed to this base model
    if state.default_profile and state.default_profile in PROFILE_MODELS:
        if PROFILE_MODELS[state.default_profile].get("base") == base_name:
            state.default_profile = None
            logger.info("Default profile cleared (model unloaded)")
    return JSONResponse({"status": "unloaded", "model": base_name})


@app.post("/v1/load")
async def load_model(body: dict = Body(...)):
    """Pre-load a model without generating any tokens.

    Optional ephemeral overrides (TUI "load one-time" — nothing touches disk):
      load_overrides: dict merged over the base .allm config for this session
      sampling:       dict merged over the profile sampling for this session
    Passing either key with an empty dict clears the corresponding override.
    """
    model_name = body.get("model", "")
    gpu_id = body.get("gpu_id")  # Optional: force specific GPU
    if model_name not in PROFILE_MODELS:
        return JSONResponse(
            status_code=404,
            content={"error": f"Model '{model_name}' not found"},
        )
    logical_cfg = PROFILE_MODELS[model_name]
    base_name = logical_cfg["base"]

    if "sampling" in body:
        sampling_overrides = body.get("sampling") or {}
        if sampling_overrides:
            state.session_sampling[model_name] = sampling_overrides
            logger.info(f"Session sampling for '{model_name}': {sampling_overrides}")
        else:
            state.session_sampling.pop(model_name, None)

    if "load_overrides" in body:
        load_overrides = body.get("load_overrides") or {}
        previous = state.session_load_overrides.get(base_name, {})
        if load_overrides != previous:
            if load_overrides:
                state.session_load_overrides[base_name] = load_overrides
                logger.info(f"Session load overrides for '{base_name}': {load_overrides}")
            else:
                state.session_load_overrides.pop(base_name, None)
            # Effective config changed — restart the backend so it applies
            with state.global_lock:
                already_running = base_name in state.active_servers
            if already_running:
                logger.info(f"Reloading {base_name} to apply session overrides")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, lambda: shutdown_server(base_name, "reload", fast=True)
                )

    cfg = state.effective_base_cfg(base_name)
    backend = cfg.get("backend", "vllm")
    await ensure_base_model(base_name, model_name, gpu_id=gpu_id)
    # Pin this profile as the default for unknown model names (e.g. "claude-sonnet-4-5")
    # so that `allma launch claude <model>` always routes to the intended model.
    state.default_profile = model_name
    logger.info(f"Default profile set to '{model_name}'")
    return JSONResponse({
        "status": "loaded",
        "model": model_name,
        "backend": backend,
        "custom_load": base_name in state.session_load_overrides,
    })


@app.post("/v1/reload-configs")
async def reload_configs():
    """Re-read configs/*.allm so saved edits take effect without a restart.

    BASE_MODELS/PROFILE_MODELS are imported by reference across modules, so they
    must be mutated in place. Running backends keep their current config until
    the next (re)load.
    """
    from core.config import load_models_from_configs
    new_base, new_profiles = load_models_from_configs()
    if not new_base and not new_profiles:
        return JSONResponse(
            status_code=500,
            content={"error": "Config reload returned nothing — keeping current configs"},
        )
    BASE_MODELS.clear()
    BASE_MODELS.update(new_base)
    PROFILE_MODELS.clear()
    PROFILE_MODELS.update(new_profiles)
    logger.info(f"Configs reloaded: {len(new_base)} bases, {len(new_profiles)} profiles")
    return JSONResponse({
        "status": "reloaded",
        "bases": len(new_base),
        "profiles": len(new_profiles),
    })


# ==============================================================================
# BANNER
# ==============================================================================
def show_banner():
    if not RICH_AVAILABLE:
        logger.info("=" * 60)
        logger.info("Allma Started")
        logger.info("=" * 60)
        for name in BASE_MODELS:
            from core.config import ALLMA_LOG_DIR
            logger.info(f"   - {name} - tail -f {ALLMA_LOG_DIR}/{name}.log")
        logger.info("=" * 60)
        logger.info(f"Models configured: {len(BASE_MODELS)} base, {len(PROFILE_MODELS)} profile")
        logger.info(f"Keep-alive: {KEEP_ALIVE_SECONDS}s")
        logger.info(f"API: http://127.0.0.1:{ALLMA_PORT}")
        logger.info("=" * 60)
        return

    from rich import box as _box
    from rich.align import Align
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _console_probe = Console()
    _term_w  = _console_probe.width
    W        = min(max(_term_w - 4, 40), 130)  # -4 = shadow (2) + margin (2)
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

    LOGO_ROWS = [
        "█████  ██╗     ██╗     ███    ███  █████ ",
        "██   ██ ██║     ██║     ████  ████ ██   ██",
        "███████ ██║     ██║     ██ ████ ██ ███████",
        "██   ██ ██║     ██║     ██  ██  ██ ██   ██",
        "██   ██ ███████╗███████╗██      ██ ██   ██",
        "        ╚══════╝╚══════╝__________________",
    ]

    # ── logo panel ────────────────────────────────────────────────────────────
    # cols 8-23 = the two L's; box-drawing chars there are rendered dim (shadow)
    # box-drawing chars on A and M are rendered bold to keep letters solid
    _BOX  = set("╗╔╚╝═║")
    _L_COLS = range(8, 24)
    LOGO_COLORS = [C_RED, C_ORG, C_YELL, C_GRN, C_BLUE, C_BLUE]
    logo_text   = Text(justify="center")
    for i, row in enumerate(LOGO_ROWS):
        color = LOGO_COLORS[i]
        for j, ch in enumerate(row):
            if ch == "█":
                logo_text.append(ch, style=f"bold {color}")
            elif ch in _BOX:
                if j in _L_COLS:
                    logo_text.append(ch, style=f"dim {color}")
                else:
                    logo_text.append(" ")
            elif ch == "_":
                logo_text.append(ch, style=C_BG)
            else:
                logo_text.append(ch)
        if i < 5:
            logo_text.append("\n")
    # ── configuration summary ─────────────────────────────────────────────────
    cfg_line = Text()
    cfg_line.append(f"  {len(BASE_MODELS)}", style=f"bold {C_FG}")
    cfg_line.append(" base model(s)  ·  ", style=C_DIM)
    cfg_line.append(f"{len(PROFILE_MODELS)}", style=f"bold {C_FG}")
    cfg_line.append(" profile (s)  ·  keep-alive: ", style=C_DIM)
    cfg_line.append(f"{KEEP_ALIVE_SECONDS}s", style=f"bold {C_FG}")

    # ── base models table — sorted by backend group then size desc ───────
    import re as _re

    def _param_b(model_name: str) -> float:
        m = _re.search(r'(\d+(?:\.\d+)?)\s*[bB]', model_name)
        return float(m.group(1)) if m else 0.0

    sorted_phys = sorted(
        BASE_MODELS.items(),
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
    _w_name_p = 24 if _narrow else 35

    phys_tbl = Table(box=None, padding=_tbl_pad, show_header=True,
                     header_style=f"bold {C_DIM}", expand=True)
    phys_tbl.add_column("SAFETENSORS",  style=C_FG,     no_wrap=False, min_width=_w_name_p)
    phys_tbl.add_column("GGUF",         style=C_FG,     no_wrap=False, min_width=_w_name_p)

    # separate safetensors and gguf
    _sft_items = sorted(_phys_groups.get("vllm", []), key=lambda x: x[0].lower())
    _gguf_items = sorted(_phys_groups.get("gguf", []), key=lambda x: x[0].lower())

    # pair them side by side
    _max_rows = max(len(_sft_items), len(_gguf_items))
    for _i in range(_max_rows):
        _sft_name = _sft_items[_i][0] if _i < len(_sft_items) else ""
        _gguf_name = _gguf_items[_i][0] if _i < len(_gguf_items) else ""
        phys_tbl.add_row(_sft_name, _gguf_name)

    sorted_log = sorted(PROFILE_MODELS.items(), key=lambda kv: kv[0].lower())

    # hierarchical grouping: Model Base → Variant → Profile Variant
    import re as _re_hier
    _hier: dict[str, dict[str, list]] = {}  # {base: {variant: [(name, cfg), ...]}}
    for _lname, _lcfg in sorted_log:
        # parse name like "Qwen3.6:27B-Code-FP8" or "Qwen3.6:35B-A3B-Reasoning-FP8".
        # The param group captures dense ("27B") and MoE ("35B-A3B") sizes.
        _match = _re_hier.match(r'^([^:]+):(\d+[A-Za-z](?:-A\d+[A-Za-z])?)(?:-(.+))?$', _lname)
        if _match:
            _base, _variant, _suffix = _match.groups()
            if _base not in _hier:
                _hier[_base] = {}
            if _variant not in _hier[_base]:
                _hier[_base][_variant] = []
            _hier[_base][_variant].append((_lname, _lcfg, _suffix or ""))

    _w_name_l = 14 if _narrow else 32

    log_tbl = Table(box=None, padding=_tbl_pad, show_header=True,
                    header_style=f"bold {C_DIM}", expand=True)
    log_tbl.add_column("PROFILE", style=C_FG,  no_wrap=False, min_width=_w_name_l)
    log_tbl.add_column("TEMP", style=C_DIM, justify="right", min_width=4)
    log_tbl.add_column("TOP_P", style=C_DIM, justify="right", min_width=4)
    log_tbl.add_column("TOP_K", style=C_DIM, justify="right", min_width=4)

    # render hierarchy with elegant unicode tree
    _bases = sorted(_hier.keys())
    for _bi, _base in enumerate(_bases):
        # base model header with diamond
        _base_text = Text(f"◆ {_base}", style=f"bold {C_FG}")
        log_tbl.add_row(_base_text, "", "", "")

        _variants = sorted(_hier[_base].keys())
        for _vi, _variant in enumerate(_variants):
            _is_last_var = _vi == len(_variants) - 1
            # variant row with tree connector and small circle
            _var_prefix = "  └─ ◦ " if _is_last_var else "  ├─ ◦ "
            _var_text = Text(_var_prefix + _variant, style=C_DIM)
            log_tbl.add_row(_var_text, "", "", "")

            # render profiles under variant
            _profiles = _hier[_base][_variant]
            for _pi, (_lname, _lcfg, _suffix) in enumerate(_profiles):
                _sampling = _lcfg.get("sampling", {})
                _is_last_prof = _pi == len(_profiles) - 1

                # extract just the variant part + suffix (e.g., "27b", "27b-Code")
                _profile_name = _lname.split(':')[1]

                # tree connector with pipe for branches
                _t = _sampling.get("temperature", "—")
                _p = _sampling.get("top_p", "—")
                _k = _sampling.get("top_k", "—")
                _pipe = "  │  " if not _is_last_var else "     "
                _connector = _pipe + ("└─ " if _is_last_prof else "├─ ")

                _pname = Text(_connector + _profile_name, style=C_FG)

                log_tbl.add_row(
                    _pname,
                    str(_t),
                    str(_p),
                    str(_k),
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
        title=_sec_title("Base Models"), title_align="left",
        box=_box.SQUARE, border_style=C_DIM,
        style=_S, padding=_inner_pad,
    )

    log_panel = Panel(
        log_tbl,
        title=_sec_title("Profiles"), title_align="left",
        box=_box.SQUARE, border_style=C_DIM,
        style=_S, padding=_inner_pad,
    )

    # ── status bar ────────────────────────────────────────────────────────────
    status_line = Text(justify="center")
    status_line.append("● ", style=f"bold {C_OK}")
    status_line.append("READY", style=f"bold {C_OK}")
    status_line.append("  ·  ", style=C_DIM)
    if _narrow:
        status_line.append(f":{ALLMA_PORT}", style=C_ACCENT)
    else:
        status_line.append(f"http://127.0.0.1:{ALLMA_PORT}", style=C_ACCENT)
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
