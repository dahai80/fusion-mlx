"""OpenClaw Agent Protocol API routes for fusion-mlx.

Extends the standard OpenAI-compatible endpoints with OpenClaw-specific
agent protocol features:
- /v1/openclaw/agent/sessions    — session lifecycle
- /v1/openclaw/agent/turns       — multi-turn agent turns with tool calling
- /v1/openclaw/agent/stream       — SSE stream of agent events
- /v1/openclaw/agent/steer        — mid-conversation steering
"""

import asyncio
import json
import logging
import time
import uuid
from collections import OrderedDict
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/openclaw/agent", tags=["openclaw-agent"])

# In-memory session store with TTL and max cap to prevent memory leak.
# OrderedDict for LRU eviction when cap is reached.
_SESSION_TTL_SECONDS = 3600  # 1 hour
_SESSION_MAX_COUNT = 1000
_sessions: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _init_session() -> dict[str, Any]:
    return {
        "messages": [],
        "tools": [],
        "active": False,
        "turn_count": 0,
        "created_at": time.time(),
        "last_accessed": time.time(),
    }


def _cleanup_expired_sessions() -> None:
    """Remove sessions older than TTL, evict LRU if over cap."""
    now = time.time()
    # Remove expired sessions
    expired = [
        sid
        for sid, s in _sessions.items()
        if now - s["last_accessed"] > _SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del _sessions[sid]

    # Evict oldest sessions if still over cap
    while len(_sessions) > _SESSION_MAX_COUNT:
        _sessions.popitem(last=False)


# ── Request/Response Models ──────────────────────────────────────────────


class TurnRequest(BaseModel):
    """Agent turn request with optional tool definitions."""

    messages: list[dict[str, Any]] = Field(..., description="Conversation messages")
    tools: list[dict[str, Any]] | None = None
    max_tokens: int = Field(default=4096, ge=1, le=131072)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    model: str | None = None
    auto_execute: bool = Field(
        default=False,
        description="If True, server auto-executes tool calls and loops until final answer",
    )
    max_auto_iterations: int = Field(default=10, ge=1, le=50)


class TurnResponse(BaseModel):
    """Agent turn response with content and optional tool calls."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = []
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
    session_id: str


class SessionCreateRequest(BaseModel):
    """Create a new agent session."""

    system_prompt: str | None = None
    tools: list[dict[str, Any]] | None = None
    model: str | None = None


class SessionInfo(BaseModel):
    """Agent session metadata."""

    session_id: str
    turn_count: int = 0
    active: bool = False
    model: str | None = None
    tools_count: int = 0


class SteerRequest(BaseModel):
    """Inject a steering message into an active agent turn."""

    session_id: str
    message: dict[str, Any] = Field(..., description="Message to inject")
    mode: str = Field(
        default="append",
        description="append=add to end, prepend=add to front, replace=replace last",
    )


class ToolResultRequest(BaseModel):
    """Submit tool execution results back to the agent."""

    session_id: str
    tool_call_id: str
    result: str


# ── Session Management ───────────────────────────────────────────────────


@router.post("/sessions", response_model=SessionInfo)
async def create_session(req: SessionCreateRequest):
    _cleanup_expired_sessions()
    session_id = uuid.uuid4().hex[:16]
    session = _init_session()
    session["system_prompt"] = req.system_prompt
    session["model"] = req.model
    if req.tools:
        session["tools"] = req.tools
    session["tools_count"] = len(req.tools or [])
    _sessions[session_id] = session
    logger.info(
        "Created agent session %s (model=%s, tools=%d)",
        session_id,
        req.model,
        session["tools_count"],
    )
    return SessionInfo(
        session_id=session_id,
        turn_count=0,
        active=False,
        model=req.model,
        tools_count=session["tools_count"],
    )


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(404, f"Session {session_id} not found")
    s = _sessions[session_id]
    s["last_accessed"] = time.time()
    _sessions.move_to_end(session_id)
    return SessionInfo(
        session_id=session_id,
        turn_count=s["turn_count"],
        active=s["active"],
        model=s.get("model"),
        tools_count=len(s.get("tools", [])),
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in _sessions:
        del _sessions[session_id]
    return {"deleted": True}


@router.get("/sessions")
async def list_sessions() -> list[SessionInfo]:
    _cleanup_expired_sessions()
    result = []
    for sid, s in _sessions.items():
        result.append(
            SessionInfo(
                session_id=sid,
                turn_count=s["turn_count"],
                active=s["active"],
                model=s.get("model"),
                tools_count=len(s.get("tools", [])),
            )
        )
    return result


# ── Agent Turn Execution ─────────────────────────────────────────────────


@router.post("/turns", response_model=TurnResponse)
async def execute_turn(session_id: str, req: TurnRequest):
    """Execute one agent turn with optional tool calling.

    When ``auto_execute=True``, the server automatically executes tool calls
    and loops until the model produces a final answer (no more tool calls)
    or ``max_auto_iterations`` is reached.
    """
    if session_id not in _sessions:
        raise HTTPException(404, f"Session {session_id} not found")

    session = _sessions[session_id]
    session["active"] = True
    session["turn_count"] += 1
    session["last_accessed"] = time.time()
    session["messages"].extend(req.messages)

    if req.tools:
        session["tools"] = req.tools

    pool = getattr(execute_turn, "_pool", None)
    if pool is None:
        raise HTTPException(450, "Engine pool not initialized")

    if req.auto_execute:
        return await _execute_turn_auto(pool, session, req, session_id)

    return await _execute_single_turn(pool, session, req, session_id)


async def _execute_single_turn(pool, session: dict, req: TurnRequest, session_id: str) -> TurnResponse:
    """Execute a single LLM turn without auto tool-calling loop."""
    messages = _build_request_messages(session, req)
    body = _build_request_body(session, req, messages)
    try:
        result = await _call_chat_completion(pool, body)
        result.session_id = session_id
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Agent turn failed for session %s", session_id)
        raise HTTPException(500, str(exc))


async def _execute_turn_auto(pool, session: dict, req: TurnRequest, session_id: str) -> TurnResponse:
    """Execute a full auto-loop: LLM → tool calls → execute → submit → LLM → ... → final answer."""
    all_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    final_content = ""
    all_tool_calls = []

    for iteration in range(req.max_auto_iterations):
        messages = _build_request_messages(session, req)
        body = _build_request_body(session, req, messages)

        try:
            result = await _call_chat_completion(pool, body)
        except Exception as exc:
            logger.exception("Auto-turn %d failed for session %s", iteration + 1, session_id)
            raise HTTPException(500, str(exc))

        # Accumulate usage
        for key in ("prompt_tokens", "completion_tokens"):
            all_usage[key] = all_usage.get(key, 0) + result.usage.get(key, 0)

        # Store assistant response
        assistant_msg = {"role": "assistant", "content": result.content}
        if result.tool_calls:
            assistant_msg["tool_calls"] = result.tool_calls
        session["messages"].append(assistant_msg)

        # Check for tool calls
        if not result.tool_calls:
            final_content = result.content
            break

        # Execute each tool call and submit results
        for tc in result.tool_calls:
            all_tool_calls.append(tc)
            tool_result = _execute_local_tool(tc)
            session["messages"].append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": tool_result,
            })

        if iteration == req.max_auto_iterations - 1:
            final_content = result.content

    return TurnResponse(
        content=final_content,
        tool_calls=all_tool_calls,
        usage=all_usage,
        session_id=session_id,
    )


def _execute_local_tool(tool_call: dict) -> str:
    """Execute a single tool call locally.

    For now, returns a stub result. In a full Agent Studio deployment this
    would dispatch to the ToolRegistry.
    """
    import json
    try:
        func_name = tool_call.get("function", {}).get("name", "unknown")
        args = tool_call.get("function", {}).get("arguments", "{}")
        # Parse arguments for display
        parsed = json.loads(args) if isinstance(args, str) else args
        return json.dumps({"executed": func_name, "args": parsed, "status": "ok"}, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        return json.dumps({"error": str(e), "status": "failed"}, ensure_ascii=False)


def _build_request_messages(session: dict, req: TurnRequest) -> list[dict]:
    """Build the message list for the LLM request."""
    messages = []
    if session.get("system_prompt"):
        messages.append({"role": "system", "content": session["system_prompt"]})
    messages.extend(session["messages"])
    return messages


def _build_request_body(session: dict, req: TurnRequest, messages: list[dict]) -> dict:
    """Build the request body for the LLM call."""
    body: dict = {
        "messages": messages,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }
    if req.model:
        body["model"] = req.model
    elif session.get("model"):
        body["model"] = session["model"]
    if session.get("tools"):
        body["tools"] = session["tools"]
    return body


async def _call_chat_completion(pool, body: dict) -> TurnResponse:
    """Route through the engine pool to get an LLM response."""
    model_name = body.get("model", "default")
    try:
        engine = await pool.get_engine(model_name)
        if engine is None:
            raise HTTPException(404, f"Model {model_name} not loaded")

        result = await engine.chat(
            messages=body.get("messages", []),
            max_tokens=body.get("max_tokens", 4096),
            temperature=body.get("temperature", 0.7),
            tools=body.get("tools"),
        )

        content = ""
        tool_calls = []
        prompt_tokens = 0
        completion_tokens = 0
        if isinstance(result, dict):
            content = result.get("text", result.get("content", ""))
            tool_calls = result.get("tool_calls", [])
            prompt_tokens = result.get("prompt_tokens", 0)
            completion_tokens = result.get("completion_tokens", 0)
        elif hasattr(result, "text"):
            content = result.text or ""
            if hasattr(result, "tool_calls") and result.tool_calls:
                tool_calls = result.tool_calls
            prompt_tokens = getattr(result, "prompt_tokens", 0) or 0
            completion_tokens = getattr(result, "completion_tokens", 0) or 0

        return TurnResponse(
            content=content,
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
            session_id="",
        )
    except (HTTPException, RuntimeError):
        raise
    except Exception as exc:
        logger.exception("Chat completion failed")
        return TurnResponse(content=f"Error: {exc}", session_id="")


# ── Tool Result Submission ───────────────────────────────────────────────


@router.post("/tool-results")
async def submit_tool_result(req: ToolResultRequest):
    """Submit tool execution results back to the agent."""
    if req.session_id not in _sessions:
        raise HTTPException(404, f"Session {req.session_id} not found")

    session = _sessions[req.session_id]
    session["messages"].append(
        {
            "role": "tool",
            "tool_call_id": req.tool_call_id,
            "content": req.result,
        }
    )
    logger.info(
        "Tool result submitted for session %s, tool %s",
        req.session_id,
        req.tool_call_id,
    )
    return {"accepted": True, "messages_count": len(session["messages"])}


# ── Agent Steering ───────────────────────────────────────────────────────


@router.post("/steer")
async def steer_agent(req: SteerRequest):
    """Inject a steering message into an active agent session."""
    if req.session_id not in _sessions:
        raise HTTPException(404, f"Session {req.session_id} not found")

    session = _sessions[req.session_id]
    msgs = session["messages"]

    if req.mode == "append":
        msgs.append(req.message)
    elif req.mode == "prepend":
        msgs.insert(0, req.message)
    elif req.mode == "replace":
        if msgs:
            msgs[-1] = req.message
        else:
            msgs.append(req.message)
    else:
        raise HTTPException(400, f"Unknown steering mode: {req.mode}")

    logger.info("Steered session %s with mode=%s", req.session_id, req.mode)
    return {"applied": True, "mode": req.mode, "messages_count": len(msgs)}


# ── SSE Event Stream ────────────────────────────────────────────────────


@router.get("/stream/{session_id}")
async def stream_events(session_id: str):
    """SSE stream of agent events for a session."""
    if session_id not in _sessions:
        raise HTTPException(404, f"Session {session_id} not found")

    async def event_generator():
        yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id})}\n\n"

        session = _sessions.get(session_id)
        if session:
            state = {
                "type": "session_state",
                "session_id": session_id,
                "turn_count": session["turn_count"],
                "active": session["active"],
                "messages_count": len(session["messages"]),
            }
            yield f"data: {json.dumps(state)}\n\n"

        try:
            while True:
                await asyncio.sleep(30)
                if session_id in _sessions:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'session_closed'})}\n\n"
                    break
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Context Injection ────────────────────────────────────────────────────


def set_openclaw_agent_pool(pool) -> None:
    """Inject engine pool for internal chat completion calls."""
    execute_turn._pool = pool
