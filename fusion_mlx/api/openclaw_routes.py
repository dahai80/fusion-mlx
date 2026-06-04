"""OpenClaw Agent Protocol API routes for fusion-mlx.

Extends the standard OpenAI-compatible endpoints with OpenClaw-specific
agent protocol features:
- /v1/openclaw/agent/sessions   — session lifecycle
- /v1/openclaw/agent/turns      — multi-turn agent turns with tool calling
- /v1/openclaw/agent/stream      — SSE stream of agent events
- /v1/openclaw/agent/steer       — mid-conversation steering
"""

import json
import logging
import uuid
from collections import defaultdict
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/openclaw/agent", tags=["openclaw-agent"])

# In-memory session store (replace with Redis for production)
_sessions: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
    "messages": [],
    "tools": [],
    "active": False,
    "turn_count": 0,
})


# ── Request/Response Models ──────────────────────────────────────────────

class TurnRequest(BaseModel):
    """Agent turn request with optional tool definitions."""
    messages: list[Dict[str, Any]] = Field(..., description="Conversation messages")
    tools: Optional[list[Dict[str, Any]]] = None
    max_tokens: int = Field(default=4096, ge=1, le=131072)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    model: Optional[str] = None


class TurnResponse(BaseModel):
    """Agent turn response with content and optional tool calls."""
    content: str = ""
    tool_calls: list[Dict[str, Any]] = []
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
    session_id: str


class SessionCreateRequest(BaseModel):
    """Create a new agent session."""
    system_prompt: Optional[str] = None
    tools: Optional[list[Dict[str, Any]]] = None
    model: Optional[str] = None


class SessionInfo(BaseModel):
    """Agent session metadata."""
    session_id: str
    turn_count: int = 0
    active: bool = False
    model: Optional[str] = None
    tools_count: int = 0


class SteerRequest(BaseModel):
    """Inject a steering message into an active agent turn."""
    session_id: str
    message: Dict[str, Any] = Field(..., description="Message to inject")
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
    session_id = uuid.uuid4().hex[:16]
    session = _sessions[session_id]
    session["system_prompt"] = req.system_prompt
    session["model"] = req.model
    if req.tools:
        session["tools"] = req.tools
    session["tools_count"] = len(req.tools or [])
    logger.info("Created agent session %s (model=%s, tools=%d)", session_id, req.model, session["tools_count"])
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
    result = []
    for sid, s in _sessions.items():
        result.append(SessionInfo(
            session_id=sid,
            turn_count=s["turn_count"],
            active=s["active"],
            model=s.get("model"),
            tools_count=len(s.get("tools", [])),
        ))
    return result


# ── Agent Turn Execution ─────────────────────────────────────────────────

@router.post("/turns", response_model=TurnResponse)
async def execute_turn(session_id: str, req: TurnRequest):
    """Execute one agent turn with optional tool calling."""
    if session_id not in _sessions:
        raise HTTPException(404, f"Session {session_id} not found")

    session = _sessions[session_id]
    session["active"] = True
    session["turn_count"] += 1
    session["messages"].extend(req.messages)

    if req.tools:
        session["tools"] = req.tools

    messages = []
    if session.get("system_prompt"):
        messages.append({"role": "system", "content": session["system_prompt"]})
    messages.extend(session["messages"])

    body = {
        "messages": messages,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }
    if req.model:
        body["model"] = req.model
    if session.get("tools"):
        body["tools"] = session["tools"]

    try:
        pool = getattr(execute_turn, "_pool", None)
        if pool is None:
            raise HTTPException(450, "Engine pool not initialized")
        result = await _call_chat_completion(pool, body)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Agent turn failed for session %s", session_id)
        raise HTTPException(500, str(exc))


async def _call_chat_completion(pool, body: dict) -> TurnResponse:
    """Route through the engine pool to get an LLM response."""
    model_name = body.get("model", "default")
    try:
        engine = await pool.get_engine(model_name)
        if engine is None:
            raise HTTPException(404, f"Model {model_name} not loaded")

        result = await engine.generate(
            messages=body.get("messages", []),
            max_tokens=body.get("max_tokens", 4096),
            temperature=body.get("temperature", 0.7),
            tools=body.get("tools"),
        )

        content = ""
        tool_calls = []
        if isinstance(result, dict):
            content = result.get("content", "")
            tool_calls = result.get("tool_calls", [])
        elif hasattr(result, "content"):
            content = result.content or ""

        return TurnResponse(
            content=content,
            tool_calls=tool_calls,
            usage={"prompt_tokens": 0, "completion_tokens": 0},
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
    session["messages"].append({
        "role": "tool",
        "tool_call_id": req.tool_call_id,
        "content": req.result,
    })
    logger.info("Tool result submitted for session %s, tool %s", req.session_id, req.tool_call_id)
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

        session = _sessions[session_id]
        yield f"data: {json.dumps({
            'type': 'session_state',
            'session_id': session_id,
            'turn_count': session['turn_count'],
            'active': session['active'],
            'messages_count': len(session['messages']),
        })}\n\n"

        import asyncio
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
