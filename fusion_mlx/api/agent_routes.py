"""Agent Graph API routes for fusion-mlx.

Provides endpoints for managing and executing agent graphs:
- /v1/agents/graphs      — CRUD for agent workflow graphs
- /v1/agents/run         — Execute a graph against a loaded model
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException

from ..admin.auth import require_admin
from ..agents.governance import GovernorLimitExceeded, get_governor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/agents", tags=["agents"])

# In-memory graph store (SQLite-backed in production)
# E-48 (#811): admin-gated but was unbounded — a single long-running
# deployment accumulated graphs without limit. Cap the in-memory store;
# once full, the oldest graph (FIFO, dict preserves insertion order) is
# evicted to make room. Env-configurable for power users.
MAX_GRAPHS = max(8, int(os.environ.get("FUSION_MLX_MAX_AGENT_GRAPHS", "128") or 128))
_graphs: dict[str, dict[str, Any]] = {}

# FC-7 (#0907 audit): persist agent graphs to disk so a server restart
# (or LRU eviction under memory pressure) does not silently evaporate
# operator-built agent configurations. Atomic write (temp + os.replace);
# loaded once at module import.
_GRAPHS_FILE = os.path.expanduser("~/.fusion-mlx/agent_graphs.json")


def _persist_graphs() -> None:
    try:
        os.makedirs(os.path.dirname(_GRAPHS_FILE), exist_ok=True)
        tmp = _GRAPHS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_graphs, f, ensure_ascii=False)
        os.replace(tmp, _GRAPHS_FILE)
    except Exception as e:
        logger.error("FC-7: failed to persist agent graphs to %s: %s", _GRAPHS_FILE, e)


def _load_graphs() -> None:
    try:
        with open(_GRAPHS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(
                "FC-7: agent graph file %s is not a JSON object; ignoring", _GRAPHS_FILE
            )
            return
        loaded = 0
        for gid, g in data.items():
            if not isinstance(g, dict):
                continue
            if len(_graphs) >= MAX_GRAPHS:
                logger.warning(
                    "FC-7: loaded graphs exceed MAX_GRAPHS=%d; %d graph(s) dropped",
                    MAX_GRAPHS,
                    len(data) - loaded,
                )
                break
            _graphs[gid] = g
            loaded += 1
        logger.info("FC-7: loaded %d agent graph(s) from %s", loaded, _GRAPHS_FILE)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("FC-7: failed to load agent graphs from %s: %s", _GRAPHS_FILE, e)


_load_graphs()

# ── Helper ──


def _now() -> float:
    return time.time()


def _validate_graph(data: dict) -> list[str]:
    """Validate a graph data structure. Returns list of errors."""
    errors: list[str] = []
    if not isinstance(data, dict):
        errors.append("Graph must be a JSON object")
        return errors
    nodes = data.get("nodes", {})
    if not isinstance(nodes, dict):
        errors.append("'nodes' must be an object")
        return errors
    if not nodes:
        errors.append("Graph must have at least one node")
    edges = data.get("edges", [])
    if not isinstance(edges, list):
        errors.append("'edges' must be an array")
        return errors
    start_node = data.get("start_node_id", "")
    if start_node and start_node not in nodes:
        errors.append(f"start_node_id '{start_node}' not found in nodes")
    # Validate all edge references
    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"Edge {i} must be an object")
            continue
        if edge.get("source_id") not in nodes:
            errors.append(
                f"Edge {i}: source_id '{edge.get('source_id')}' not found in nodes"
            )
        if edge.get("target_id") not in nodes:
            errors.append(
                f"Edge {i}: target_id '{edge.get('target_id')}' not found in nodes"
            )
    return errors


# ── Graph CRUD ──


@router.get("/graphs")
async def list_graphs(
    _is_admin: bool = Depends(require_admin),
) -> list[dict[str, Any]]:
    """List all saved agent graphs with metadata."""
    result = []
    for gid, g in _graphs.items():
        result.append(
            {
                "id": gid,
                "name": g.get("name", ""),
                "description": g.get("description", ""),
                "version": g.get("version", "1.0"),
                "node_count": len(g.get("nodes", {})),
                "edge_count": len(g.get("edges", [])),
                "created_at": g.get("created_at", 0),
                "updated_at": g.get("updated_at", 0),
            }
        )
    result.sort(key=lambda x: x["updated_at"], reverse=True)
    return result


@router.post("/graphs")
async def create_graph(
    data: dict[str, Any],
    _is_admin: bool = Depends(require_admin),
) -> dict[str, Any]:
    """Create a new agent graph."""
    errors = _validate_graph(data)
    if errors:
        raise HTTPException(400, detail="; ".join(errors))

    graph_id = data.get("id") or uuid.uuid4().hex[:16]
    if graph_id in _graphs:
        raise HTTPException(409, detail=f"Graph '{graph_id}' already exists")

    # E-48 (#811): enforce the in-memory graph cap. Evict the oldest
    # (FIFO — dict preserves insertion order) before inserting so the
    # store cannot grow without bound on a long-running deployment.
    while len(_graphs) >= MAX_GRAPHS:
        oldest_id = next(iter(_graphs))
        evicted = _graphs.pop(oldest_id, None)
        if evicted is not None:
            logger.warning(
                "Agent graph store full (MAX_GRAPHS=%d); evicted oldest "
                "graph %s to make room for %s",
                MAX_GRAPHS,
                oldest_id,
                graph_id,
            )
        else:
            break

    now = _now()
    _graphs[graph_id] = {
        **data,
        "id": graph_id,
        "created_at": now,
        "updated_at": now,
    }
    _persist_graphs()
    logger.info("Created agent graph %s: %s", graph_id, data.get("name", ""))
    return {"id": graph_id, "status": "created"}


@router.get("/graphs/{graph_id}")
async def get_graph(
    graph_id: str,
    _is_admin: bool = Depends(require_admin),
) -> dict[str, Any]:
    """Get an agent graph by ID."""
    graph = _graphs.get(graph_id)
    if graph is None:
        raise HTTPException(404, detail=f"Graph '{graph_id}' not found")
    return graph


@router.put("/graphs/{graph_id}")
async def update_graph(
    graph_id: str,
    data: dict[str, Any],
    _is_admin: bool = Depends(require_admin),
) -> dict[str, Any]:
    """Update an existing agent graph."""
    if graph_id not in _graphs:
        raise HTTPException(404, detail=f"Graph '{graph_id}' not found")

    errors = _validate_graph(data)
    if errors:
        raise HTTPException(400, detail="; ".join(errors))

    now = _now()
    _graphs[graph_id] = {
        **_graphs[graph_id],
        **data,
        "id": graph_id,
        "updated_at": now,
    }
    _persist_graphs()
    logger.info("Updated agent graph %s", graph_id)
    return {"id": graph_id, "status": "updated"}


@router.delete("/graphs/{graph_id}")
async def delete_graph(
    graph_id: str,
    _is_admin: bool = Depends(require_admin),
) -> dict[str, str]:
    """Delete an agent graph."""
    if graph_id not in _graphs:
        raise HTTPException(404, detail=f"Graph '{graph_id}' not found")
    del _graphs[graph_id]
    _persist_graphs()
    logger.info("Deleted agent graph %s", graph_id)
    return {"id": graph_id, "status": "deleted"}


# ── Graph Execution ──


@router.post("/graphs/{graph_id}/export")
async def export_graph(
    graph_id: str,
    fmt: str = "json",
    _is_admin: bool = Depends(require_admin),
) -> dict[str, Any]:
    """Export an agent graph in the specified format."""
    graph = _graphs.get(graph_id)
    if graph is None:
        raise HTTPException(404, detail=f"Graph '{graph_id}' not found")

    if fmt == "json":
        return {"format": "json", "data": graph}
    elif fmt == "python":
        # Generate a simple Python script representation
        py_code = _generate_python_script(graph)
        logger.info("Exported agent graph %s as executable python script", graph_id)
        return {"format": "python", "data": py_code}
    else:
        raise HTTPException(400, detail=f"Unsupported format: {fmt}")


def _build_run_plan(
    body: dict[str, Any], graph: dict[str, Any]
) -> dict[str, Any] | None:
    # Resolve graph + first LLM node into a chat-completions request plan.
    graph_id = body.get("graph_id", "")
    llm_node = _find_first_llm_node(graph)
    if llm_node is None:
        return None
    model = body.get("model") or llm_node.get("model", "")
    if not model:
        return None
    system_prompt = body.get("system_prompt") or llm_node.get("system_prompt", "")
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": body.get("input", "")})
    temperature = body.get("temperature", llm_node.get("temperature", 0.7))
    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        temperature = 0.7
    max_tokens = body.get("max_tokens", llm_node.get("max_tokens", 4096))
    return {
        "graph_id": graph_id,
        "graph_name": graph.get("name", ""),
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


@router.post("/plan")
async def plan_graph(
    body: dict[str, Any],
    _is_admin: bool = Depends(require_admin),
) -> dict[str, Any]:
    # FC-8 (#0907 audit): /plan honestly returns the chat-completions
    # request the graph resolves to, without executing it. Use this when
    # you want to inspect or replay the request yourself.
    graph_id = body.get("graph_id", "")
    graph = _graphs.get(graph_id)
    if graph is None:
        raise HTTPException(404, detail=f"Graph '{graph_id}' not found")
    plan = _build_run_plan(body, graph)
    if plan is None:
        raise HTTPException(400, detail="Graph has no LLM node or model configured")
    plan["status"] = "ready"
    plan["note"] = (
        "Execute by sending these messages to /v1/chat/completions, or POST /v1/agents/run"
    )
    return plan


@router.post("/run")
async def run_graph(
    body: dict[str, Any],
    _is_admin: bool = Depends(require_admin),
) -> dict[str, Any]:
    """Execute an agent graph against fusion-mlx's loaded model.

    FC-8 (#0907 audit): this endpoint now actually executes the graph by
    calling the local server's /v1/chat/completions (it previously only
    returned a plan while the route name claimed execution). The graph's
    first LLM node provides model/temperature/max_tokens; the request body
    may override them. Models must already be loaded via ``fusion-mlx serve``.

    G4 (#0910 audit): graph execution is now bounded by the agent governor
    — step-count cap, wall-clock timeout, token budget, and kill switch.
    Each run gets a run_id that can be cancelled via /v1/agents/runs/{id}/cancel.
    """
    graph_id = body.get("graph_id", "")
    graph = _graphs.get(graph_id)
    if graph is None:
        raise HTTPException(404, detail=f"Graph '{graph_id}' not found")
    plan = _build_run_plan(body, graph)
    if plan is None:
        raise HTTPException(400, detail="Graph has no LLM node or model configured")

    gov = get_governor()
    run = gov.start_run(graph_id)

    api_key = os.environ.get("FUSION_MLX_API_KEY", "")
    base_url = os.environ.get("FUSION_HOST", "http://127.0.0.1:11434").rstrip("/")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": plan["model"],
        "messages": plan["messages"],
        "temperature": plan["temperature"],
        "max_tokens": plan["max_tokens"],
    }

    try:
        gov.check_step(run)
        async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
            resp = await client.post(
                "/v1/chat/completions", json=payload, headers=headers
            )
    except GovernorLimitExceeded as e:
        logger.warning("G4: graph run %s aborted: %s", run.run_id, e.reason)
        return {
            "run_id": run.run_id,
            "graph_id": graph_id,
            "status": e.status.value,
            "error": e.reason,
        }
    except httpx.RequestError as e:
        gov.finish(run, error=str(e))
        logger.error(
            "FC-8 /v1/agents/run: failed to reach local server %s: %s", base_url, e
        )
        raise HTTPException(503, detail=f"Local inference server unreachable: {e}")
    if resp.status_code >= 400:
        gov.finish(run, error=f"HTTP {resp.status_code}")
        logger.error(
            "FC-8 /v1/agents/run: chat completions returned %d: %s",
            resp.status_code,
            resp.text[:500],
        )
        raise HTTPException(resp.status_code, detail=resp.text[:500])

    completion = resp.json()
    gov.record_usage(run, completion.get("usage"))
    gov.finish(run)
    logger.info(
        "FC-8: executed agent graph %s via %s (run=%s steps=%d tokens=%d)",
        graph_id,
        plan["model"],
        run.run_id,
        run.steps,
        run.tokens_used,
    )
    return {
        "run_id": run.run_id,
        "graph_id": graph_id,
        "graph_name": plan["graph_name"],
        "model": plan["model"],
        "status": "completed",
        "completion": completion,
    }


@router.get("/runs")
async def list_runs(
    _is_admin: bool = Depends(require_admin),
) -> list[dict[str, Any]]:
    """G4: list active and recent graph executions."""
    return get_governor().list_runs()


@router.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    _is_admin: bool = Depends(require_admin),
) -> dict[str, Any]:
    """G4: kill switch — cancel a running graph execution."""
    ok = get_governor().cancel(run_id)
    if not ok:
        raise HTTPException(404, detail=f"Run '{run_id}' not found or not running")
    return {"run_id": run_id, "status": "cancelled"}


def _find_first_llm_node(graph: dict) -> dict[str, Any] | None:
    """Find the first LLM node in the graph."""
    nodes = graph.get("nodes", {})
    for nid, node in nodes.items():
        if isinstance(node, dict) and node.get("type") == "llm":
            return node
    return None


def _generate_python_script(graph: dict) -> str:
    """Generate a simple Python script to execute the graph."""
    llm = _find_first_llm_node(graph) or {}
    model = str(llm.get("model") or "qwen3.5-9b")
    system_prompt = str(llm.get("system_prompt") or "")
    temperature = llm.get("temperature", 0.7)
    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        temperature = 0.7
    if not (0.0 <= temperature <= 2.0):
        temperature = 0.7

    # Security: embed all untrusted graph values as a JSON literal. json.dumps
    # output is a valid Python dict literal for str/float/dict and escapes
    # quotes/newslashes/control chars, so no value can break out of the string
    # or inject a statement. Previously name/system_prompt/model were
    # f-string-interpolated into source (quote breakout) and temperature was
    # interpolated unquoted into an expression (direct code injection).
    config = json.dumps(
        {
            "name": str(graph.get("name") or "untitled"),
            "model": model,
            "system_prompt": system_prompt,
            "temperature": temperature,
        },
        ensure_ascii=True,
    )

    lines = [
        "#!/usr/bin/env python3",
        "# Auto-generated agent graph script (fusion-mlx export)",
        "",
        "import asyncio",
        "import httpx",
        "",
        f"_CONFIG = {config}",
        "",
        "",
        "async def main():",
        '    client = httpx.AsyncClient(base_url="http://localhost:11434/v1", timeout=120.0)',
        "    try:",
        "        messages = []",
        '        if _CONFIG["system_prompt"]:',
        '            messages.append({"role": "system", "content": _CONFIG["system_prompt"]})',
        '        messages.append({"role": "user", "content": input("Enter your input: ")})',
        "",
        '        resp = await client.post("/chat/completions", json={',
        '            "model": _CONFIG["model"],',
        '            "messages": messages,',
        '            "temperature": _CONFIG["temperature"],',
        '            "max_tokens": 4096,',
        "        })",
        "        resp.raise_for_status()",
        "        data = resp.json()",
        '        print(data["choices"][0]["message"]["content"])',
        "    finally:",
        "        await client.aclose()",
        "",
        "",
        'if __name__ == "__main__":',
        "    asyncio.run(main())",
        "",
    ]
    return "\n".join(lines)
