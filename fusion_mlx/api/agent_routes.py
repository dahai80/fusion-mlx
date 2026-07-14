"""Agent Graph API routes for fusion-mlx.

Provides endpoints for managing and executing agent graphs:
- /v1/agents/graphs      — CRUD for agent workflow graphs
- /v1/agents/run         — Execute a graph against a loaded model
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/agents", tags=["agents"])

# In-memory graph store (SQLite-backed in production)
_graphs: dict[str, dict[str, Any]] = {}

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
            errors.append(f"Edge {i}: source_id '{edge.get('source_id')}' not found in nodes")
        if edge.get("target_id") not in nodes:
            errors.append(f"Edge {i}: target_id '{edge.get('target_id')}' not found in nodes")
    return errors


# ── Graph CRUD ──


@router.get("/graphs")
async def list_graphs() -> list[dict[str, Any]]:
    """List all saved agent graphs with metadata."""
    result = []
    for gid, g in _graphs.items():
        result.append({
            "id": gid,
            "name": g.get("name", ""),
            "description": g.get("description", ""),
            "version": g.get("version", "1.0"),
            "node_count": len(g.get("nodes", {})),
            "edge_count": len(g.get("edges", [])),
            "created_at": g.get("created_at", 0),
            "updated_at": g.get("updated_at", 0),
        })
    result.sort(key=lambda x: x["updated_at"], reverse=True)
    return result


@router.post("/graphs")
async def create_graph(data: dict[str, Any]) -> dict[str, Any]:
    """Create a new agent graph."""
    errors = _validate_graph(data)
    if errors:
        raise HTTPException(400, detail="; ".join(errors))

    graph_id = data.get("id") or uuid.uuid4().hex[:16]
    if graph_id in _graphs:
        raise HTTPException(409, detail=f"Graph '{graph_id}' already exists")

    now = _now()
    _graphs[graph_id] = {
        **data,
        "id": graph_id,
        "created_at": now,
        "updated_at": now,
    }
    logger.info("Created agent graph %s: %s", graph_id, data.get("name", ""))
    return {"id": graph_id, "status": "created"}


@router.get("/graphs/{graph_id}")
async def get_graph(graph_id: str) -> dict[str, Any]:
    """Get an agent graph by ID."""
    graph = _graphs.get(graph_id)
    if graph is None:
        raise HTTPException(404, detail=f"Graph '{graph_id}' not found")
    return graph


@router.put("/graphs/{graph_id}")
async def update_graph(graph_id: str, data: dict[str, Any]) -> dict[str, Any]:
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
    logger.info("Updated agent graph %s", graph_id)
    return {"id": graph_id, "status": "updated"}


@router.delete("/graphs/{graph_id}")
async def delete_graph(graph_id: str) -> dict[str, str]:
    """Delete an agent graph."""
    if graph_id not in _graphs:
        raise HTTPException(404, detail=f"Graph '{graph_id}' not found")
    del _graphs[graph_id]
    logger.info("Deleted agent graph %s", graph_id)
    return {"id": graph_id, "status": "deleted"}


# ── Graph Execution ──


@router.post("/graphs/{graph_id}/export")
async def export_graph(graph_id: str, fmt: str = "json") -> dict[str, Any]:
    """Export an agent graph in the specified format."""
    graph = _graphs.get(graph_id)
    if graph is None:
        raise HTTPException(404, detail=f"Graph '{graph_id}' not found")

    if fmt == "json":
        return {"format": "json", "data": graph}
    elif fmt == "python":
        # Generate a simple Python script representation
        py_code = _generate_python_script(graph)
        return {"format": "python", "data": py_code}
    else:
        raise HTTPException(400, detail=f"Unsupported format: {fmt}")


@router.post("/run")
async def run_graph(body: dict[str, Any]) -> dict[str, Any]:
    """Execute an agent graph against fusion-mlx's loaded model.

    This endpoint reads the graph's first LLM node configuration and
    calls /v1/chat/completions internally. It does NOT load or manage
    models — that must be done separately via ``fusion-mlx serve``.

    Request body:
    ```json
    {
        "graph_id": "...",
        "input": "User message",
        "model": "optional-model-override",
        "max_tokens": 4096,
        "temperature": 0.7
    }
    ```
    """
    graph_id = body.get("graph_id", "")
    graph = _graphs.get(graph_id)
    if graph is None:
        raise HTTPException(404, detail=f"Graph '{graph_id}' not found")

    # Find the first LLM node to get model config
    llm_node = _find_first_llm_node(graph)
    if llm_node is None:
        raise HTTPException(400, detail="Graph has no LLM node configured")

    model = body.get("model") or llm_node.get("model", "")
    if not model:
        raise HTTPException(400, detail="No model specified and no model in graph")

    # Build messages
    system_prompt = body.get("system_prompt") or llm_node.get("system_prompt", "")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": body.get("input", "")})

    # We return the execution plan; actual execution requires a running
    # fusion-mlx server. The client is expected to call /v1/chat/completions
    # with the returned messages.
    return {
        "graph_id": graph_id,
        "graph_name": graph.get("name", ""),
        "model": model,
        "messages": messages,
        "temperature": body.get("temperature", llm_node.get("temperature", 0.7)),
        "max_tokens": body.get("max_tokens", llm_node.get("max_tokens", 4096)),
        "status": "ready",
        "note": "Execute by sending these messages to /v1/chat/completions",
    }


def _find_first_llm_node(graph: dict) -> dict[str, Any] | None:
    """Find the first LLM node in the graph."""
    nodes = graph.get("nodes", {})
    for nid, node in nodes.items():
        if isinstance(node, dict) and node.get("type") == "llm":
            return node
    return None


def _generate_python_script(graph: dict) -> str:
    """Generate a simple Python script to execute the graph."""
    name = graph.get("name", "untitled")
    llm = _find_first_llm_node(graph) or {}
    model = llm.get("model", "qwen3.5-9b")
    system_prompt = llm.get("system_prompt", "")
    temperature = llm.get("temperature", 0.7)

    lines = [
        '#!/usr/bin/env python3',
        f'"""Auto-generated agent: {name}"""',
        '',
        'import httpx',
        'import asyncio',
        '',
        '',
        'async def main():',
        '    client = httpx.AsyncClient(base_url="http://localhost:8000/v1", timeout=120.0)',
        '    try:',
        f'        messages = [{{"role": "system", "content": "{system_prompt}"}}]' if system_prompt else '        messages = []',
        '        messages.append({"role": "user", "content": input("Enter your input: ")})',
        '',
        '        resp = await client.post("/chat/completions", json={',
        f'            "model": "{model}",',
        '            "messages": messages,',
        f'            "temperature": {temperature},',
        '            "max_tokens": 4096,',
        '        })',
        '        resp.raise_for_status()',
        '        data = resp.json()',
        '        print(data["choices"][0]["message"]["content"])',
        '    finally:',
        '        await client.aclose()',
        '',
        '',
        'if __name__ == "__main__":',
        '    asyncio.run(main())',
        '',
    ]
    return "\n".join(lines)