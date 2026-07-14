"""Canvas API routes for the Agent Studio visual editor.

Provides the backend for the drag-and-drop agent graph canvas:
- GET /admin/canvas — render the canvas page
- GET /admin/api/canvas/graphs — list saved graphs
- POST /admin/api/canvas/graphs — save a graph
- GET /admin/api/canvas/graphs/{id} — load a graph
- PUT /admin/api/canvas/graphs/{id} — update a graph
- DELETE /admin/api/canvas/graphs/{id} — delete a graph
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin-canvas"])

# In-memory graph store (shared with agent_routes in production)
_graphs: dict[str, dict[str, Any]] = {}

# Templates and static dirs are set by the parent admin router
_templates = None


def init_canvas(templates) -> None:
    """Inject the Jinja2 templates instance from the admin router."""
    global _templates
    _templates = templates


# ── Page Route ──


@router.get("/canvas", response_class=HTMLResponse, include_in_schema=False)
async def canvas_page(request: Request) -> HTMLResponse:
    """Render the agent graph canvas editor page."""
    if _templates is None:
        return HTMLResponse("<h1>Canvas not initialized</h1>", status_code=500)
    return _templates.TemplateResponse(
        "canvas.html",
        {"request": request, "graphs": list(_graphs.values())},
    )


# ── API Routes ──


@router.get("/api/canvas/graphs")
async def list_canvas_graphs() -> list[dict[str, Any]]:
    """List all saved canvas graphs."""
    result = []
    for gid, g in _graphs.items():
        result.append({
            "id": gid,
            "name": g.get("name", "Untitled"),
            "description": g.get("description", ""),
            "node_count": len(g.get("nodes", {})),
            "edge_count": len(g.get("edges", [])),
            "updated_at": g.get("updated_at", 0),
        })
    result.sort(key=lambda x: x["updated_at"], reverse=True)
    return result


@router.post("/api/canvas/graphs")
async def save_canvas_graph(data: dict[str, Any]) -> dict[str, str]:
    """Save a new canvas graph."""
    graph_id = data.get("id") or uuid.uuid4().hex[:16]
    if graph_id in _graphs:
        raise HTTPException(409, detail=f"Graph '{graph_id}' already exists")

    now = time.time()
    _graphs[graph_id] = {
        **data,
        "id": graph_id,
        "created_at": now,
        "updated_at": now,
    }
    logger.info("Saved canvas graph %s: %s", graph_id, data.get("name", "Untitled"))
    return {"id": graph_id, "status": "saved"}


@router.get("/api/canvas/graphs/{graph_id}")
async def load_canvas_graph(graph_id: str) -> dict[str, Any]:
    """Load a canvas graph by ID."""
    graph = _graphs.get(graph_id)
    if graph is None:
        raise HTTPException(404, detail=f"Graph '{graph_id}' not found")
    return graph


@router.put("/api/canvas/graphs/{graph_id}")
async def update_canvas_graph(graph_id: str, data: dict[str, Any]) -> dict[str, str]:
    """Update an existing canvas graph."""
    if graph_id not in _graphs:
        raise HTTPException(404, detail=f"Graph '{graph_id}' not found")

    _graphs[graph_id] = {
        **_graphs[graph_id],
        **data,
        "id": graph_id,
        "updated_at": time.time(),
    }
    return {"id": graph_id, "status": "updated"}


@router.delete("/api/canvas/graphs/{graph_id}")
async def delete_canvas_graph(graph_id: str) -> dict[str, str]:
    """Delete a canvas graph."""
    if graph_id not in _graphs:
        raise HTTPException(404, detail=f"Graph '{graph_id}' not found")
    del _graphs[graph_id]
    return {"id": graph_id, "status": "deleted"}