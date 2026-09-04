# SPDX-License-Identifier: Apache-2.0
"""
MCP (Model Context Protocol) API routes.

This module provides FastAPI routes for MCP tool management:
- GET /v1/mcp/tools - List available MCP tools
- GET /v1/mcp/servers - List MCP server status
- POST /v1/mcp/execute - Execute an MCP tool
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException

logger = logging.getLogger(__name__)

from ..admin.auth import require_admin
from ..mcp.security import MCPSecurityError, get_sandbox
from .openai_models import (
    MCPExecuteRequest,
    MCPExecuteResponse,
    MCPServerInfo,
    MCPServersResponse,
    MCPToolInfo,
    MCPToolsResponse,
)

router = APIRouter(prefix="/v1/mcp", tags=["mcp"])


# Callback function to get MCP manager (set by server.py)
_get_mcp_manager = None


def set_mcp_manager_getter(getter):
    """
    Set the callback function to get MCP manager.

    Args:
        getter: A callable that returns the MCP manager instance or None
    """
    global _get_mcp_manager
    _get_mcp_manager = getter


def _get_manager():
    """Get the MCP manager instance."""
    if _get_mcp_manager is None:
        return None
    return _get_mcp_manager()


@router.get("/tools")
async def list_mcp_tools(
    _auth: bool = Depends(require_admin),
) -> MCPToolsResponse:
    """List all available MCP tools."""
    manager = _get_manager()
    if manager is None:
        return MCPToolsResponse(tools=[], count=0)

    tools = []
    for tool in manager.get_all_tools():
        tools.append(
            MCPToolInfo(
                name=tool.full_name,
                description=tool.description,
                server=tool.server_name,
                parameters=tool.input_schema,
            )
        )

    return MCPToolsResponse(tools=tools, count=len(tools))


@router.get("/servers")
async def list_mcp_servers(
    _auth: bool = Depends(require_admin),
) -> MCPServersResponse:
    """Get status of all MCP servers."""
    manager = _get_manager()
    if manager is None:
        return MCPServersResponse(servers=[])

    servers = []
    for status in manager.get_server_status():
        servers.append(
            MCPServerInfo(
                name=status.name,
                state=status.state.value,
                transport=status.transport.value,
                tools_count=status.tools_count,
                error=status.error,
            )
        )

    return MCPServersResponse(servers=servers)


@router.post("/execute")
async def execute_mcp_tool(
    request: MCPExecuteRequest,
    _is_admin: bool = Depends(require_admin),
) -> MCPExecuteResponse:
    """Execute an MCP tool."""
    manager = _get_manager()
    if manager is None:
        raise HTTPException(
            status_code=503, detail="MCP not configured. Start server with --mcp-config"
        )

    tool_name = request.tool_name
    server_name = (
        tool_name.split("__")[0] if "__" in tool_name else "unknown"
    )
    bare_tool = tool_name.split("__")[-1] if "__" in tool_name else tool_name

    sandbox = get_sandbox()
    try:
        sandbox.validate_tool_execution(bare_tool, server_name, request.arguments)
    except MCPSecurityError as e:
        logger.warning("mcp/execute blocked by sandbox: %s", e)
        sandbox.record_execution(
            bare_tool,
            server_name,
            request.arguments,
            success=False,
            error_message=str(e),
        )
        raise HTTPException(status_code=403, detail=str(e))

    start_time = time.time()
    result = await manager.execute_tool(tool_name, request.arguments)
    execution_time_ms = (time.time() - start_time) * 1000
    sandbox.record_execution(
        bare_tool,
        server_name,
        request.arguments,
        success=not result.is_error,
        error_message=result.error_message,
        execution_time_ms=execution_time_ms,
    )

    return MCPExecuteResponse(
        tool_name=result.tool_name,
        content=result.content,
        is_error=result.is_error,
        error_message=result.error_message,
    )
