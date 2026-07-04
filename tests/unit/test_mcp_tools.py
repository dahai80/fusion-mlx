# SPDX-License-Identifier: Apache-2.0
import pytest

from fusion_mlx.mcp.tools import (
    extract_tool_calls,
    format_tool_result,
    format_tool_results,
    has_tool_calls,
    merge_tools,
    mcp_tool_to_openai,
    mcp_tools_to_openai,
    openai_call_to_mcp,
)
from fusion_mlx.mcp.types import MCPTool, MCPToolResult


class TestMcpToolToOpenai:
    def test_basic_conversion(self):
        tool = MCPTool(
            server_name="srv",
            name="add",
            description="Add numbers",
            input_schema={"type": "object", "properties": {}},
        )
        result = mcp_tool_to_openai(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "srv__add"
        assert result["function"]["description"] == "Add numbers"
        assert result["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_empty_schema_gets_default(self):
        tool = MCPTool(
            server_name="srv",
            name="noop",
            description="No-op",
            input_schema={},
        )
        result = mcp_tool_to_openai(tool)
        params = result["function"]["parameters"]
        assert params["type"] == "object"
        assert params["properties"] == {}

    def test_none_schema_gets_default(self):
        tool = MCPTool(
            server_name="srv",
            name="noop",
            description="No-op",
            input_schema=None,
        )
        result = mcp_tool_to_openai(tool)
        params = result["function"]["parameters"]
        assert params["type"] == "object"
        assert params["properties"] == {}


class TestMcpToolsToOpenai:
    def test_converts_list(self):
        tools = [
            MCPTool(server_name="a", name="x", description="X"),
            MCPTool(server_name="b", name="y", description="Y"),
        ]
        result = mcp_tools_to_openai(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a__x"
        assert result[1]["function"]["name"] == "b__y"


class TestOpenaiCallToMcp:
    def test_namespaced_name(self):
        tool_call = {
            "function": {
                "name": "srv__add",
                "arguments": '{"a": 1}',
            }
        }
        server, tool, args = openai_call_to_mcp(tool_call)
        assert server == "srv"
        assert tool == "add"
        assert args == {"a": 1}

    def test_non_namespaced_name(self):
        tool_call = {
            "function": {
                "name": "simple_tool",
                "arguments": "{}",
            }
        }
        server, tool, args = openai_call_to_mcp(tool_call)
        assert server == ""
        assert tool == "simple_tool"

    def test_invalid_json_arguments(self):
        tool_call = {
            "function": {
                "name": "srv__add",
                "arguments": "not json",
            }
        }
        server, tool, args = openai_call_to_mcp(tool_call)
        assert args == {}

    def test_dict_arguments(self):
        tool_call = {
            "function": {
                "name": "srv__add",
                "arguments": {"a": 1},
            }
        }
        server, tool, args = openai_call_to_mcp(tool_call)
        assert args == {"a": 1}

    def test_none_arguments(self):
        tool_call = {
            "function": {
                "name": "srv__add",
                "arguments": None,
            }
        }
        server, tool, args = openai_call_to_mcp(tool_call)
        assert args == {}

    def test_missing_function_key(self):
        server, tool, args = openai_call_to_mcp({})
        assert server == ""
        assert tool == ""
        assert args == {}


class TestFormatToolResult:
    def test_string_content(self):
        result = MCPToolResult(tool_name="t", content="hello")
        msg = format_tool_result(result, "call_1")
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_1"
        assert msg["content"] == "hello"

    def test_error_result(self):
        result = MCPToolResult(
            tool_name="t",
            content=None,
            is_error=True,
            error_message="timeout",
        )
        msg = format_tool_result(result, "call_2")
        assert msg["content"] == "Error: timeout"

    def test_non_string_content(self):
        result = MCPToolResult(tool_name="t", content={"key": "value"})
        msg = format_tool_result(result, "call_3")
        import json
        assert json.loads(msg["content"]) == {"key": "value"}


class TestFormatToolResults:
    def test_multiple_results(self):
        results = [
            (MCPToolResult(tool_name="t1", content="a"), "id1"),
            (MCPToolResult(tool_name="t2", content="b"), "id2"),
        ]
        msgs = format_tool_results(results)
        assert len(msgs) == 2
        assert msgs[0]["tool_call_id"] == "id1"
        assert msgs[1]["tool_call_id"] == "id2"


class TestMergeTools:
    def test_mcp_only(self):
        tools = [
            MCPTool(server_name="s", name="x", description="X"),
        ]
        result = merge_tools(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "s__x"

    def test_user_tools_override(self):
        mcp_tools = [
            MCPTool(server_name="s", name="x", description="MCP X"),
        ]
        user_tools = [
            {
                "type": "function",
                "function": {
                    "name": "s__x",
                    "description": "User X",
                    "parameters": {},
                },
            }
        ]
        result = merge_tools(mcp_tools, user_tools)
        assert len(result) == 1
        assert result[0]["function"]["description"] == "User X"

    def test_combined_no_overlap(self):
        mcp_tools = [
            MCPTool(server_name="s", name="x", description="MCP X"),
        ]
        user_tools = [
            {
                "type": "function",
                "function": {
                    "name": "custom_y",
                    "description": "User Y",
                    "parameters": {},
                },
            }
        ]
        result = merge_tools(mcp_tools, user_tools)
        assert len(result) == 2

    def test_none_user_tools(self):
        mcp_tools = [
            MCPTool(server_name="s", name="x", description="X"),
        ]
        result = merge_tools(mcp_tools, None)
        assert len(result) == 1


class TestExtractToolCalls:
    def test_with_tool_calls(self):
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "tc_1", "function": {"name": "a", "arguments": "{}"}}
                        ]
                    }
                }
            ]
        }
        calls = extract_tool_calls(response)
        assert len(calls) == 1
        assert calls[0]["id"] == "tc_1"

    def test_no_tool_calls(self):
        response = {"choices": [{"message": {"content": "hello"}}]}
        calls = extract_tool_calls(response)
        assert calls == []

    def test_no_choices(self):
        calls = extract_tool_calls({})
        assert calls == []


class TestHasToolCalls:
    def test_true(self):
        response = {
            "choices": [
                {"message": {"tool_calls": [{"id": "tc_1"}]}}
            ]
        }
        assert has_tool_calls(response) is True

    def test_false(self):
        response = {"choices": [{"message": {"content": "hi"}}]}
        assert has_tool_calls(response) is False
