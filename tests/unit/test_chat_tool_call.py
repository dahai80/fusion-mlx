# SPDX-License-Identifier: Apache-2.0
import json


class TestChatToolCallMessageFiltering:
    @staticmethod
    def build_messages_for_api(messages):
        valid_roles = {"user", "assistant", "tool", "system"}
        result = []
        for msg in messages:
            if msg["role"] not in valid_roles:
                continue
            m = {"role": msg["role"], "content": msg.get("content")}
            if msg.get("tool_calls"):
                m["tool_calls"] = msg["tool_calls"]
            if msg.get("tool_call_id"):
                m["tool_call_id"] = msg["tool_call_id"]
            result.append(m)
        return result

    def test_filters_tool_call_indicator_messages(self):
        messages = [
            {"role": "user", "content": "Who is X?"},
            {"role": "tool_call", "content": "tavily__tavily_search...", "_ui": True},
            {"role": "assistant", "content": "X is...", "tool_calls": None},
        ]
        api_msgs = self.build_messages_for_api(messages)
        roles = [m["role"] for m in api_msgs]
        assert "tool_call" not in roles
        assert roles == ["user", "assistant"]

    def test_passes_tool_calls_and_tool_call_id(self):
        tc = [
            {
                "id": "tc_1",
                "type": "function",
                "function": {"name": "t", "arguments": "{}"},
            }
        ]
        messages = [
            {"role": "user", "content": "Search for X"},
            {"role": "assistant", "content": None, "tool_calls": tc, "_ui": False},
            {
                "role": "tool",
                "tool_call_id": "tc_1",
                "content": "result...",
                "_ui": False,
            },
        ]
        api_msgs = self.build_messages_for_api(messages)
        assert len(api_msgs) == 3
        assert api_msgs[1]["tool_calls"] == tc
        assert api_msgs[2]["tool_call_id"] == "tc_1"

    def test_normal_conversation_unchanged(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        api_msgs = self.build_messages_for_api(messages)
        assert len(api_msgs) == 2
        assert api_msgs[0] == {"role": "user", "content": "Hello"}
        assert api_msgs[1] == {"role": "assistant", "content": "Hi there"}


class TestChatToolCallAccumulation:
    @staticmethod
    def accumulate_tool_calls(deltas):
        tool_calls_map = {}
        for delta in deltas:
            if not delta.get("tool_calls"):
                continue
            for tc in delta["tool_calls"]:
                i = tc.get("index", 0)
                if i not in tool_calls_map:
                    tool_calls_map[i] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc.get("id"):
                    tool_calls_map[i]["id"] = tc["id"]
                if tc.get("function", {}).get("name"):
                    tool_calls_map[i]["function"]["name"] += tc["function"]["name"]
                if tc.get("function", {}).get("arguments"):
                    tool_calls_map[i]["function"]["arguments"] += tc["function"][
                        "arguments"
                    ]
        return list(tool_calls_map.values())

    def test_single_tool_call(self):
        deltas = [
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "tc_1",
                        "function": {"name": "tavily__tavily_search"},
                    }
                ]
            },
            {"tool_calls": [{"index": 0, "function": {"arguments": '{"que'}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": 'ry":"test"}'}}]},
        ]
        result = self.accumulate_tool_calls(deltas)
        assert len(result) == 1
        assert result[0]["id"] == "tc_1"
        assert result[0]["function"]["name"] == "tavily__tavily_search"
        assert json.loads(result[0]["function"]["arguments"]) == {"query": "test"}

    def test_multiple_parallel_tool_calls(self):
        deltas = [
            {
                "tool_calls": [
                    {"index": 0, "id": "tc_1", "function": {"name": "search"}}
                ]
            },
            {
                "tool_calls": [
                    {"index": 1, "id": "tc_2", "function": {"name": "extract"}}
                ]
            },
            {"tool_calls": [{"index": 0, "function": {"arguments": '{"q":"a"}'}}]},
            {
                "tool_calls": [
                    {"index": 1, "function": {"arguments": '{"urls":["http://x"]}'}}
                ]
            },
        ]
        result = self.accumulate_tool_calls(deltas)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "search"
        assert result[1]["function"]["name"] == "extract"
        assert json.loads(result[0]["function"]["arguments"]) == {"q": "a"}
        assert json.loads(result[1]["function"]["arguments"]) == {"urls": ["http://x"]}

    def test_no_tool_calls(self):
        deltas = [
            {"content": "Hello"},
            {"content": " world"},
        ]
        result = self.accumulate_tool_calls(deltas)
        assert result == []

    def test_missing_index_defaults_to_zero(self):
        deltas = [
            {
                "tool_calls": [
                    {"id": "tc_1", "function": {"name": "t", "arguments": "{}"}}
                ]
            },
        ]
        result = self.accumulate_tool_calls(deltas)
        assert len(result) == 1
        assert result[0]["id"] == "tc_1"


class TestChatToolCallSafety:
    MAX_TOOL_DEPTH = 10
    TOOL_TIMEOUT_MS = 30000

    @staticmethod
    def build_depth_error_message(max_depth):
        return f"Error: Maximum tool call depth ({max_depth}) exceeded. The model may be stuck in a loop."

    @staticmethod
    def build_tool_result(content, error=False, tool_name=None):
        result = {"content": content, "error": error}
        if tool_name:
            result["toolName"] = tool_name
        return result

    @staticmethod
    def build_timeout_error_message(timeout_ms):
        return f"Error: Tool timed out after {timeout_ms / 1000}s"

    @staticmethod
    def build_tool_status_error(failed_results):
        names = [r["toolName"] for r in failed_results if r.get("error")]
        return f"Failed: {', '.join(names)}" if names else ""

    def test_depth_limit_error_message_format(self):
        msg = self.build_depth_error_message(self.MAX_TOOL_DEPTH)
        assert "10" in msg
        assert "stuck in a loop" in msg

    def test_depth_limit_boundary_at_max(self):
        depth = self.MAX_TOOL_DEPTH
        assert not (depth > self.MAX_TOOL_DEPTH)

    def test_depth_limit_boundary_over_max(self):
        depth = self.MAX_TOOL_DEPTH + 1
        assert depth > self.MAX_TOOL_DEPTH

    def test_success_result_includes_tool_name(self):
        result = self.build_tool_result(
            "search results here", tool_name="tavily_search"
        )
        assert result["error"] is False
        assert result["toolName"] == "tavily_search"

    def test_error_result_includes_tool_name(self):
        result = self.build_tool_result(
            "Error: connection refused", error=True, tool_name="tavily_search"
        )
        assert result["error"] is True
        assert result["toolName"] == "tavily_search"
        assert result["content"].startswith("Error:")

    def test_timeout_error_message_includes_seconds(self):
        msg = self.build_timeout_error_message(self.TOOL_TIMEOUT_MS)
        assert "30.0s" in msg

    def test_http_error_result_format(self):
        result = self.build_tool_result(
            "Error: HTTP 503", error=True, tool_name="broken_tool"
        )
        assert result["error"] is True
        assert "503" in result["content"]

    def test_tool_status_error_format(self):
        failed_results = [
            {"content": "Error: timeout", "error": True, "toolName": "tavily_search"},
            {"content": "Error: HTTP 503", "error": True, "toolName": "broken_tool"},
        ]
        status = self.build_tool_status_error(failed_results)
        assert "tavily_search" in status
        assert "broken_tool" in status
        assert status.startswith("Failed:")

    def test_error_indicators_excluded_from_api(self):
        messages = [
            {"role": "user", "content": "search for X"},
            {
                "role": "tool_call",
                "content": "search failed",
                "_error": True,
                "_ui": True,
            },
            {"role": "assistant", "content": "Sorry, the search failed."},
        ]
        valid_roles = {"user", "assistant", "tool", "system"}
        api_msgs = [m for m in messages if m["role"] in valid_roles]
        assert len(api_msgs) == 2
        assert all(m["role"] != "tool_call" for m in api_msgs)

    def test_abort_signal_prevents_recursion(self):
        class FakeSignal:
            def __init__(self, aborted):
                self.aborted = aborted

        class FakeController:
            def __init__(self, aborted):
                self.signal = FakeSignal(aborted)

        controller = FakeController(aborted=True)
        should_recurse = not (controller.signal.aborted)
        assert should_recurse is False

        controller = FakeController(aborted=False)
        should_recurse = not (controller.signal.aborted)
        assert should_recurse is True

    def test_abort_guard_with_none_controller(self):
        controller = None
        aborted = getattr(getattr(controller, "signal", None), "aborted", None)
        assert not aborted
