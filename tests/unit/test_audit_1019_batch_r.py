# SPDX-License-Identifier: Apache-2.0
"""Regression tests for audit 1019 L13: one Ollama stream frame walk.

/api/generate and /api/chat used to carry a ~90-line copy of the SSE frame
walk, so M20-class fixes had to land twice. Both endpoints now delegate to
``_ollama_stream_lines`` and only supply their chunk shape.
"""

import ast
import asyncio
import json
import pathlib

_REPO = pathlib.Path(__file__).resolve().parents[2]


def _route_source():
    return (_REPO / "fusion_mlx/api/ollama_routes.py").read_text()


def _endpoint(name):
    tree = ast.parse(_route_source())
    return next(
        n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == name
    )


def _calls(node, name):
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == name
    ]


class TestBothEndpointsDelegate:
    def test_generate_uses_shared_walk(self):
        assert _calls(_endpoint("api_generate"), "_ollama_stream_lines")

    def test_chat_uses_shared_walk(self):
        assert _calls(_endpoint("api_chat"), "_ollama_stream_lines")

    def test_no_second_sse_frame_walk(self):
        source = _route_source()
        assert source.count('payload == "[DONE]"') == 1
        assert source.count('chunk.startswith("data: ")') == 1


class TestSharedWalk:
    @staticmethod
    def _run(chunks, render_delta=None, render_finish=None):
        from fusion_mlx.api.ollama_routes import _ollama_stream_lines

        async def gen():
            for chunk in chunks:
                yield chunk

        async def collect():
            return [
                line
                async for line in _ollama_stream_lines(
                    gen(),
                    model="m",
                    render_delta=render_delta or (lambda delta: ()),
                    render_finish=render_finish
                    or (lambda fr, p, e: [f"done:{fr}:{p}:{e}"]),
                    log_label="/api/test",
                    request_id="req-1",
                    engine=None,
                )
            ]

        return asyncio.run(collect())

    def test_upstream_error_frame_is_forwarded_and_ends_stream(self):
        chunks = [
            'data: {"error": {"message": "engine poisoned"}}\n\n',
            'data: {"choices": [{"delta": {"content": "unreachable"}}]}\n\n',
            "data: [DONE]\n\n",
        ]
        lines = self._run(chunks)
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"error": "engine poisoned"}

    def test_finish_reason_and_usage_reach_the_finish_renderer(self):
        chunks = [
            'data: {"choices": [{"delta": {"content": "hi"}, "finish_reason": null}]}\n\n',
            'data: {"choices": [{"delta": {}, "finish_reason": "length"}],'
            ' "usage": {"prompt_tokens": 7, "completion_tokens": 3}}\n\n',
            "data: [DONE]\n\n",
        ]
        seen = []

        def render_delta(delta):
            seen.append(delta.get("content"))
            return ()

        lines = self._run(chunks, render_delta=render_delta)
        assert seen == ["hi", None]
        assert lines == ["done:length:7:3"]

    def test_keepalive_passes_through_unparsed(self):
        lines = self._run([": keepalive\n\n", "data: [DONE]\n\n"])
        assert lines == [": keepalive\n\n", "done:None:0:0"]

    def test_malformed_json_frame_is_skipped(self):
        lines = self._run(["data: {not json\n\n", "data: [DONE]\n\n"])
        assert lines == ["done:None:0:0"]


class TestChunkEnvelope:
    def test_envelope_carries_model_and_timestamp(self):
        from fusion_mlx.api.ollama_routes import _ollama_chunk_line

        payload = json.loads(_ollama_chunk_line("qwen3.5-4b", {"done": False}))
        assert payload["model"] == "qwen3.5-4b"
        assert payload["created_at"].endswith(".000000Z")
        assert payload["done"] is False

    def test_done_body_maps_length_and_normalizes_stop(self):
        from fusion_mlx.api.ollama_routes import _ollama_done_body

        assert _ollama_done_body("length", 1, 2)["done_reason"] == "length"
        assert _ollama_done_body("stop", 1, 2)["done_reason"] == "stop"
        assert _ollama_done_body(None, 1, 2)["done_reason"] == "stop"
        assert _ollama_done_body("length", 5, 6) == {
            "done": True,
            "done_reason": "length",
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 5,
            "prompt_eval_duration": 0,
            "eval_count": 6,
            "eval_duration": 0,
        }
