from __future__ import annotations

import ast
import importlib.resources
import pathlib

import pytest

from fusion_mlx.cache.response_cache import CachePolicy, get_response_cache


def _chat_route_source() -> str:
    pkg = importlib.resources.files("fusion_mlx")
    path = pathlib.Path(str(pkg)) / "api" / "openai" / "chat.py"
    return path.read_text(encoding="utf-8")


def _streaming_source() -> str:
    pkg = importlib.resources.files("fusion_mlx")
    path = pathlib.Path(str(pkg)) / "api" / "openai" / "streaming.py"
    return path.read_text(encoding="utf-8")


def test_bypass_header_contract_1655():
    cache = get_response_cache()
    policy = cache.resolve_policy(0.0, {"X-Cache": "bypass"})
    assert policy == CachePolicy.BYPASS


def test_fresh_non_stream_marks_miss_1655():
    src = _chat_route_source()
    assert '"X-Cache"' in src and '"MISS"' in src
    tree = ast.parse(src)
    assigns = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Subscript)
            and isinstance(t.value, ast.Attribute)
            and t.value.attr == "headers"
            for t in n.targets
        )
    ]
    hits = [
        a
        for a in assigns
        if any(
            isinstance(k, ast.Constant) and k.value == "X-Cache" for k in ast.walk(a)
        )
    ]
    assert hits, "chat route must set X-Cache on fresh responses"


def test_stream_responses_marked_miss_1655():
    src = _streaming_source()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and any(
            isinstance(k, ast.Constant) and k.value == "X-Cache" for k in node.keys
        ):
            vals = [v.value for v in node.values if isinstance(v, ast.Constant)]
            assert "MISS" in vals, "stream headers must carry X-Cache: MISS"
            return
    pytest.fail("_stream_headers must declare X-Cache: MISS")
