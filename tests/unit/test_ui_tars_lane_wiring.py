# SPDX-License-Identifier: Apache-2.0
"""Lane-completeness invariant: all 3 request lanes wire
``inject_ui_tars_sysprompt_for_lane`` so a UI-TARS-aliased model that
declares a Computer-Use tool gets the action-API sysprompt on every
surface (chat / messages / responses).

The 70 tests in test_ui_tars_lane_parity.py verify the injection
*function* directly. This file guards the *wiring* — that each route
module actually calls the lane helper at its messages-prep site. A
regression that unwires one lane (e.g. a refactor that drops the call)
trips these asserts before it can ship a silent no-op lane.
"""

from __future__ import annotations

import inspect

from fusion_mlx.tool_parsers.ui_tars_tool_parser import (
    inject_ui_tars_sysprompt_for_lane,
    resolve_ui_tars_parser_name,
)


def _source(module) -> str:
    return inspect.getsource(module)


def test_helper_exports_exist():
    assert callable(inject_ui_tars_sysprompt_for_lane)
    assert callable(resolve_ui_tars_parser_name)


def test_chat_lane_wires_injection():
    from fusion_mlx.api import openai_routes

    src = _source(openai_routes)
    assert (
        "inject_ui_tars_sysprompt_for_lane" in src
    ), "chat lane (/v1/chat/completions) must call inject_ui_tars_sysprompt_for_lane"


def test_anthropic_lane_wires_injection():
    from fusion_mlx.api import anthropic_routes

    src = _source(anthropic_routes)
    assert (
        "inject_ui_tars_sysprompt_for_lane" in src
    ), "anthropic lane (/v1/messages) must call inject_ui_tars_sysprompt_for_lane"


def test_responses_lane_wires_injection():
    from fusion_mlx.routes_internal import responses

    src = _source(responses)
    assert (
        "inject_ui_tars_sysprompt_for_lane" in src
    ), "responses lane (/v1/responses) must call inject_ui_tars_sysprompt_for_lane"


def test_resolve_returns_ui_tars_for_alias():
    assert resolve_ui_tars_parser_name("ui-tars-7b-4bit") == "ui_tars"


def test_resolve_returns_none_for_non_ui_tars():
    out = resolve_ui_tars_parser_name("qwen3-8b-4bit")
    assert out != "ui_tars"


def test_inject_noop_without_computer_tool():
    msgs = [{"role": "user", "content": "what is 2+2?"}]
    out = inject_ui_tars_sysprompt_for_lane(
        msgs, model_name="ui-tars-7b-4bit", tool_choice="auto", tools=None
    )
    assert out is msgs, "no computer tool declared -> no injection"


def test_inject_fires_with_computer_tool():
    msgs = [{"role": "user", "content": "click the search button"}]
    tools = [{"type": "function", "function": {"name": "computer"}}]
    out = inject_ui_tars_sysprompt_for_lane(
        msgs, model_name="ui-tars-7b-4bit", tool_choice="auto", tools=tools
    )
    assert len(out) == len(msgs) + 1, "computer tool declared -> sysprompt prepended"
    assert out[0]["role"] == "system"
    assert "click" in out[0]["content"] or "Action" in out[0]["content"]


def test_inject_skipped_for_tool_choice_none():
    msgs = [{"role": "user", "content": "plan the steps"}]
    tools = [{"type": "function", "function": {"name": "computer"}}]
    out = inject_ui_tars_sysprompt_for_lane(
        msgs, model_name="ui-tars-7b-4bit", tool_choice="none", tools=tools
    )
    assert out is msgs, "tool_choice=none -> skip injection"
