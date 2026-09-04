# SPDX-License-Identifier: Apache-2.0
from fusion_mlx.telemetry.redact import normalize_caller_agent


def test_known_agents_bucketed():
    assert normalize_caller_agent("claude-cli/1.0 something") == "claude-code"
    assert normalize_caller_agent("Cursor/0.42") == "cursor"
    assert normalize_caller_agent("aider 0.5") == "aider"


def test_unknown_agent_other():
    assert normalize_caller_agent("Mozilla/5.0") == "other"
    assert normalize_caller_agent("") == "other"
    assert normalize_caller_agent(None) == "other"


def test_no_raw_ua_leaks():
    out = normalize_caller_agent("claude-cli/1.0 (secret-token-here)")
    assert "secret-token-here" not in out
    assert out == "claude-code"
