# Agent Compatibility Matrix

doc_id: fusion-mlx/docs/agent-compatibility
status: published
updated: 2026-11-12
harness: `fusion_mlx/agents/testing.py` — run via `fusion-mlx agents <name> --test`

## Summary

fusion-mlx ships an in-tree agent-integration harness: 10 first-class agent
profiles (config adapters + per-agent test plans) exercised against the live
server. This document publishes the compatibility matrix measured on
production (v0.10.60, Qwen3.5-4B-MLX-4bit, 2026-11-12).

**Result: 79 PASS / 0 FAIL / 0 ERROR / 9 SKIP** across 10 agents — every
executable test passed; skips are agent CLI binaries (or bundled integration
files) not present on the measurement host, not product gaps.

## Matrix

PASS counts exclude SKIPs (binaries/bundled files absent locally). Test names
are the harness plan items: API-level protocol checks (tool calling, choice,
multi-turn, streaming, leak/stress guards) plus real-binary e2e when
installed.

| Agent | Result | Coverage notes |
|---|---|---|
| opencode | 10/11 | all API tests pass; e2e skip (binary absent) |
| qwen-code | 10/11 | all API tests pass; e2e skip (binary absent) |
| kilo-code | 10/11 | all API tests pass; e2e skip (binary absent) |
| hermes | 11/13 | all executable pass; e2e + bundled-file skips |
| codex | 9/10 | all API tests pass; e2e skip (binary absent) |
| langchain | 9/10 | all API tests pass; bundled-file skip |
| pydanticai | 9/10 | all API tests pass; bundled-file skip |
| smolagents | 7/8 | all API tests pass; bundled-file skip |
| openhands | 2/2 | all pass |
| aider | 2/2 | all pass |
| Claude Code | dogfood | production daily driver — this session's own traffic (streaming, thinking, tool calls, 100k+ contexts) runs through fusion-mlx over both OpenAI and Anthropic APIs |

Per-agent test-plan detail lives in the profiles
(`fusion_mlx/agents/profiles/*.yaml`): which agent needs function calling,
streaming tool calls, tag suppression, many-tool handling, and which CLI
binary the e2e tier launches.

## What is tested

- **plain chat** — coherent completion
- **single_tool_call / tool_choice / multi_turn_tool** — OpenAI tool-calling
  protocol incl. forced tool choice and multi-turn tool result round-trips
- **no_tool_leak / stress_no_leak** — raw markup (`<tool_call>`,
  `<function=`, im_end) never leaks into content, under repeated load
- **streaming_basic / streaming_tool_call** — SSE deltas for content and
  reasoning, tool-call chunk assembly with finish_reason
- **e2e_chat / e2e_file_read / e2e_terminal** — the agent's real CLI binary
  driven against the server (only when the binary is installed)

## Running it

```bash
# one agent
fusion-mlx agents codex --test --base-url http://127.0.0.1:11434/v1

# servers with auth enabled
FUSION_MLX_AGENT_TEST_API_KEY=<key> fusion-mlx agents hermes --test
```

Adding a new agent = one YAML profile
(`fusion_mlx/agents/profiles/<name>.yaml`); the test plan, config adapter,
and setup guide derive from it. See `fusion-mlx agents` for the list.

## Release gating

The harness is the integration-tier test suite for agent compatibility. The
issue that requested this matrix (#1538) also asked for a CI job gating
releases on the full 5-agent real-binary matrix; that tier requires agent
binaries + real weights in CI and is tracked separately — the matrix above
is refreshed from production before each release train instead.
