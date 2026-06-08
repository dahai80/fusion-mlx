# API Reference

fusion-mlx exposes 14 public API endpoints plus a full admin panel. All `/v1/*` endpoints are OpenAI-compatible.

## Chat Completions (OpenAI)

### `POST /v1/chat/completions`

Generate a chat completion. Supports streaming, tool calling, and structured output.

**Request:**
```json
{
    "model": "Qwen2.5-3B-Instruct-4bit",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": false
}
```

**Response (non-streaming):**
```json
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1780501235,
    "model": "Qwen2.5-3B-Instruct-4bit",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "The capital of France is Paris.",
            "tool_calls": null
        },
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 47,
        "completion_tokens": 8,
        "total_tokens": 55,
        "cached_tokens": 0
    }
}
```

**Streaming response (SSE):**
```
data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-2","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"The"}}]}

data: {"id":"chatcmpl-3","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" capital"}}]}

data: {"id":"chatcmpl-4","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{...}}

data: [DONE]
```

**Cloud routing**: When `cloud_router_enabled=True` and the request has >32K uncached tokens, both streaming and non-streaming requests are automatically routed to the configured cloud provider. The circuit breaker opens after 5 consecutive local failures, forcing all requests to cloud until a local success resets it.

**Optional parameters:**
- `stream` (bool) — Enable SSE streaming
- `top_k` (int) — Top-k sampling
- `min_p` (float) — Minimum probability threshold
- `repetition_penalty` (float) — Repeat penalty (default 1.0)
- `presence_penalty` (float) — Presence penalty
- `frequency_penalty` (float) — Frequency penalty
- `stop` (list[str]) — Stop sequences
- `tools` (list[dict]) — Tool definitions for function calling
- `tool_choice` (str|dict) — Tool selection strategy
- `response_format` (dict) — JSON schema for structured output
- `enable_thinking` (bool) — Enable reasoning/thinking mode
- `backend_override` (str) — Force specific backend: "omlx", "rapid", "cloud"
- `task_tag` (str) — Priority tag: "claude_code" (REALTIME), "openclaw" (BATCH), "background"

### `POST /v1/completions`

Legacy text completion endpoint. Converts to chat format internally.

**Request:**
```json
{
    "model": "Qwen2.5-3B-Instruct-4bit",
    "prompt": "Once upon a time",
    "max_tokens": 50
}
```

### `GET /v1/models`

List available models in the engine pool.

**Response:**
```json
{
    "object": "list",
    "data": [
        {"id": "Qwen2.5-3B-Instruct-4bit", "object": "model"},
        {"id": "Qwen3.6-27B-mxfp8", "object": "model"}
    ]
}
```

---

## Messages (Anthropic)

### `POST /v1/messages`

Anthropic-compatible Messages API. Supports tools, extended thinking, and streaming.

**Request:**
```json
{
    "model": "Qwen2.5-3B-Instruct-4bit",
    "messages": [
        {"role": "user", "content": "What is the weather in Paris?"}
    ],
    "max_tokens": 256,
    "tools": [{
        "name": "get_weather",
        "description": "Get current weather",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }]
}
```

**Response:**
```json
{
    "id": "msg_abc123",
    "type": "message",
    "role": "assistant",
    "model": "Qwen2.5-3B-Instruct-4bit",
    "content": [
        {"type": "text", "text": "Let me check the weather in Paris."}
    ],
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 35,
        "output_tokens": 8
    }
}
```

### `POST /v1/count_tokens`

Count tokens for a given input.

**Request:**
```json
{
    "model": "Qwen2.5-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello world"}]
}
```

**Response:**
```json
{"input_tokens": 2}
```

---

## Audio

### `POST /v1/audio/transcriptions`

Convert speech to text (STT). Accepts audio file upload.

**Request (multipart/form-data):**
- `file` — Audio file (wav, mp3, m4a, etc.)
- `model` — STT model name (e.g., "whisper-large")
- `language` (optional) — Target language code
- `response_format` (optional) — "json", "text", "srt", "verbose_json"
- `temperature` (optional) — Decoding temperature
- `max_tokens` (optional) — Raise output cap for long audio (e.g., 65536 for VibeVoice-ASR)
- `word_timestamps` (optional) — Enable word-level alignment for Whisper models

**Response:**
```json
{
    "text": "The quick brown fox jumps over the lazy dog.",
    "language": "en",
    "duration": 2.5
}
```

### `POST /v1/audio/speech`

Convert text to speech (TTS). Returns WAV audio.

**Request:**
```json
{
    "model": "kokoro",
    "input": "Hello, this is a text-to-speech demo.",
    "voice": "default",
    "speed": 1.0,
    "response_format": "wav"
}
```

**Response:** Raw WAV bytes (`Content-Type: audio/wav`)

### `POST /v1/audio/process`

Process audio files — enhancement, source separation, etc.

**Request (multipart/form-data):**
- `file` — Input audio file
- `model` — Processing model name
- `task` — Processing task type

---

## Images

### `POST /v1/images/generate`

Generate images from text prompts using Flux 2.

**Request:**
```json
{
    "prompt": "A golden sunset over a mountain lake",
    "n": 1,
    "width": 1024,
    "height": 1024,
    "steps": 20,
    "guidance": 7.5
}
```

**Response:**
```json
{
    "created": 1780501235,
    "data": [
        {
            "b64_json": "iVBORw0KGgoAAAANSUhEUgAA...",
            "url": null
        }
    ]
}
```

---

## MCP (Model Context Protocol)

### `GET /v1/mcp/tools`

List all available MCP tools.

**Response:**
```json
{
    "tools": [
        {"name": "weather_lookup", "description": "Look up weather by city"},
        {"name": "code_search", "description": "Search code repositories"}
    ]
}
```

### `GET /v1/mcp/servers`

List MCP server status.

### `POST /v1/mcp/execute`

Execute an MCP tool by name.

**Request:**
```json
{
    "tool_name": "weather_lookup",
    "arguments": {"city": "Paris"}
}
```

**Response:**
```json
{
    "tool_name": "weather_lookup",
    "content": [{"type": "text", "text": "22C, partly cloudy"}],
    "is_error": false
}
```

---

## System

### `GET /health`

Health check with MLX memory stats.

**Response:**
```json
{
    "status": "ok",
    "version": "0.1.0",
    "engines": ["Qwen2.5-3B-Instruct-4bit", "Qwen3.6-27B-mxfp8"],
    "mx_memory": {
        "active": "2.5 GB",
        "cached": "1.2 GB",
        "peak": "4.0 GB"
    }
}
```

### `GET /metrics`

Server metrics — request counts, token totals, per-model stats.

**Response:**
```json
{
    "total_requests": 150,
    "successful_requests": 148,
    "failed_requests": 2,
    "total_tokens_generated": 45000,
    "total_tokens_prompt": 12000,
    "active_requests": 3,
    "model_stats": {
        "Qwen2.5-3B-Instruct-4bit": {
            "requests": 100,
            "tokens_generated": 30000
        }
    }
}
```

---

## Admin Panel

The admin panel is accessible at `http://localhost:8000/admin/` and provides:

- **Dashboard** — System overview, memory usage, model status
- **Chat** — Interactive chat interface for testing models
- **Model management** — Load/unload/pin models dynamically
- **HuggingFace integration** — Search and download models directly
- **ModelScope integration** — Alternative model source
- **Quantization (oQ)** — Online quantization pipeline
- **Profiles** — Per-model performance profiles
- **Sub-API keys** — API key management
- **Benchmarks** — Built-in benchmarking tools

Key admin API endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/admin/api/models` | List all discovered models |
| POST | `/admin/api/models/{id}/load` | Load a model into memory |
| POST | `/admin/api/models/{id}/unload` | Unload a model |
| PUT | `/admin/api/models/{id}/settings` | Update model settings |
| GET | `/admin/api/global-settings` | Get server configuration |
| POST | `/admin/api/global-settings` | Update server configuration |
| GET | `/admin/api/stats` | Detailed server statistics |
| POST | `/admin/api/stats/clear` | Clear session metrics |
| POST | `/admin/api/stats/clear-alltime` | Clear all-time metrics |
| POST | `/admin/api/ssd-cache/clear` | Clear SSD cache files |
| POST | `/admin/api/hot-cache/clear` | Clear in-memory hot cache |
| POST | `/admin/api/cache/probe` | Probe cache state for messages |
| GET | `/admin/api/hf/models` | Search HuggingFace models |
| POST | `/admin/api/hf/download` | Start a model download |

### Cache Probe

Probe how a chat message list maps to cache state:

```json
POST /admin/api/cache/probe
{
    "model_id": "Qwen2.5-3B-Instruct-4bit",
    "messages": [
        {"role": "user", "content": "Hello"}
    ]
}
```

**Response:**
```json
{
    "model_id": "Qwen2.5-3B-Instruct-4bit",
    "model_loaded": true,
    "total_tokens": 12,
    "block_size": 64,
    "total_blocks": 1,
    "blocks_ssd_hot": 1,
    "blocks_ssd_disk": 0,
    "blocks_cold": 0,
    "ssd_hit_tokens": 64,
    "cold_tokens": 0
}
```

## OpenClaw Agent Protocol

The OpenClaw Agent Protocol extends the standard OpenAI API with agent-specific
features: multi-turn session management, tool calling, conversation steering, and
SSE event streaming.

**Session lifecycle**: Sessions have a 1-hour TTL (from last access) and a maximum
cap of 1000 concurrent sessions. Oldest inactive sessions are evicted via LRU when
the cap is reached.

### `POST /v1/openclaw/agent/sessions`

Create a new agent session with optional system prompt and tool definitions.

**Request:**
```json
{
    "model": "Qwen2.5-3B-Instruct-4bit",
    "system_prompt": "You are a helpful assistant with access to weather data.",
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }]
}
```

**Response:**
```json
{
    "session_id": "a1b2c3d4e5f6",
    "turn_count": 0,
    "active": false,
    "model": "Qwen2.5-3B-Instruct-4bit",
    "tools_count": 1
}
```

### `GET /v1/openclaw/agent/sessions/{session_id}`

Get session metadata and state. Resets the 1h TTL timer.

### `DELETE /v1/openclaw/agent/sessions/{session_id}`

Delete a session and free resources.

### `GET /v1/openclaw/agent/sessions`

List all active sessions. Automatically expires sessions older than 1 hour.

### `POST /v1/openclaw/agent/turns?session_id={id}`

Execute one agent turn. The agent processes input messages and returns either
text content or tool call requests. Resets the 1h TTL timer.

**Request:**
```json
{
    "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
    "max_tokens": 4096,
    "temperature": 0.7
}
```

**Response (text):**
```json
{
    "content": "Let me check the weather in Tokyo for you.",
    "tool_calls": [],
    "usage": {"prompt_tokens": 12, "completion_tokens": 10},
    "session_id": "a1b2c3d4e5f6"
}
```

**Response (tool call):**
```json
{
    "content": "",
    "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": "{\"city\": \"Tokyo\"}"
        }
    }],
    "usage": {"prompt_tokens": 12, "completion_tokens": 15},
    "session_id": "a1b2c3d4e5f6"
}
```

### `POST /v1/openclaw/agent/tool-results`

Submit the result of a tool execution back to the agent for continued processing.

**Request:**
```json
{
    "session_id": "a1b2c3d4e5f6",
    "tool_call_id": "call_abc123",
    "result": "{\"temperature\": 22, \"condition\": \"sunny\"}"
}
```

### `POST /v1/openclaw/agent/steer`

Inject a steering message into an active session. Modes:
- `append` — Add at end of history
- `prepend` — Add before last user message
- `replace` — Replace last message

**Request:**
```json
{
    "session_id": "a1b2c3d4e5f6",
    "message": {"role": "system", "content": "Now respond in Japanese."},
    "mode": "append"
}
```

### `GET /v1/openclaw/agent/stream/{session_id}`

SSE stream of agent events. Events include:
- `connected` — Connection established
- `session_state` — Current session snapshot
- `turn_start` / `turn_end` — Turn lifecycle
- `tool_call` — Agent requested a tool call
- `tool_result` — Tool result was submitted
- `heartbeat` — Keep-alive (every 30s)
- `session_closed` — Session was deleted or expired

**Usage:**
```bash
curl -N http://localhost:8000/v1/openclaw/agent/stream/a1b2c3d4e5f6
```

### Typical Agent Flow

```
1. POST /sessions               → Create session with tools
2. POST /turns?session_id=X     → "What's the weather in Tokyo?"
3. ← Response with tool_calls   → Agent wants to call get_weather("Tokyo")
4. (Caller executes tool)
5. POST /tool-results           → Submit: {"temp": 22, "condition": "sunny"}
6. POST /turns?session_id=X     → Continue with empty message
7. ← Final text response         → "Tokyo is currently 22°C and sunny."
```
