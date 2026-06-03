# fusion-mlx

Unified local model serving for Apple Silicon — merges **omlx** (long-context, memory control, multi-model concurrency) with **Rapid-MLX** (speculative decoding, multi-modal, cloud routing).

Drop-in replacement for Ollama, vLLM, or any OpenAI-compatible inference server — runs natively on Metal via MLX.

## Features

- **8 engine types**: LLM, VLM, Embedding, Reranker, STT, TTS, STS, ImageGen (Flux 2)
- **OpenAI + Anthropic API** compatibility — one server, two API flavors
- **Continuous batching** with vLLM-style scheduler (chunked prefill, preemption, KV cache)
- **Speculative decoding**: SuffixDecoding, DFlash, MTP, VLM MTP — 2-5× faster generation
- **Paged KV cache** with SSD cold layer and block-aware prefix caching (COW sharing)
- **4-tier memory enforcer**: safe / balanced / aggressive / custom hard limits
- **Multi-model concurrency**: EnginePool with LRU eviction, pinning, and TTL
- **MCP tool support**: list, discover, and execute MCP tools via API
- **Admin web panel**: model management, live chat, HuggingFace downloads, quantization
- **CLI integrations**: `launch claude`, `launch openclaw`, `launch comfyui`

## Quick Start

```bash
# Install
pip install -e ./fusion-mlx

# Start server with your MLX models
fusion-mlx serve --model-dir ~/.cache/huggingface --port 8000

# Test it
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 10
  }'
```

Or use the OpenAI Python client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
resp = client.chat.completions.create(
    model="Qwen2.5-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    max_tokens=10,
)
print(resp.choices[0].message.content)
```

## Supported Models

| Type | Engine | Example Models |
|------|--------|----------------|
| LLM | `BatchedEngine` | Qwen, Llama, Mistral (any MLX-format text model) |
| VLM | `VLMBatchedEngine` | LLaVA, Qwen2-VL, InternVL |
| Embedding | `EmbeddingEngine` | BGE, E5, GTE |
| Reranker | `RerankerEngine` | Cohere, Jina rerankers |
| STT | `STTEngine` | Whisper, VibeVoice-ASR |
| TTS | `TTSEngine` | Kokoro, VibeVoice |
| ImageGen | `ImageGenEngine` | Flux 2 |

## API Compatibility

| API | Endpoints | Status |
|-----|-----------|--------|
| OpenAI Chat | `/v1/chat/completions`, `/v1/models` | Fully compatible |
| OpenAI Legacy | `/v1/completions` | Supported |
| Anthropic Messages | `/v1/messages`, `/v1/count_tokens` | Fully compatible |
| Audio | `/v1/audio/transcriptions`, `/v1/audio/speech` | Supported |
| Images | `/v1/images/generate` | Supported (Flux 2) |
| MCP | `/v1/mcp/tools`, `/v1/mcp/execute` | Supported |

## Model Aliases

Use familiar names instead of full model IDs:

```bash
fusion-mlx serve --model claude-4.6-sonnet  # → Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit
fusion-mlx serve --model gpt-4o              # → Qwen3-32B-A3B-Think-2512-MLX
```

## Performance

Benchmarks on M4 Max (64 GB RAM):

| Model | Throughput | Latency (first token) |
|-------|-----------|----------------------|
| Qwen2.5-3B-4bit | ~32 tok/s | ~0.5s |
| Qwen3.6-27B-mxfp8 | ~8 tok/s | ~1s (after cold load) |

## Documentation

- [API Reference](docs/api-reference.md) — All endpoints with request/response examples
- [Architecture](docs/architecture.md) — EnginePool, Scheduler, Cache layers
- [CLI Reference](docs/cli-reference.md) — All commands and flags
- [Configuration](docs/configuration.md) — Memory tiers, scheduler settings, aliases

## Examples

See [`examples/`](examples/) for working code:

- `01-basic-chat.py` — Simple non-streaming chat
- `02-streaming-chat.py` — SSE streaming responses
- `03-anthropic-api.py` — Anthropic Messages API
- `04-tool-calling.py` — Function calling with JSON schema
- `05-multi-model.py` — Concurrent multi-model requests
- `06-image-generation.py` — Flux 2 image generation
- `07-speech-to-text.py` — Whisper STT via API
- `08-text-to-speech.py` — Kokoro TTS with WAV output
- `09-mcp-tools.py` — MCP tool discovery and execution
- `10-python-sdk.py` — OpenAI Python client integration

## Admin Panel

Access the web admin at `http://localhost:8000/admin`:

- Model management (load/unload/pin models dynamically)
- Live chat interface for testing models
- HuggingFace / ModelScope model downloads
- Online quantization (oQ) pipeline
- Memory and performance monitoring
- Sub-API key management

## Project Structure

```
fusion-mlx/
├── fusion_mlx/
│   ├── api/            # OpenAI, Anthropic, Audio, Images, MCP routes
│   ├── cache/          # PagedCache, PagedSSDCache, PrefixCache
│   ├── engines/        # 8 engine types (LLM, VLM, Embedding, etc.)
│   ├── integrations/   # Claude Code, OpenClaw, Copilot, ComfyUI
│   ├── parsers/        # Tool call parsers (Gemma, Harmony, etc.)
│   ├── pool/           # EnginePool, MemoryEnforcer, ModelDiscovery
│   ├── router/         # RequestRouter, CloudRouter
│   ├── speculative/    # SuffixDecoding, DFlash, MTP, VLM MTP
│   └── admin/          # Web panel routes, benchmarking, downloads
├── downstream/         # Sync scripts for omlx and Rapid-MLX forks
├── docs/               # API reference, architecture, CLI guide
├── examples/           # Working code examples
└── tests/              # Test suite
```

## License

Apache-2.0
