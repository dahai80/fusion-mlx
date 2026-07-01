# CLI Reference

fusion-mlx provides the `fusion-mlx` CLI entry point.

## Global Flags

```
fusion-mlx [--version] [--host HOST] [--port PORT] <command>
```

| Flag | Default | Description |
|------|---------|-------------|
| `--version` | — | Print version and exit |
| `--host` | `localhost` | Server host for query commands |
| `--port` | `8000` | Server port for query commands |

---

## Commands

### `serve` — Start the inference server

```bash
fusion-mlx serve [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host HOST` | `0.0.0.0` | Bind address |
| `--port PORT` | `8000` | TCP port |
| `--model-dir PATH` | `~/.fusion-mlx/models` | Directory containing MLX models |
| `--memory-tier TIER` | `balanced` | Memory limit: `safe`, `balanced`, `aggressive`, `custom` |
| `--enable-ssd-cache` | off | Enable SSD-based KV cache cold layer |
| `--admin` | on | Enable admin web panel |
| `--no-admin` | off | Disable admin web panel |

**Examples:**
```bash
# Basic start with HuggingFace cache
fusion-mlx serve --model-dir ~/.cache/huggingface

# Specific model
fusion-mlx serve --model Qwen3-4B-Q4_K_M

# Aggressive memory usage (maximize model capacity)
fusion-mlx serve --memory-tier aggressive --port 9000

# With SSD cache for large contexts
fusion-mlx serve --model-dir ~/.cache/huggingface --enable-ssd-cache

# With admin panel for model management
fusion-mlx serve --model-dir ~/.cache/huggingface --admin

# Custom memory limit (16 GB)
fusion-mlx serve --memory-tier custom --custom-limit-mb 16384
```

### `launch` — Launch an integration

```bash
fusion-mlx launch <integration> [--model MODEL]
```

| Integration | What it does |
|-------------|-------------|
| `claude` | Sets `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` to point Claude Code at the local server |
| `openclaw` | Writes `~/.openclaw/config.yaml` with local server URL |
| `comfyui` | Sets up ComfyUI integration for Flux 2 image generation |
| `copilot` | Configures GitHub Copilot to use local server |
| `codex` | Sets up OpenAI Codex CLI integration |
| `opencode` | Configures OpenCode integration |
| `pi` | Sets up Pi integration |

**Examples:**
```bash
# Configure Claude Code to use fusion-mlx
fusion-mlx launch claude

# Set up OpenClaw with a specific model
fusion-mlx launch openclaw --model Qwen3-4B-Q4_K_M

# Launch ComfyUI for image generation
fusion-mlx launch comfyui
```

### `ps` — Show loaded models and memory

```bash
fusion-mlx ps
```

Queries the `/health` endpoint and displays:
- Loaded model names
- MLX active/cached/peak memory
- Server uptime and version

### `stats` — Show server metrics

```bash
fusion-mlx stats
```

Queries the `/metrics` endpoint and displays:
- Total/successful/failed request counts
- Token generation totals
- Per-model statistics

### `models` — List available models

```bash
fusion-mlx models
```

Queries `/v1/models` and shows:
- All discovered models in the model directory
- Default model aliases (e.g., `claude-4.6-sonnet` → real model ID)
- Model types and sizes

### `diagnose` — Run system diagnostics

```bash
fusion-mlx diagnose [--model-dir PATH]
```

Reports:
- System RAM and CPU info
- MLX metadata (device, version, Metal support)
- Model directory scan (models found, sizes, types)
- Server health check (if server is running)

---

## Usage Patterns

### Daily workflow
```bash
# 1. Start server
fusion-mlx serve --model-dir ~/.cache/huggingface

# 2. Check what's loaded
fusion-mlx ps

# 3. Configure Claude Code
fusion-mlx launch claude

# 4. Monitor usage
fusion-mlx stats
```

### Multi-model serving
```bash
# Start with multiple model directories
fusion-mlx serve --model-dir ~/.cache/huggingface --enable-ssd-cache

# Load specific models via admin panel
# Visit http://localhost:8000/admin
```

### Diagnostics
```bash
# Full system check before deploying
fusion-mlx diagnose --model-dir ~/.cache/huggingface

# Check if server is healthy
fusion-mlx ps
```
