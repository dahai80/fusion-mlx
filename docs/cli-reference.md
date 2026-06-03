# CLI Reference

fusion-mlx provides two CLI entry points: `fusion-mlx` and `fm` (short alias).

## Global Flags

```
fusion-mlx [--version] [--host HOST] [--port PORT] <command>
fm         [--version] [--host HOST] [--port PORT] <command>
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

**Examples:**
```bash
# Basic start with HuggingFace cache
fusion-mlx serve --model-dir ~/.cache/huggingface

# Aggressive memory usage (maximize model capacity)
fusion-mlx serve --memory-tier aggressive --port 9000

# With SSD cache for large contexts
fusion-mlx serve --model-dir ~/.cache/huggingface --enable-ssd-cache
```

### `launch` — Launch an integration

```bash
fusion-mlx launch <integration>
```

| Integration | What it does |
|-------------|--------------|
| `claude` | Prints environment variables to point Claude Code at the local server |
| `openclaw` | Writes `~/.openclaw/config.yaml` with local server URL |
| `comfyui` | Sets up ComfyUI integration (stub) |

**Examples:**
```bash
# Configure Claude Code to use fusion-mlx
fusion-mlx launch claude

# Set up OpenClaw
fusion-mlx launch openclaw
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

### Diagnostics
```bash
# Full system check before deploying
fusion-mlx diagnose --model-dir ~/.cache/huggingface
```
