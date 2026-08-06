# Windows CUDA Backend Node (#365)

Fusion-mlx's native engine is Apple-Silicon MLX. For heavy LLM inference
(DeepSeek 70B / Qwen 72B FP8) on Windows CUDA hardware, fusion-mlx ships an
optional **Windows CUDA backend node** powered by [vLLM](https://github.com/vllm-project/vllm).
The node exposes an OpenAI-compatible HTTP API and self-registers with the
cluster via mDNS under `platform=windows-cuda`, so a `fusion-gateway` can route
heavy-model intents to it while keeping lightweight inference on Mac nodes.

> Scope (#365, patch v0.8.4): **LLM-only**. Diffusion-on-CUDA is tracked in a
> separate issue.

## Architecture

```
                     fusion-gateway (platform routing)
                     /v1/chat/completions  ──┐
                                            ├── mac node (MLX, lightweight)
                                            └── windows-cuda node (vLLM, heavy)
                                                 platform=windows-cuda (mDNS TXT)
```

- **Platform tag** — every node advertises a `platform` TXT record
  (`mac` or `windows-cuda`) over mDNS. The gateway's
  `HealthyNodesByPlatform` / `SelectNodeByPlatform` selects nodes by tag.
  Detection: `FUSION_PLATFORM` env var → `sys.platform` + CUDA probe →
  fallback `mac`. See `fusion_mlx/cluster/platform.py`.
- **CUDA node** — `fusion_mlx/backends/cuda_node/node.py` builds a FastAPI app
  embedding vLLM's `AsyncLLMEngine`, serving `/health`, `/v1/models`,
  `/v1/chat/completions`, `/v1/completions`. On startup it registers mDNS with
  `platform=windows-cuda`.
- **Lazy vLLM** — vLLM is a heavy CUDA-only dependency. It is imported lazily
  inside `create_cuda_app`/`run_cuda_node`, so importing the package on a Mac
  never fails. Starting the node without vLLM raises a clear `RuntimeError`.

## Install (Windows CUDA host)

```powershell
# vLLM is not a fusion-mlx dependency; install it separately on the CUDA host.
pip install vllm
pip install fusion-mlx
```

## Start a CUDA node

```bash
# Heavy model, single GPU, FP8 quantization
fusion-mlx cuda-node Qwen/Qwen2.5-72B-Instruct-FP8 \
  --port 8000 --quantization fp8 --tensor-parallel-size 1

# Multi-GPU
fusion-mlx cuda-node deepseek-ai/DeepSeek-V2-Lite-Chat \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --max-model-len 32768

# Disable cluster mDNS advertising
fusion-mlx cuda-node Qwen/Qwen2.5-72B-Instruct --no-cluster-advertise
```

CLI flags:

| Flag | Default | Description |
| --- | --- | --- |
| `model` (positional) | — | HF repo id or local path |
| `--host` | `0.0.0.0` | Bind host |
| `--port` | `8000` | Bind port |
| `--tensor-parallel-size`, `-tp` | `1` | vLLM tensor parallelism |
| `--gpu-memory-utilization` | `0.90` | vLLM GPU memory utilization |
| `--quantization` | `None` | `fp8` / `awq` / `gptq` / `None` |
| `--max-model-len` | `None` | Max context length |
| `--trust-remote-code` | off | Trust remote code |
| `--no-cluster-advertise` | off | Disable mDNS registration |

## OpenAI API

The node speaks the standard OpenAI API:

```bash
curl http://windows-host:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-72B-Instruct-FP8","messages":[{"role":"user","content":"hi"}]}'
```

## Gateway registration

The node advertises itself via mDNS (`_fusion-mlx._tcp.local.`) with TXT
records including `platform=windows-cuda`, `node_id`, `host`, `port`,
`models_csv`. Register it in a `fusion-gateway` cluster config:

```yaml
cluster:
  enabled: true
  nodes:
    - id: "win-cuda-01"
      url: "http://windows-host:8000"
      platform: windows-cuda
```

The gateway then routes heavy-model / high-context intents to the `windows-cuda`
platform, falling back to cloud when no healthy CUDA node is available.

## Programmatic use

```python
from fusion_mlx.backends.cuda_node import CudaNodeConfig, run_cuda_node

cfg = CudaNodeConfig(
    model="Qwen/Qwen2.5-72B-Instruct-FP8",
    port=8000,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    quantization="fp8",
)
run_cuda_node(cfg)  # blocking uvicorn server
```

## Testing

Unit tests (`tests/unit/test_cuda_node_365.py`) stub vLLM in `sys.modules` so
they run on Mac without CUDA. They cover platform detection, the mDNS
`platform` TXT record, `CudaNodeConfig` defaults, and the node's OpenAI routes
(health / models / chat / completions, including the 400 on missing messages).

> Real-model E2E validation requires a Windows CUDA host with vLLM installed —
> not runnable from the Mac dev environment.

## Files

| File | Purpose |
| --- | --- |
| `fusion_mlx/cluster/platform.py` | `Platform` enum + `detect_platform()` |
| `fusion_mlx/cluster/mdns.py` | `build_txt_records` adds `platform` TXT record |
| `fusion_mlx/backends/cuda_node/node.py` | vLLM FastAPI node + `CudaNodeConfig` |
| `fusion_mlx/config.py` | `ServerConfig.platform` field |
| `fusion_mlx/server.py` | `_node_load_snapshot` surfaces `platform` |
| `fusion_mlx/cli.py` | `cuda-node` subcommand |
| `tests/unit/test_cuda_node_365.py` | Unit tests (vLLM stubbed) |

## Gotchas

- **vLLM is not bundled** — install `pip install vllm` on the CUDA host.
  Importing the node on Mac is fine; *starting* it without vLLM raises
  `RuntimeError`.
- **Platform detection** prefers `FUSION_PLATFORM` env var over the CUDA probe
  (vLLM/torch may import lazily). Set `FUSION_PLATFORM=windows-cuda` to force.
- **Diffusion on CUDA** is out of scope for #365 — LLM-only.
- **`on_event` → `lifespan`** — the node uses the modern FastAPI `lifespan`
  context manager (matching `server.py`), not the deprecated `on_event`.
