"""CLI for fusion-mlx.

Unified CLI combining omlx (serve, launch, ps, stats, diagnose)
with Rapid-MLX conveniences.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import requests

from ._version import __version__
from .config import DEFAULT_ALIASES, MemoryTier, ServerConfig


def _get_server_addr(host: str = "localhost", port: int = 8000) -> str:
    return f"http://{host}:{port}"


def _api_get(path: str, host: str = "localhost", port: int = 8000) -> Optional[dict]:
    try:
        r = requests.get(_get_server_addr(host, port) + path, timeout=5)
        if r.status_code == 200:
            return r.json()
        return None
    except requests.RequestException:
        return None


def _get_api_key() -> str:
    import os
    return os.environ.get("FUSION_MLX_API_KEY", "local-key")

def serve_command(args):
    from .server import create_app

    config = ServerConfig(
        host=args.host,
        port=args.port,
        model_dir=args.model_dir,
      )
    if args.memory_tier:
        config.memory.tier = MemoryTier(args.memory_tier)
    if args.enable_ssd_cache:
        config.memory.ssd_cache_enabled = True

    app = create_app(config)
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port)


def launch_command(args):
    print(f"fusion-mlx {__version__}: launching '{args.integration}'")
    server = _get_server_addr(args.host, args.port)
    model = getattr(args, "model", None)
    if args.integration == "claude":
        token = _get_api_key()
        print(f"Exporting environment for Claude Code:")
        print(f'  export ANTHROPIC_BASE_URL="{server}"')
        print(f'  export ANTHROPIC_AUTH_TOKEN="{token}"')
        print(f"\nOr run inline:")
        print(f'  ANTHROPIC_BASE_URL="{server}" ANTHROPIC_AUTH_TOKEN="{token}" claude')
    elif args.integration == "openclaw":
        print(f"Configuring OpenClaw to use local server at {server}")
        from .integrations.openclaw import OpenClawIntegration
        OpenClawIntegration().launch(
            port=args.port, api_key=_get_api_key(),
            model=model or "select-a-model", host=args.host,
          )
    elif args.integration == "comfyui":
        print(f"Configuring ComfyUI to use fusion-mlx at {server}")
        from .integrations.comfyui import ComfyUIIntegration
        ComfyUIIntegration().launch(
            port=args.port, api_key=_get_api_key(),
            model=model or "flux-2", host=args.host,
          )
    else:
        print(f"Unknown integration: {args.integration}", file=sys.stderr)
        sys.exit(1)


def ps_command(args):
    host = getattr(args, "host", None) or "localhost"
    port = getattr(args, "port", None) or 8000
    data = _api_get("/health", host, port)
    if not data:
        print("Error: cannot reach server. Is fusion-mlx running?")
        sys.exit(1)

    engines = data.get("engines", [])
    if not engines:
        print("No models loaded.")
        return

    print(f"{'MODEL':<45} {'TYPE':<10} {'LOADED':<7} {'PINNED':<7} {'SIZE':<12}")
    print("-" * 85)
    for e in engines:
        model_id = e.get("id", "?")[:42]
        mtype = e.get("model_type", "?")
        loaded = "yes" if e.get("loaded") else "no"
        pinned = "yes" if e.get("pinned") else "no"
        size = e.get("estimated_size", "?")
        if isinstance(size, (int, float)) and size > 0:
            size = f"{size / 1e9:.1f} GB"
        print(f"{model_id:<45} {mtype:<10} {loaded:<7} {pinned:<7} {size:<12}")

    mx_stats = data.get("mx_memory", {})
    print()
    print("MLX Memory:")
    print(f"  Active:      {mx_stats.get('active', '?')}")
    print(f"  Cached:      {mx_stats.get('cached', '?')}")
    print(f"  Peak:         {mx_stats.get('peak', '?')}")
    limit = mx_stats.get("memory_limit")
    if limit is not None:
        print(f"  Limit:       {limit}")


def stats_command(args):
    host = getattr(args, "host", None) or "localhost"
    port = getattr(args, "port", None) or 8000
    data = _api_get("/stats", host, port)
    if not data:
        data = _api_get("/health", host, port)
    if not data:
        print("Error: cannot reach server. Is fusion-mlx running?")
        sys.exit(1)

    print(json.dumps(data, indent=2, default=str))


def models_command(args):
    host = getattr(args, "host", None) or "localhost"
    port = getattr(args, "port", None) or 8000
    data = _api_get("/v1/models", host, port)
    if not data:
        print("Error: cannot reach server. Is fusion-mlx running?")
        sys.exit(1)

    models = data.get("data", [])
    if not models:
        print("No models found.")
        return

    print(f"{'MODEL ID':<50} {'TYPE':<12}")
    print("-" * 65)
    for m in models:
        mid = m.get("id", "?")[:47]
        mtype = m.get("type", "llm")
        print(f"{mid:<50} {mtype:<12}")

    print()
    print("Default aliases:")
    for alias, real in DEFAULT_ALIASES.items():
        print(f"    {alias:<25} -> {real}")


def diagnose_command(args):
    import psutil
    import mlx.core as mx

    print("=== fusion-mlx Diagnostics ===\n")

    mem = psutil.virtual_memory()
    print(f"Physical RAM:       {mem.total / 1e9:.1f} GB")
    print(f"Available RAM:      {mem.available / 1e9:.1f} GB")
    print(f"Used RAM:           {mem.used / 1e9:.1f} GB ({mem.percent}%)")

    try:
        ver = mx.__version__ if hasattr(mx, "__version__") else "unknown"
        print(f"\nMLX version:        {ver}")
        print(f"MLX active mem:     {mx.get_active_memory() / 1e9:.2f} GB")
        print(f"MLX cache mem:      {mx.get_cache_memory() / 1e9:.2f} GB")
        print(f"MLX peak mem:       {mx.get_peak_memory() / 1e9:.2f} GB")
    except Exception as e:
        print(f"\nMLX info:        unavailable ({e})")

    host = getattr(args, "host", None) or "localhost"
    port = getattr(args, "port", None) or 8000
    health = _api_get("/health", host, port)
    if health:
        print(f"\nServer:          running on {host}:{port}")
        print(f"Engines:          {len(health.get('engines', []))}")
    else:
        print(f"\nServer:          not reachable at {host}:{port}")

    server_model_dir = health.get("model_dir") if health else None
    model_dir = Path(args.model_dir) if args.model_dir else (
        Path(server_model_dir) if server_model_dir
        else Path.home() / ".fusion-mlx" / "models"
     )
    print(f"\nModel directory: {model_dir}")
    if model_dir.exists():
        models = [d.name for d in model_dir.iterdir() if d.is_dir()]
        print(f"Models found:       {len(models)}")
        for m in models:
            print(f"   - {m}")
    else:
        print("Model directory does not exist yet.")


def main():
    parser = argparse.ArgumentParser(
        prog="fusion-mlx",
        description="Unified local model management for Apple Silicon",
      )
    parser.add_argument("--version", action="version", version=f"fusion-mlx {__version__}")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve
    serve_p = subparsers.add_parser("serve", help="Start the inference server")
    serve_p.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    serve_p.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve_p.add_argument("--model-dir", default=None, help="Directory containing MLX models")
    serve_p.add_argument(
        "--memory-tier",
        choices=["safe", "balanced", "aggressive", "custom"],
        default="balanced",
        help="Memory enforcement tier (default: balanced)",
    )
    serve_p.add_argument("--enable-ssd-cache", action="store_true", help="Enable SSD cold layer")
    serve_p.set_defaults(func=serve_command)

    # launch
    launch_p = subparsers.add_parser("launch", help="Launch an integration")
    launch_p.add_argument(
        "integration",
        choices=["claude", "openclaw", "comfyui"],
        help="Integration to launch",
    )
    launch_p.add_argument("--model", default=None, help="Model name to use")
    launch_p.set_defaults(func=launch_command)

    # ps
    ps_p = subparsers.add_parser("ps", help="Show loaded models and memory usage")
    ps_p.set_defaults(func=ps_command)

    # stats
    stats_p = subparsers.add_parser("stats", help="Show server metrics")
    stats_p.set_defaults(func=stats_command)

    # models
    models_p = subparsers.add_parser("models", help="List available models and aliases")
    models_p.set_defaults(func=models_command)

    # diagnose
    diag_p = subparsers.add_parser("diagnose", help="Run system diagnostics")
    diag_p.add_argument("--model-dir", default=None, help="Directory containing MLX models")
    diag_p.set_defaults(func=diagnose_command)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
