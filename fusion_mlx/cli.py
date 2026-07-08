#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
# SPDX-License-Identifier: Apache-2.0
"""CLI for fusion-mlx.

Commands:
    fusion-mlx serve <model> --port 8000    Start OpenAI-compatible server
    fusion-mlx bench <model>                Run benchmark
    fusion-mlx chat <model>                 Interactive chat REPL

Usage:
    fusion-mlx serve qwen3.5-4b-4bit --port 8000
    fusion-mlx bench qwen3.5-4b-4bit --num-prompts 10
    fusion-mlx chat qwen3.5-4b-4bit
"""

import argparse
import os
import sys

from fusion_mlx._cli_base import (
    _listen_fd_arg,
    _log_level_choice,
    _port_arg,
    _print_unknown_model_help,
)
from fusion_mlx._completion import alias_completer
from fusion_mlx.cli_commands import (
    agents_command,
    chat_command,
    doctor_command,
    info_command,
    models_command,
    ps_command,
    pull_command,
    rm_command,
    telemetry_command,
    upgrade_command,
)
from fusion_mlx.cli_serve import (
    _add_pflash_args,
    bench_command,
    serve_command,
)


def main():
    from importlib.metadata import version as pkg_version

    try:
        _version = pkg_version("fusion-mlx")
    except Exception:
        _version = "dev"

    parser = argparse.ArgumentParser(
        description="Fusion-MLX: AI inference for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  fusion-mlx chat                                      # interactive REPL (defaults to qwen3.5-4b-4bit)
  fusion-mlx chat qwen3.5-9b-4bit --think                   # larger model, surface reasoning
  fusion-mlx serve qwen3.5-9b-4bit --port 8000              # OpenAI-compatible server
  fusion-mlx serve mlx-community/Qwen3.5-9B-4bit       # full HF repo also works
  fusion-mlx models                                    # list all aliases
  fusion-mlx info qwen3.5-9b-4bit                           # show per-alias profile
""",
    )
    parser.add_argument(
        "--version", "-V", action="version", version=f"fusion-mlx {_version}"
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Disable anonymous usage telemetry for this run "
        "(equivalent to FUSION_MLX_TELEMETRY=0).",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Serve command. ``allow_abbrev=False`` blocks unique-prefix matches
    # like ``--no-thin`` resolving silently to ``--no-thinking``: with the
    # hidden ``--no-think`` cross-alias added in D4, both flags share the
    # ``--no-thi`` prefix and prefix matching becomes ambiguous (an
    # ambiguity which argparse does NOT report by default for hidden
    # aliases). Force users to type the flag in full.
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start OpenAI-compatible server",
        allow_abbrev=False,
    )
    serve_parser.add_argument(
        "model",
        nargs="?",
        type=str,
        default=None,
        help="Model to serve (alias or HF repo). Omit when using --model-dir.",
    ).completer = alias_completer
    # Released 1.0/2.0/3.0 contract: `serve --model-dir <dir>` boots the
    # multi-model engine-pool server that auto-discovers every model in
    # <dir> via create_app(ServerConfig(model_dir)). Mutually exclusive
    # with the positional <model> (Rapid-MLX single-model path). Restored
    # after the Rapid-MLX migration rerouted `serve` to `serve <model>`;
    # existing docs/scripts keep working.
    serve_parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory containing MLX models (multi-model server). "
        "Mutually exclusive with positional <model>.",
    )
    # FusionMLX macOS app / omlx-style launch: the app spawns
    # `serve --base-path <dir> --port <port>` where <dir> is the app's data
    # home (default ~/.fusion-mlx). Models live at <base-path>/models, so this
    # is an alias for `--model-dir <base-path>/models` routed to the
    # multi-model engine-pool server. Restored after the Rapid-MLX migration
    # dropped it; without it argparse (allow_abbrev=False) rejects --base-path
    # and the app cannot start its server subprocess.
    serve_parser.add_argument(
        "--base-path",
        default=None,
        help="Base data directory (FusionMLX app / omlx style); serves "
        "<base-path>/models via the multi-model server. Mutually exclusive "
        "with <model>/--model/--model-dir.",
    )
    # Released 1.0/2.0/3.0 contract: `serve --model <name>` (docs/cli-reference.md
    # documents `fusion-mlx serve --model Qwen3-4B-Q4_K_M`). The Rapid-MLX
    # migration replaced this with a positional <model>; keep both so released
    # scripts and the positional form both work. dest=model_flag avoids clashing
    # with the positional's dest=model; serve_command folds them together.
    serve_parser.add_argument(
        "--model",
        dest="model_flag",
        default=None,
        help="Model to serve (alias or HF repo). Released 1.0/2.0/3.0 "
        "contract; positional <model> is also accepted.",
    )
    serve_parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. If not specified, the model argument is used.",
    )
    serve_parser.add_argument(
        "--force-disk-check",
        action="store_true",
        help=(
            "Skip the pre-flight disk-space check that aborts when the model "
            "is larger than free disk. Use only if you know the HF cache lives "
            "on a different filesystem (e.g. external drive via HF_HOME)."
        ),
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help=(
            "Host to bind (default: 127.0.0.1, loopback-only). Pass "
            '0.0.0.0 (or "") to expose the server on every '
            "interface (LAN reachable) — only do this once the "
            "bearer-auth posture has been reviewed. The wildcard "
            "bind also widens the PortSweep collision window: macOS "
            "lets a wildcard listener coexist with a more-specific "
            "(127.0.0.1) listener on the same port, so a second "
            "server may start and silently shadow the first on the "
            "loopback path. The pre-flight bind check below probes "
            "127.0.0.1 explicitly whenever --host is a wildcard "
            "alias to keep that bypass closed."
        ),
    )
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    # Socket activation — let an external supervisor (launchd, systemd,
    # parent process) bind the listening socket and execve into
    # ``fusion-mlx`` with the pre-bound fd. This closes the bind→auth
    # TOCTOU window described in issue #574: no co-located process can
    # land an unauthenticated request between socket bind and FastAPI
    # auth dependency registration, because by the time
    # ``fusion-mlx serve`` runs, the app (with auth dependencies wired
    # into chat/embeddings/audio/models routers) is already constructed
    # before ``uvicorn.run`` starts ``accept()``-ing on the fd.
    #
    # When ``--listen-fd`` is set, ``--host``/``--port`` are IGNORED:
    # the supervisor controls the bind address. The "Ready:" banner
    # prints the inherited fd shape (``Ready: inherited fd N``) — NOT
    # the user-supplied host/port, since those don't reflect the
    # supervisor's actual bind. Setting both ``--listen-fd`` and a
    # non-default ``--port`` is allowed but the port has no effect;
    # the active listener is the inherited fd.
    serve_parser.add_argument(
        "--listen-fd",
        type=_listen_fd_arg,
        default=None,
        metavar="FD",
        help=(
            "File descriptor of a pre-bound listening socket (3-1023). "
            "Used for socket activation (launchd/systemd/parent-process "
            "supervision) — supervisor binds the loopback socket, "
            "validates auth secret, then execve's into fusion-mlx. "
            "When set, --host/--port are ignored for binding."
        ),
    )
    serve_parser.add_argument(
        "--log-level",
        type=_log_level_choice,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level for Python logging and uvicorn (case-insensitive)",
    )
    serve_parser.add_argument(
        "--max-num-seqs", type=int, default=256, help="Max concurrent sequences"
    )
    serve_parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=256,
        help=(
            "Admission cap on in-flight requests (queued + running). When "
            "exceeded, new requests return HTTP 503 with Retry-After. "
            "Default 256; operators on memory-constrained devices may want "
            "to set this near ``--max-num-seqs`` to limit queue depth."
        ),
    )
    serve_parser.add_argument(
        "--prefill-batch-size", type=int, default=8, help="Prefill batch size"
    )
    serve_parser.add_argument(
        "--completion-batch-size", type=int, default=32, help="Completion batch size"
    )
    serve_parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        default=True,
        help="Enable prefix caching for repeated prompts (default: enabled)",
    )
    serve_parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable prefix caching",
    )
    serve_parser.add_argument(
        "--prefix-cache-size",
        type=int,
        default=100,
        help="Max entries in prefix cache (default: 100, legacy mode only)",
    )
    # Memory-aware cache options (recommended for large models)
    serve_parser.add_argument(
        "--cache-memory-mb",
        type=int,
        default=None,
        help="Cache memory limit in MB (default: auto-detect ~20%% of RAM)",
    )
    serve_parser.add_argument(
        "--cache-memory-percent",
        type=float,
        default=0.20,
        help="Fraction of available RAM for cache if auto-detecting (default: 0.20)",
    )
    serve_parser.add_argument(
        "--no-memory-aware-cache",
        action="store_true",
        help="Disable memory-aware cache, use legacy entry-count based cache",
    )
    # R15-P1 (task #303): radix-tree prefix-cache index. Default ``radix``
    # accelerates lookup and accounts for cross-request prefix dedup on
    # shared-system-prompt workloads. ``hash`` is the legacy bisect path,
    # kept as an escape hatch if a regression is found in production.
    serve_parser.add_argument(
        "--prefix-cache-index",
        type=str,
        default="radix",
        choices=("radix", "hash"),
        help=(
            "Prefix-cache lookup index: 'radix' (default, R15-P1) uses a "
            "token trie for O(prefix_len) lookups and surfaces dedup-bytes-"
            "saved on /metrics; 'hash' falls back to the legacy bisect-over-"
            "sorted-keys path."
        ),
    )
    # KV cache quantization options
    # ``--kv-cache-dtype`` (R15 task #300) is the canonical knob: int4 is
    # the new default because Apple Silicon decode is memory-bandwidth-
    # bound and a 4×-smaller KV cache cuts bandwidth proportionally
    # (mlx#3134 UMA discussion, Feb 2026 — Phi-3.5-mini +1.1%
    # throughput, 3.2× more context room on Qwen2.5-14B). The
    # safelist in :mod:`fusion_mlx.kv_cache_dtype` auto-downgrades
    # sliding-window (Gemma 3, GPT-OSS) and MLA (DeepSeek V3+,
    # Kimi-K2.5) families to bf16 where int4 breaks decode quality.
    # ``--reasoning`` pins to int8 for AIME-class hard math where
    # sub-4-bit drops -20pt on thinking variants.
    #
    # Qwen3.5-9B-4bit bench (M3, 292-tok prompt, 5×400-tok decode median):
    # int4 113.6 tok/s / 119 ms TTFT / 5388 MB RSS vs bf16 113.7 tok/s /
    # 120 ms TTFT / 5392 MB RSS — int4 is a free swap at this size; the
    # +1.1 % / 3.2× headroom land at multi-k contexts (PR #910 comment).
    serve_parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="int4",
        choices=["bf16", "int8", "int4"],
        help=(
            "KV cache dtype (R15 #300, default: int4). Apple Silicon decode "
            "is memory-bandwidth-bound; int4 yields ~4× less bandwidth per "
            "decode step with 97-98%% quality retention. Sliding-window "
            "(Gemma 3, GPT-OSS) and MLA (DeepSeek V3+, Kimi K2.5) models "
            "auto-downgrade to bf16. Use --reasoning for AIME / hard math."
        ),
    )
    serve_parser.add_argument(
        "--reasoning",
        action="store_true",
        default=False,
        help=(
            "Reasoning profile: pins --kv-cache-dtype to int8 regardless of "
            "the dtype flag (sub-4-bit drops -20pt on AIME-class math for "
            "Qwen3 thinking variants)."
        ),
    )
    serve_parser.add_argument(
        "--kv-cache-quantization",
        action="store_true",
        help=(
            "[deprecated alias of --kv-cache-dtype int8] Quantize stored "
            "KV caches to reduce memory (8-bit by default). When both "
            "flags are passed, this one wins for backwards compatibility."
        ),
    )
    serve_parser.add_argument(
        "--kv-cache-quantization-bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Bit width for KV cache quantization (default: 8)",
    )
    serve_parser.add_argument(
        "--kv-cache-quantization-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization (default: 64)",
    )
    serve_parser.add_argument(
        "--kv-cache-min-quantize-tokens",
        type=int,
        default=256,
        help="Minimum tokens for quantization to apply (default: 256)",
    )
    # TurboQuant KV cache compression (experimental, R15 Phase 4).
    #
    # Accepts an optional mode value:
    #   --kv-cache-turboquant              → V-only legacy (v4)
    #   --kv-cache-turboquant v4           → V-only explicit
    #   --kv-cache-turboquant k8v4         → K-8bit + V-4bit mix (R15 Phase 4)
    #
    # The bare-flag form preserves PR #157 backward compatibility. Mode
    # is mutually exclusive with --kv-cache-quantization.
    serve_parser.add_argument(
        "--kv-cache-turboquant",
        nargs="?",
        const="v4",
        default=None,
        choices=["v4", "k8v4"],
        help="Enable TurboQuant KV-cache compression. ``v4`` (default when "
        "the flag is bare) is V-only 3-4 bit Lloyd-Max with K in FP16; "
        "``k8v4`` is the R15 Phase 4 mix — K at 8-bit Walsh-Hadamard + V at "
        "4-bit Lloyd-Max (~4.6x KV compression on dense models). "
        "Experimental — mutually exclusive with --kv-cache-quantization.",
    )
    serve_parser.add_argument(
        "--kv-cache-turboquant-bits",
        type=int,
        default=None,
        choices=[3, 4],
        help="V-side bit width for TurboQuant (default: auto-select by head_dim — "
        "3-bit for head_dim>=96, 4-bit for head_dim=64). Ignored when "
        "--kv-cache-turboquant=k8v4 (V is pinned to 4-bit there).",
    )
    serve_parser.add_argument(
        "--kv-cache-turboquant-group-size",
        type=int,
        default=32,
        help="Group size for TurboQuant V-side quantization (default: 32)",
    )
    # R15-P1 (task #296): disk-backed KV checkpointing at 256-tok boundaries.
    # 0 disables the feature entirely (no scheduler-hot-path cost, no
    # ~/.cache/fusion-mlx/kv_checkpoints/ directory creation); the default
    # 256 matches MLX-LM's KVCache.step and LMCache's external-chunk size
    # so the on-disk shape aligns with the in-memory shape on reload.
    serve_parser.add_argument(
        "--kv-disk-checkpoint-interval",
        type=int,
        default=256,
        help=(
            "Token interval at which the scheduler snapshots KV state to "
            "~/.cache/fusion-mlx/kv_checkpoints/ for resume / shared-prefix "
            "reload (R15 #296, default 256). 0 disables. Pairs with the "
            "FUSION_MLX_KV_CHECKPOINT_MAX_BYTES env var (default 20 GiB) "
            "for the oldest-first disk-cap eviction policy."
        ),
    )
    serve_parser.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        help="Tokens to batch before streaming (1=smooth, higher=throughput)",
    )
    serve_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Default max tokens for generation (default: 32768)",
    )
    serve_parser.add_argument(
        "--continuous-batching",
        action="store_true",
        default=True,
        help="Enable continuous batching (default: on).",
    )
    # Deprecated flags — accepted silently to avoid breaking user scripts
    serve_parser.add_argument(
        "--simple-engine",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        choices=[4, 8],
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--kv-group-size",
        type=int,
        default=64,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    # DFlash — block-diffusion drafter speculative decoding (z-lab / mlx-vlm).
    # Currently single-user serial mode; runs a dedicated DFlash server that
    # bypasses BatchedEngine. Eligible aliases declare ``supports_dflash=true``
    # in aliases.json (dense, ≥8-bit, drafter available — qwen3.5-27b-8bit
    # is the only validated one today). PoC: 1.83–2.18× on Qwen3.5-27B-8bit.
    serve_parser.add_argument(
        "--enable-dflash",
        action="store_true",
        default=False,
        help="Enable DFlash speculative decoding (block-diffusion drafter, "
        "single-user serial mode). Requires a DFlash-eligible alias "
        "(see ``fusion-mlx info <alias>``). Loads the drafter from the "
        "alias's ``dflash_draft_model`` field. Install with "
        "``pip install 'fusion-mlx[dflash]'``.",
    )
    serve_parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=4,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--specprefill",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--specprefill-threshold",
        type=int,
        default=8192,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--specprefill-keep-pct",
        type=float,
        default=0.3,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--specprefill-draft-model",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of device memory for Metal allocation limit and emergency "
        "cache clear threshold (0.0-1.0, default: 0.90). Increase to 0.95 for "
        "large models (200GB+) that need more memory headroom.",
    )
    # Paged cache options (experimental)
    serve_parser.add_argument(
        "--use-paged-cache",
        action="store_true",
        help="Use paged KV cache for memory efficiency (experimental)",
    )
    serve_parser.add_argument(
        "--paged-cache-block-size",
        type=int,
        default=64,
        help="Tokens per cache block (default: 64)",
    )
    serve_parser.add_argument(
        "--max-cache-blocks",
        type=int,
        default=1000,
        help="Maximum number of cache blocks (default: 1000)",
    )
    # Chunked prefill
    serve_parser.add_argument(
        "--chunked-prefill-tokens",
        type=int,
        default=0,
        help="Max prefill tokens per scheduler step (0=disabled). "
        "Breaks large prompts into chunks to prevent concurrent requests from starving. "
        "Recommended for Claude Code and agentic workloads with large tool schemas: "
        "--chunked-prefill-tokens 2048",
    )
    # Task #292: opt-in for ``/v1/audio/*`` routes on a text-only server.
    # The audio-mode boot path (``fusion-mlx serve kokoro`` etc.) auto-
    # enables the routes via the registry hit — this flag is the
    # escape hatch for operators who want the audio router mounted
    # alongside a text engine (e.g. side-car deployments that proxy the
    # audio paths to a separate process). Mirrors the ``--enable-mtp``
    # / ``--enable-dflash`` pattern so the surface stays consistent.
    serve_parser.add_argument(
        "--enable-audio",
        action="store_true",
        default=False,
        help="Mount the ``/v1/audio/*`` routes even when the loaded model "
        "is text-only. Useful for side-car deployments that proxy audio "
        "requests to a separate process. Audio-capable models "
        "(kokoro / whisper / parakeet / chatterbox / vibevoice / voxcpm) "
        "auto-mount the routes — this flag is only needed on text-mode boots.",
    )
    # MTP (Multi-Token Prediction)
    serve_parser.add_argument(
        "--enable-mtp",
        action="store_true",
        default=False,
        help="Enable MTP (Multi-Token Prediction) for models with built-in MTP heads. "
        "Uses cache snapshot/restore for speculative generation.",
    )
    serve_parser.add_argument(
        "--mtp-num-draft-tokens",
        type=int,
        default=1,
        help="Number of draft tokens per MTP step (default: 1)",
    )
    serve_parser.add_argument(
        "--mtp-optimistic",
        action="store_true",
        default=False,
        help="Skip MTP acceptance check for maximum speed. "
        "~5-10%% wrong tokens. Best for chat, not for code.",
    )
    # R15-P1 #302: native Qwen3.5/3.6 MTP via vendored mlx-lm PR #990.
    # Lives next to the existing ``--enable-mtp`` (Qwen3-Next runtime
    # injection) rather than replacing it because the two paths target
    # DIFFERENT architectures — Qwen3-Next uses a hybrid Gated-DeltaNet
    # + attention layout that the existing ``_install_mtp`` patches
    # at the BatchGenerator level, while Qwen3.5/3.6 uses a
    # GatedDeltaNet + MTP-head split that needs the PR #990
    # ``mtp_generate_step`` loop (cache rollback, n_confirmed split,
    # probabilistic acceptance). Coexistence keeps existing dogfood
    # users on ``--enable-mtp`` working while the new path is opt-in.
    #
    # Default ``none`` because the lossless contract has not yet been
    # verified end-to-end against a converted Qwen3.5/3.6 checkpoint
    # (R15-P1 follow-up bench is GPU-contended with Stage B Viterbi
    # conversion). The CLI rejects ``--spec-decode mtp`` at boot if
    # the loaded model's ``config.json`` lacks
    # ``mtp_num_hidden_layers >= 1`` so an operator who passes the
    # flag against a non-eligible model sees a clear error rather
    # than silent fallback.
    serve_parser.add_argument(
        "--spec-decode",
        dest="spec_decode",
        choices=["none", "mtp", "dflash", "dspark", "auto"],
        default="none",
        help=(
            "R15-P1 model-side speculative decode. "
            "``none`` (default) disables; ``mtp`` enables Qwen3.5/3.6 "
            "native MTP via vendored mlx-lm PR #990 — requires a "
            "checkpoint converted with the PR #990 sanitize() path "
            "that preserves ``mtp.*`` weights; ``dflash`` enables the "
            "block-diffusion drafter from arxiv 2410.04097 (R15-P1 "
            "#313) for Qwen3.5/3.6 with a bound drafter (default "
            "block size 16); ``dspark`` enables DeepSeek DeepSpec "
            "lossless block speculative decode; ``auto`` asks "
            "SpecAutoRouter to pick at boot from the model's shape "
            "and drafter flags — MTP-eligible checkpoints get ``mtp``, "
            "everything else gets n-gram suffix decoding (zero GPU "
            "cost). Drafter-backed methods (dflash/dspark) stay "
            "operator-selected even under ``auto``. Rejects at boot "
            "if the model doesn't qualify so misuse fails loud."
        ),
    )
    # R15-P1 #313: DFlash drafter HF path override. Empty by default
    # so the side-registry's per-alias binding wins; an operator who
    # wants to swap the default drafter for a fine-tuned variant can
    # pass this without editing the registry.
    serve_parser.add_argument(
        "--dflash-drafter-path",
        dest="dflash_drafter_path",
        default="",
        help=(
            "Override the per-alias DFlash drafter HF path. "
            "Defaults to the empty string, in which case "
            "fusion_mlx.speculative.dflash.drafter_registry resolves "
            "the drafter for the loaded alias. Only consulted when "
            "--spec-decode dflash is set; ignored otherwise."
        ),
    )
    # DSpark — DeepSeek DeepSpec lossless block speculative decoder.
    # Self-contained DSparkGenerator loads its own target + converted MLX
    # draft (drafter taps the target's own hidden states) and runs a
    # propose→verify loop with distribution-preserving rejection sampling
    # (lossless). Draft checkpoints: deepseek-ai/dspark_qwen3_{4b,8b,14b}
    # _block7; targets must be Qwen3 4B/8B/14B bf16. Early-forks the serve
    # path into a dedicated single-user-serial DSpark server (like audio
    # mode). PoC target: ~1.7x on Qwen3-8B-bf16 + q8 draft.
    serve_parser.add_argument(
        "--enable-dspark",
        action="store_true",
        default=False,
        help="Enable DSpark speculative decoding (DeepSeek DeepSpec lossless "
        "block spec-decode, single-user serial mode). Requires a Qwen3 "
        "bf16 target (4B/8B/14B) and a converted MLX draft — pass "
        "--dspark-drafter-path. Convert a draft with "
        "``dspark-metal-convert deepseek-ai/dspark_qwen3_8b_block7 "
        "--target mlx-community/Qwen3-8B-bf16``.",
    )
    serve_parser.add_argument(
        "--dspark-drafter-path",
        dest="dspark_drafter_path",
        default="",
        help="Path to the converted MLX DSpark draft directory (produced by "
        "dspark-metal-convert). Required when --enable-dspark is set.",
    )
    # Boot-time LoRA adapter (Phase B LoRA slice 1). Applies a PEFT LoRA
    # adapter at model load via mlx_lm.load(adapter_path=...). Single-model
    # ``serve --model`` path only; multi-model ``--model-dir`` per-model LoRA
    # and runtime hot-swap are follow-ups.
    serve_parser.add_argument(
        "--lora-path",
        dest="lora_path",
        default=None,
        help="Path to a PEFT LoRA adapter directory (adapter_config.json) to "
        "apply at boot. Fuses the adapter into the base model weights via "
        "mlx_lm.load(adapter_path=...). Single-model ``serve --model`` only; "
        "ignored by multi-model ``--model-dir``. Runtime hot-swap is not yet "
        "supported.",
    )
    serve_parser.add_argument(
        "--dspark-draft-quant-bits",
        dest="dspark_draft_quant_bits",
        type=int,
        default=8,
        help="Quantization bits for the DSpark draft model (default 8). "
        "Lower bits speed the drafter forward at some acceptance cost.",
    )
    # SuffixDecoding — drafter-free spec-decode using a suffix tree over
    # generated tokens. Big wins on agent/tool/JSON workloads (3-5x);
    # ~zero overhead on free-form chat. Pure-attention only.
    serve_parser.add_argument(
        "--suffix-decoding",
        action="store_true",
        default=False,
        help="Enable SuffixDecoding spec-decode (drafter-free, statistical). "
        "Speedup is workload-dependent: 3-5x on tool-call/JSON/code-edit, "
        "~1x on free-form chat. Auto-disabled on hybrid models "
        "(Qwen3.5/3.6, Granite4, Mamba/Jamba/RWKV).",
    )
    serve_parser.add_argument(
        "--suffix-max-draft",
        type=int,
        default=8,
        help="Max draft tokens per verify step (default: 8). "
        "Verify forward cost grows linearly with this.",
    )
    serve_parser.add_argument(
        "--suffix-max-suffix-len",
        type=int,
        default=4,
        help="Max k-gram length indexed for suffix matching (default: 4).",
    )
    serve_parser.add_argument(
        "--suffix-min-confidence",
        type=float,
        default=0.3,
        help="Vote confidence floor for draft truncation (default: 0.3). "
        "Lower → more optimistic drafts; higher → fewer but more reliable.",
    )
    serve_parser.add_argument(
        "--suffix-min-draft-len",
        type=int,
        default=2,
        help="Skip the verify forward when drafter returns fewer than "
        "this many tokens (default: 2). Protects free-form chat from "
        "verify overhead on weak 1-token drafts. Set to 1 to verify "
        "every draft (more aggressive; can regress chat).",
    )
    # Prefill step size
    serve_parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="Chunk size for prompt prefill processing. Larger values use more memory "
        "but can improve prefill throughput. (default: 2048)",
    )
    # MCP options
    serve_parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (JSON/YAML) for tool integration",
    )
    # Security options
    # ``--api-key`` accepts an inline value OR falls back to the
    # ``FUSION_MLX_API_KEY`` env var. ``fusion-mlx share`` uses the env-var
    # form so the bearer key never lands in argv (visible to ``ps`` for
    # any local user). Inline value still works for backwards-compat
    # with existing scripts; if both are set, the inline value wins.
    serve_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "API key for authentication (if not set, falls back to the "
            "FUSION_MLX_API_KEY env var; if neither, no auth required)"
        ),
    )
    serve_parser.add_argument(
        "--cors-origins",
        type=str,
        nargs="+",
        default=None,
        metavar="ORIGIN",
        help=(
            "Allowed CORS origins (default: * for all origins). "
            "Example: --cors-origins http://localhost:3000 https://myapp.com"
        ),
    )
    serve_parser.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Rate limit requests per minute per client (0 = disabled)",
    )
    # Hard cap on per-request body size — DoS defense.
    # See ``fusion_mlx/middleware/body_size.py`` for the rationale (pre-fix:
    # a 10 MB body silently ran a ~60 s full prefill on a 27B alias before
    # the client timed out; rapid-desktop#273 + #463). Default 8 MiB fits
    # a 128k-token prompt with tool schemas; 0 disables the cap.
    serve_parser.add_argument(
        "--max-request-bytes",
        type=int,
        default=None,
        help=(
            "Maximum HTTP request body size in bytes (default: 8 MiB = "
            "8388608). Requests over this cap are rejected with HTTP 413 "
            "before JSON parsing or tokenization runs. 0 disables the cap. "
            "Falls back to the FUSION_MLX_MAX_REQUEST_BYTES env var if unset."
        ),
    )
    serve_parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Default request timeout in seconds (default: 1800 = 30 min)",
    )
    # Tool calling options
    serve_parser.add_argument(
        "--enable-auto-tool-choice",
        action="store_true",
        help="Enable auto tool choice for supported models. Use --tool-call-parser to specify which parser to use.",
    )
    serve_parser.add_argument(
        "--tool-call-parser",
        type=str,
        default=None,
        # Choices NOT enforced at argparse level — the canonical set is the
        # ToolParserManager registry, which has ~39 entries (canonical
        # names + per-family aliases like ``deepseek_v31``, ``llama4``,
        # ``moonshot`` for kimi, ``nous`` for hermes). The argparse hard-
        # coded list drifted to 19 over multiple releases and rejected
        # legitimate aliases users discovered via ``fusion-mlx info``.
        # Validation now happens post-parse in
        # ``_validate_tool_call_parser_choice`` against the live registry.
        # v0.6.63 onboarding sweep finding #1.
        help=(
            "Select the tool call parser for the model. Canonical options: "
            "auto (auto-detect), mistral, qwen/qwen3/qwen3_xml (reasoning models, "
            "<tool_call>JSON</tool_call> format), qwen3_coder/qwen3_coder_xml "
            "(Coder model, <function=NAME> XML format), llama/llama3/llama4, "
            "hermes/nous, deepseek/deepseek_v3/deepseek_v31, kimi/moonshot/kimi_k2, "
            "granite/granite3, nemotron/nemotron3, xlam, functionary/meetkai, "
            "glm47/glm4, minimax/minimax_m2, harmony/gpt-oss/gpt_oss, "
            "gemma4/gemma_4, seed_oss/seed. "
            "Run `python -c 'from fusion_mlx.tool_parsers import ToolParserManager;"
            "print(sorted(ToolParserManager.tool_parsers))'` for the live list. "
            "Required for --enable-auto-tool-choice."
        ),
    )
    # Tool logits bias (jump-forward decoding for tool call structural tokens)
    serve_parser.add_argument(
        "--enable-tool-logits-bias",
        action="store_true",
        default=False,
        help="Bias logits toward structural tool call tokens for faster generation. "
        "Only active when --tool-call-parser is also set. Currently supports minimax.",
    )
    # Reasoning parser options - choices loaded dynamically from registry
    from .reasoning import list_parsers

    reasoning_choices = list_parsers()
    serve_parser.add_argument(
        "--reasoning-parser",
        type=str,
        default=None,
        choices=reasoning_choices,
        help=(
            "Enable reasoning content extraction with specified parser. "
            "Extracts <think>...</think> tags into reasoning_content field. "
            f"Options: {', '.join(reasoning_choices)}."
        ),
    )
    serve_parser.add_argument(
        "--no-thinking",
        action="store_true",
        default=False,
        help=(
            "Disable reasoning/thinking parser even if auto-detected. "
            "Thinking tokens will appear as regular content. "
            "Useful for faster responses when chain-of-thought is not needed."
        ),
    )
    # Hidden cross-alias mirroring ``chat --no-thinking`` (see the chat
    # parser for the full rationale). ``serve --no-think`` lands on the
    # same ``no_thinking`` destination so users who reach for the shorter
    # name don't get an ``unrecognized arguments`` error.
    serve_parser.add_argument(
        "--no-think",
        dest="no_thinking",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    serve_parser.add_argument(
        "--no-tool-call-parser",
        dest="no_tool_call_parser",
        action="store_true",
        default=False,
        help=(
            "Force-disable tool-call parser auto-detection from the alias "
            "profile. Escape hatch (SOP §10) when AliasProfile's auto-"
            "selected parser misfires for a specific deployment. Mutually "
            "exclusive with --tool-call-parser."
        ),
    )
    serve_parser.add_argument(
        "--no-reasoning-parser",
        dest="no_reasoning_parser",
        action="store_true",
        default=False,
        help=(
            "Force-disable reasoning parser auto-detection from the alias "
            "profile. Distinct from --no-thinking (which also suppresses "
            "the chain-of-thought prompt template) — this flag ONLY skips "
            "the auto-config step. Mutually exclusive with --reasoning-parser."
        ),
    )
    # SOP §10 profile-override escape hatches. Pair every binary
    # auto-routing field with both force-on and force-off CLI flags so
    # users always have an override path when the AliasProfile
    # auto-detection misfires. Registered in
    # tests/test_no_mllm_flag.py::test_auto_routing_flags_have_force_on_and_force_off_pair.
    serve_parser.add_argument(
        "--force-hybrid",
        dest="force_hybrid",
        action="store_true",
        default=False,
        help=(
            "Force-treat the model as a hybrid (linear-attention / Mamba) "
            "architecture even when AliasProfile says otherwise. Disables "
            "spec/suffix decode paths that are unsound on hybrids. "
            "Mutually exclusive with --no-hybrid."
        ),
    )
    serve_parser.add_argument(
        "--no-hybrid",
        dest="no_hybrid",
        action="store_true",
        default=False,
        help=(
            "Force-treat the model as non-hybrid (full attention) even when "
            "AliasProfile says it's hybrid. Use when the profile mis-labels "
            "your model and you want spec/suffix decode enabled. "
            "Mutually exclusive with --force-hybrid."
        ),
    )
    serve_parser.add_argument(
        "--force-spec-decode",
        dest="force_spec_decode",
        action="store_true",
        default=False,
        help=(
            "Force-enable speculative-decode eligibility even when "
            "AliasProfile says the model doesn't support it. Risky on "
            "hybrid models — use only when you've verified the profile "
            "is wrong. Mutually exclusive with --no-spec-decode."
        ),
    )
    serve_parser.add_argument(
        "--no-spec-decode",
        dest="no_spec_decode",
        action="store_true",
        default=False,
        help=(
            "Force-disable speculative-decode eligibility (suffix / MTP / "
            "DFlash) even when AliasProfile says the model supports it. "
            "Mutually exclusive with --force-spec-decode."
        ),
    )
    # #516 — HarmonyStreamingRouter auto-upgrade escape hatches (G11).
    # PR #515 introduced an auto-upgrade from the legacy harmony state
    # machine to openai-harmony's StreamableParser for matched-vocab
    # gpt-oss tokenizers. The auto-detection is conservative (three-layer
    # compat check) but the SOP requires every binary auto-routing
    # decision expose both force-on and force-off CLI flags.
    serve_parser.add_argument(
        "--force-openai-harmony-streaming",
        dest="force_openai_harmony_streaming",
        action="store_true",
        default=False,
        help=(
            "Force-on: construct HarmonyStreamingRouter even when the "
            "compat gate would reject. Use to debug a regression in the "
            "gate itself; production should leave this off. Mutually "
            "exclusive with --no-openai-harmony-streaming."
        ),
    )
    serve_parser.add_argument(
        "--no-openai-harmony-streaming",
        dest="no_openai_harmony_streaming",
        action="store_true",
        default=False,
        help=(
            "Force-off: skip the HarmonyStreamingRouter upgrade and use "
            "the legacy custom harmony state machine even on matched-vocab "
            "gpt-oss tokenizers. Escape hatch for a hypothetical false "
            "positive in the compat gate. Mutually exclusive with "
            "--force-openai-harmony-streaming."
        ),
    )
    # GC control (Tier 0 optimization)
    serve_parser.add_argument(
        "--gc-control",
        action="store_true",
        default=True,
        help="Enable Python GC pausing during generation to avoid latency spikes (default: enabled)",
    )
    serve_parser.add_argument(
        "--no-gc-control",
        action="store_true",
        help="Disable GC control (allow normal Python GC during generation)",
    )
    # Pinned prefix cache (Tier 0 optimization)
    serve_parser.add_argument(
        "--pin-system-prompt",
        action="store_true",
        default=False,
        help="Auto-pin system prompt in prefix cache to prevent eviction under memory pressure",
    )
    # Multimodal option
    serve_parser.add_argument(
        "--mllm",
        action="store_true",
        help="Force load model as multimodal (vision) even if name doesn't match auto-detection patterns",
    )
    serve_parser.add_argument(
        "--no-mllm",
        "--text-only",
        dest="no_mllm",
        action="store_true",
        help="Force load model as text-only LLM even when auto-detection would route it to the multimodal/VLM path. Escape hatch for incomplete vision-tower checkpoints (#393) and text-only forks of multimodal architectures whose config.json still declares vision_config.",
    )
    # Generation defaults
    serve_parser.add_argument(
        "--default-temperature",
        type=float,
        default=None,
        help="Override default temperature for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-top-p",
        type=float,
        default=None,
        help="Override default top_p for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-top-k",
        type=int,
        default=None,
        help="Override default top_k for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-min-p",
        type=float,
        default=None,
        help="Override default min_p for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-repetition-penalty",
        type=float,
        default=None,
        help="Override default repetition_penalty for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-presence-penalty",
        type=float,
        default=None,
        help="Override default presence_penalty for all requests (default: use model default)",
    )
    serve_parser.add_argument(
        "--default-frequency-penalty",
        type=float,
        default=None,
        help="Override default frequency_penalty for all requests (default: use model default)",
    )
    # Cloud routing options
    serve_parser.add_argument(
        "--cloud-model",
        type=str,
        default=None,
        help="Cloud model string for litellm (e.g. 'anthropic/claude-sonnet-4-5-20250929'). "
        "When set, large-context requests are routed to the cloud provider.",
    )
    serve_parser.add_argument(
        "--cloud-threshold",
        type=int,
        default=20000,
        help="New token threshold to trigger cloud routing (default: 20000). "
        "Only requests with more new (uncached) tokens than this are routed.",
    )
    serve_parser.add_argument(
        "--cloud-api-base",
        type=str,
        default=None,
        help="Custom API base URL for cloud model (for OpenAI-compatible providers like Zhipu).",
    )
    serve_parser.add_argument(
        "--cloud-api-key",
        type=str,
        default=None,
        help="API key for cloud model (overrides environment variable).",
    )
    # Embedding model option
    serve_parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help=(
            "Pre-load an embedding model at startup (e.g. "
            "mlx-community/embeddinggemma-300m-6bit). Requires the "
            "[embeddings] extra: pip install 'fusion-mlx[embeddings]'."
        ),
    )
    # Parent-PID watchdog (rapid-desktop issue #449). When set, the
    # sidecar polls ``os.getppid()`` every 2 s and self-terminates if
    # the parent dies (re-parent to launchd / init on macOS/Linux). The
    # supervisor passes its own PID at spawn so a SIGKILL on the desktop
    # cannot leave a 30 GB orphan holding the model + port. ``0`` /
    # negative / unset disables. The ``FUSION_MLX_WATCHDOG_PPID`` env var
    # is honoured as a fallback when the CLI flag is omitted; the flag
    # wins when both are present.
    serve_parser.add_argument(
        "--watchdog-ppid",
        type=int,
        default=None,
        metavar="PID",
        help=(
            "Self-terminate when the parent with this PID dies (defeats "
            "orphan-sidecar after SIGKILL on the supervisor). Honors "
            "$FUSION_MLX_WATCHDOG_PPID as a fallback. Set to 0 / unset to "
            "disable."
        ),
    )
    # PFlash long-prompt prefill compression (#287). Off by default; see
    # fusion_mlx/pflash.py for the design and the prefix-cache bypass.
    _add_pflash_args(serve_parser)
    # Bench command
    bench_parser = subparsers.add_parser("bench", help="Run benchmark")
    bench_parser.add_argument(
        "model", type=str, help="Model to benchmark"
    ).completer = alias_completer
    bench_parser.add_argument(
        "--force-disk-check",
        action="store_true",
        help=(
            "Skip the pre-flight disk-space check that aborts when the model "
            "is larger than free disk. Use only if you know the HF cache lives "
            "on a different filesystem (e.g. external drive via HF_HOME)."
        ),
    )
    bench_parser.add_argument(
        "--num-prompts", type=int, default=10, help="Number of prompts"
    )
    bench_parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens per prompt"
    )
    bench_parser.add_argument(
        "--max-num-seqs", type=int, default=32, help="Max concurrent sequences"
    )
    bench_parser.add_argument(
        "--prefill-batch-size", type=int, default=8, help="Prefill batch size"
    )
    bench_parser.add_argument(
        "--completion-batch-size", type=int, default=16, help="Completion batch size"
    )
    bench_parser.add_argument(
        "--enable-prefix-cache",
        action="store_true",
        default=True,
        help="Enable prefix caching (default: enabled)",
    )
    bench_parser.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        help="Disable prefix caching",
    )
    bench_parser.add_argument(
        "--prefix-cache-size",
        type=int,
        default=100,
        help="Max entries in prefix cache (default: 100, legacy mode only)",
    )
    # Memory-aware cache options (recommended for large models)
    bench_parser.add_argument(
        "--cache-memory-mb",
        type=int,
        default=None,
        help="Cache memory limit in MB (default: auto-detect ~20%% of RAM)",
    )
    bench_parser.add_argument(
        "--cache-memory-percent",
        type=float,
        default=0.20,
        help="Fraction of available RAM for cache if auto-detecting (default: 0.20)",
    )
    bench_parser.add_argument(
        "--no-memory-aware-cache",
        action="store_true",
        help="Disable memory-aware cache, use legacy entry-count based cache",
    )
    # KV cache quantization options
    bench_parser.add_argument(
        "--kv-cache-quantization",
        action="store_true",
        help="Quantize stored KV caches to reduce memory (8-bit by default)",
    )
    bench_parser.add_argument(
        "--kv-cache-quantization-bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Bit width for KV cache quantization (default: 8)",
    )
    bench_parser.add_argument(
        "--kv-cache-quantization-group-size",
        type=int,
        default=64,
        help="Group size for KV cache quantization (default: 64)",
    )
    bench_parser.add_argument(
        "--kv-cache-min-quantize-tokens",
        type=int,
        default=256,
        help="Minimum tokens for quantization to apply (default: 256)",
    )
    # Paged cache options (experimental)
    bench_parser.add_argument(
        "--use-paged-cache",
        action="store_true",
        help="Use paged KV cache for memory efficiency (experimental)",
    )
    bench_parser.add_argument(
        "--paged-cache-block-size",
        type=int,
        default=64,
        help="Tokens per cache block (default: 64)",
    )
    bench_parser.add_argument(
        "--max-cache-blocks",
        type=int,
        default=1000,
        help="Maximum number of cache blocks (default: 1000)",
    )
    # Community benchmark submission. Mutually-exclusive with the
    # freeform bench above — when --submit is set the standardized
    # B=1 runner takes over and every other knob is ignored.
    bench_parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Run the standardized B=1 community benchmark and open a PR to "
            "community-benchmarks/. Locks every comparability knob; ignores "
            "the freeform --num-prompts / --max-tokens / --max-num-seqs args."
        ),
    )
    bench_parser.add_argument(
        "--sampled",
        action="store_true",
        help=(
            "With --submit, run the bench at temp=0.7/top_p=0.9 instead of "
            "greedy. Stored as a separate 'sampled' bucket — useful for "
            "comparing against Artificial Analysis-style real-world numbers."
        ),
    )
    bench_parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help=(
            "Optional free-text annotation attached to the submission "
            "(e.g. 'on battery', 'fresh boot'). Max 200 chars."
        ),
    )
    bench_parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help=(
            "Path to the Fusion-MLX git checkout. Defaults to the current "
            "working directory. The --submit flow writes the JSON file and "
            "opens the PR from this checkout."
        ),
    )
    # --tier: user-facing tier dispatcher (PR #2). Mutually-exclusive
    # with --submit (PR #3 will consolidate them, but for now the two
    # are independent code paths).
    bench_parser.add_argument(
        "--tier",
        type=str,
        choices=["smoke", "speed", "harness", "all"],
        default=None,
        help=(
            "Run one of the standardized validation tiers: "
            "'smoke' (boot + 1 prompt), "
            "'speed' (B=1 perf probe), "
            "'harness' (5 first-class agent harnesses: "
            "codex/opencode/hermes/aider/langchain), "
            "'all' (smoke → speed → harness sequentially, abort on smoke "
            "fail). Boots the model server exactly once per invocation."
        ),
    )
    bench_parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help=(
            "For --tier: attach to an already-running server at this URL "
            "(e.g. http://localhost:8000) instead of booting one. Used by "
            "release_check_m3.sh G7b to reuse the gauntlet's server."
        ),
    )
    bench_parser.add_argument(
        "--long-prompt-tokens",
        type=int,
        default=0,
        help="Approximate reference-context tokens to prepend to each "
        "benchmark prompt. Used with --pflash auto/always for "
        "long-prompt TTFT replication (#287).",
    )
    _add_pflash_args(bench_parser)

    # Convert — HF -> MLX conversion + weight quantization (wraps mlx-lm
    # convert). fusion-mlx had no convert command; users had to call
    # ``python -m mlx_lm convert`` directly. Adds model-alias resolution so
    # ``fusion-mlx convert qwen3.5-9b --quant-bits 4`` works the same way as
    # every other fusion-mlx subcommand. Note: this is WEIGHT quantization
    # (saved to disk); TurboQuant KV-cache compression is a separate runtime
    # knob (``--kv-cache-turboquant``) and is not a weight format.
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert a HuggingFace model to MLX format (optionally quantized)",
    )
    convert_parser.add_argument(
        "model",
        help="Model alias (e.g. qwen3.5-9b) or HF repo (org/name)",
    ).completer = alias_completer
    convert_parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Output directory (default: ./<model-basename>)",
    )
    convert_parser.add_argument(
        "--quant-bits",
        type=int,
        default=None,
        choices=[2, 3, 4, 6, 8],
        help="Quantize weights to N bits (enables quantization). "
        "Omit for a plain MLX (bf16) conversion.",
    )
    convert_parser.add_argument(
        "--quant-group-size",
        type=int,
        default=64,
        help="Group size for weight quantization (default: 64)",
    )
    convert_parser.add_argument(
        "--quant-mode",
        type=str,
        default="affine",
        choices=["affine", "mxfp4", "nvfp4", "mxfp8"],
        help="Quantization mode (default: affine). affine uses --quant-bits/--quant-group-size. "
        "mxfp4/nvfp4/mxfp8 are fixed-width float modes (mlx main) that ignore "
        "--quant-bits/--quant-group-size and use per-mode defaults; setting one of "
        "these modes enables quantization on its own.",
    )
    convert_parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["bf16", "fp16", "fp32"],
        help="Cast weights to this dtype (default: keep source dtype)",
    )
    convert_parser.add_argument(
        "--dequantize",
        action="store_true",
        default=False,
        help="Dequantize a quantized model back to float",
    )
    convert_parser.add_argument(
        "--upload-repo",
        default=None,
        help="Upload the converted model to this HF repo (e.g. org/name)",
    )
    convert_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Allow custom modeling code from the source repo",
    )

    # Models command. ``ls`` is registered as a top-level alias that
    # defaults to ``models --cached`` (the locally-cached view) — two
    # muscle-memory entry points, one underlying impl.
    models_parser = subparsers.add_parser("models", help="List available model aliases")
    models_parser.add_argument(
        "--cached",
        action="store_true",
        default=False,
        help="Only list models that are downloaded to the local HuggingFace "
        "cache (alias, HF repo, size on disk, last modified).",
    )
    # Released 1.0/2.0/3.0 `models` contract (docs/cli-reference.md): the
    # default view queries the running server's /v1/models, so --host/--port
    # target it. Restored after the Rapid-MLX migration dropped them; defaults
    # match the released parser (localhost:8000).
    models_parser.add_argument(
        "--host", default="localhost", help="Server host (default: localhost)"
    )
    models_parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )
    subparsers.add_parser(
        "ls",
        help="List models in the local HuggingFace cache (alias for `models --cached`)",
    )

    # Version + help — utility commands that mirror the existing flags but
    # are scriptable as plain subcommands.
    subparsers.add_parser("version", help="Show version number")
    help_parser = subparsers.add_parser("help", help="Show help for a subcommand")
    help_parser.add_argument(
        "subcommand", nargs="?", help="Subcommand to show help for (omit for top-level)"
    )

    # Pull / rm / ps — Ollama-style cache and process management.
    pull_parser = subparsers.add_parser(
        "pull", help="Download a model to the HuggingFace cache (no server)"
    )
    pull_parser.add_argument(
        "model", help="Model alias (e.g. qwen3.5-4b-4bit) or HF repo (org/name)"
    ).completer = alias_completer
    rm_parser = subparsers.add_parser(
        "rm", help="Remove a cached model from the HuggingFace cache"
    )
    rm_parser.add_argument(
        "model", help="Model alias (e.g. qwen3.5-4b-4bit) or HF repo (org/name)"
    ).completer = alias_completer
    rm_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt and remove the model immediately.",
    )
    subparsers.add_parser("ps", help="List running fusion-mlx servers")

    # Lifecycle — managed background server control (macOS app / Homebrew).
    # Delegates to the FusionMLX.app control socket or `brew services`.
    for _name, _help_text in (
        ("start", "Start fusion-mlx as a managed background server"),
        ("stop", "Stop the managed background fusion-mlx server"),
        ("restart", "Restart the managed background fusion-mlx server"),
    ):
        _lifecycle_parser = subparsers.add_parser(
            _name,
            help=_help_text,
            description=_help_text,
        )
        _lifecycle_parser.add_argument(
            "--timeout",
            type=float,
            default=60.0,
            help="Seconds to wait for the macOS app/server to reach the requested state",
        )
        if _name in {"start", "restart"}:
            _lifecycle_parser.add_argument(
                "--no-wait",
                action="store_true",
                help="Return after sending the request without waiting for server health",
            )

    # Upgrade — detect install method and run the right upgrade command
    upgrade_parser = subparsers.add_parser(
        "upgrade",
        help="Upgrade fusion-mlx to the latest version (brew / pip / install.sh)",
    )
    upgrade_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt and run the upgrade immediately.",
    )
    upgrade_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the detected install method and the upgrade command, "
            "then exit without running it."
        ),
    )

    # Chat — interactive REPL backed by a (spawned or existing) server.
    # ``run`` is exposed as a subparser alias purely for Ollama-muscle-memory
    # parity (``ollama run <model>``). Both names route to ``chat_command``.
    chat_parser = subparsers.add_parser(
        "chat",
        aliases=["run"],
        help="Interactive chat REPL with a model",
        description=(
            "Interactive chat REPL with a model.\n\n"
            "Note: 'fusion-mlx run' is an alias for 'chat' (Ollama compatibility)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # See serve_parser for the rationale: ``--think``/``--no-think`` +
        # ``--thinking``/``--no-thinking`` cross-aliases create ambiguous
        # prefixes that argparse silently resolves to whichever flag was
        # added first.
        allow_abbrev=False,
    )
    chat_parser.add_argument(
        "model",
        nargs="?",
        default="qwen3.5-4b-4bit",
        help="Model alias (e.g. qwen3.5-4b-4bit) or HF repo (org/name). "
        "Defaults to qwen3.5-4b-4bit when omitted.",
    ).completer = alias_completer
    chat_parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System prompt prepended to the conversation",
    )
    chat_parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable thinking/reasoning mode (default: off in chat REPL — "
        "reasoning models like Qwen3.5 otherwise leak raw chain-of-thought "
        "and can loop until max-tokens). Use --think to surface reasoning, "
        "--no-think is also accepted for back-compat.",
    )
    # Hidden cross-alias for users who picked up the ``--no-thinking`` muscle
    # memory from ``fusion-mlx serve``. ``serve --no-thinking`` and
    # ``chat --no-think`` mean different things internally (server-side
    # parser disable vs. per-request ``enable_thinking=false``), but the
    # flag-name difference trips users. We accept the wrong-side name as
    # an alias for the right-side semantics: ``chat --no-thinking`` simply
    # forwards to the same destination as ``--no-think``.
    chat_parser.add_argument(
        "--no-thinking",
        dest="think",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    chat_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens per assistant response (default: 2048; raised to "
        "4096 when --think is set so reasoning + answer fit the budget).",
    )
    chat_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    chat_parser.add_argument(
        "--port",
        type=_port_arg,
        default=None,
        help="Connect to existing server on 127.0.0.1:<port> instead of spawning",
    )
    chat_parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Connect to existing server URL (e.g. http://host:8000) "
        "instead of spawning. Overrides --port.",
    )
    chat_parser.add_argument(
        "--ready-timeout",
        type=int,
        default=600,
        help="Seconds to wait for the spawned server to become ready (default: 600)",
    )
    chat_parser.add_argument(
        "--response-timeout",
        type=int,
        default=600,
        help="Seconds to wait for a single assistant response (default: 600)",
    )

    # Info command — show the per-model profile (parsers + capability gates)
    info_parser = subparsers.add_parser(
        "info",
        help="Show the per-model profile for a model name or alias",
    )
    info_parser.add_argument(
        "model",
        help="Model alias (e.g. qwen3.5-4b-4bit) or HF repo (e.g. mlx-community/SmolLM3-3B-4bit)",
    ).completer = alias_completer

    # Agents command
    agents_parser = subparsers.add_parser(
        "agents", help="List, configure, and test agent integrations"
    )
    agents_parser.add_argument(
        "agent_name",
        nargs="?",
        default=None,
        help="Agent name (e.g. hermes, goose, aider). Omit to list all.",
    )
    agents_parser.add_argument(
        "--setup",
        action="store_true",
        help="Auto-configure the agent to point at this server",
    )
    agents_parser.add_argument(
        "--test",
        action="store_true",
        help="Run integration tests for this agent",
    )
    agents_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: auto-detect from running server)",
    ).completer = alias_completer
    agents_parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Fusion-MLX server URL (default: http://localhost:8000/v1)",
    )
    agents_parser.add_argument(
        "--agent-version",
        type=str,
        default=None,
        help="Agent version for version-specific config (e.g. 0.8.5)",
    )

    # Doctor command — pure env-health probe (≤5 s, no model load, no server).
    # Model-validation tiers (smoke/check/full/benchmark) moved to
    # ``fusion-mlx bench --tier ...`` as of v0.7.22.
    #
    # The legacy positional ``tier`` plus ``--model``, ``--models``, and
    # ``--update-baselines`` are intentionally retained (SUPPRESSed from
    # --help) for one release so users hitting the old form
    # ``fusion-mlx doctor check --model qwen3.5-9b-4bit`` get the actionable
    # bench redirect from ``doctor_command`` instead of an argparse
    # ``unrecognized arguments`` wall. Codex review round 1 flagged this:
    # rejecting at argparse-time defeated the redirect. Drop these in a
    # future release once telemetry confirms no one's still calling them.
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check environment health (Python, packages, HF cache, network, ...)",
    )
    doctor_parser.add_argument(
        "tier",
        nargs="?",
        default=None,
        choices=["smoke", "check", "full", "benchmark"],
        help=argparse.SUPPRESS,
    )
    doctor_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print the underlying probe detail for each check",
    )
    # Legacy compatibility shims — accepted-but-ignored so the redirect
    # message in ``doctor_command`` can fire (see comment above).
    doctor_parser.add_argument(
        "--model",
        default=None,
        help=argparse.SUPPRESS,
    )
    doctor_parser.add_argument(
        "--models",
        default=None,
        help=argparse.SUPPRESS,
    )
    doctor_parser.add_argument(
        "--update-baselines",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    # Telemetry subcommand — opt-in anonymous usage data (Issue #236).
    # See fusion_mlx/telemetry/ for what we collect / don't collect, and
    # the README "Telemetry" section for the user-facing summary.
    telemetry_parser = subparsers.add_parser(
        "telemetry",
        help="Manage anonymous usage telemetry (opt-in)",
    )
    telemetry_subparsers = telemetry_parser.add_subparsers(
        dest="telemetry_action",
        help="Telemetry actions",
    )
    telemetry_subparsers.add_parser(
        "status", help="Show whether telemetry is enabled and why"
    )
    telemetry_subparsers.add_parser(
        "enable", help="Opt in to anonymous usage telemetry"
    )
    telemetry_subparsers.add_parser(
        "disable", help="Opt out of anonymous usage telemetry"
    )
    telemetry_subparsers.add_parser(
        "preview",
        help="Print a sample payload showing exactly what telemetry would send",
    )
    telemetry_subparsers.add_parser(
        "reset",
        help="Delete the consent + client-id files (next run re-prompts)",
    )

    # Share subcommand — expose a local serve behind a public fusionmlx.com URL.
    try:
        from fusion_mlx.share.cli import register as _register_share

        _register_share(subparsers)
    except ModuleNotFoundError:
        pass

    # Launch subcommand — one-shot bootstrap that patches IDE/agent
    # client configs (Cline, Claude Code, Continue, Cursor) to route
    # at the local fusion-mlx server. See GH issue #566 for motivation.
    # Registered AFTER share so the help-text ordering reads
    # serve→…→share→launch, matching the rough "more common first" flow.
    try:
        from fusion_mlx.launch.cli import register as _register_launch

        _register_launch(subparsers)
    except ModuleNotFoundError:
        pass

    # Shell tab completion via argcomplete. Must fire before parse_args:
    # when the shell completion handler invokes us with the
    # ``_ARGCOMPLETE`` env var set, this function short-circuits before
    # any heavy import paths or model resolution runs, so the user gets
    # snappy ``fusion-mlx chat gemma-4-<TAB>`` even on a cold shell.
    #
    # ``_action_conflicts`` and ``_seen_non_default_actions`` are
    # populated by argcomplete inside ``IntrospectiveArgumentParser._
    # parse_known_args`` — but option completion (``finders.py:_
    # action_allowed``) reads them before parsing has run on a
    # subparser, raising ``AttributeError`` on the first Tab. We
    # pre-walk the parser tree and seed empty containers so completion
    # works at the very first keystroke. Issue tracked upstream at
    # kislyuk/argcomplete (no mutex groups → conflict set is just
    # empty; this is the documented null-init).
    def _preinit_argcomplete_state(p: argparse.ArgumentParser) -> None:
        if not hasattr(p, "_action_conflicts"):
            p._action_conflicts = {}  # type: ignore[attr-defined]
        if not hasattr(p, "_seen_non_default_actions"):
            p._seen_non_default_actions = set()  # type: ignore[attr-defined]
        if not hasattr(p, "active_actions"):
            p.active_actions = []  # type: ignore[attr-defined]
        for action in p._actions:
            if isinstance(action, argparse._SubParsersAction):
                for sub in action.choices.values():
                    if isinstance(sub, argparse.ArgumentParser):
                        _preinit_argcomplete_state(sub)

    _preinit_argcomplete_state(parser)
    try:
        import argcomplete
    except ModuleNotFoundError as exc:
        # Best-effort: tab completion silently no-ops if argcomplete is
        # missing. Listed as a required dep in ``pyproject.toml`` so
        # this path only fires in minimal test envs or stripped images.
        # Narrow the swallow to the top-level argcomplete package — if a
        # transitive import inside argcomplete is missing we want that
        # to surface, not get mistaken for "argcomplete not installed".
        if exc.name != "argcomplete":
            raise
    else:
        argcomplete.autocomplete(parser)

    args = parser.parse_args()

    # First-run consent prompt — fires at most once per machine, only on
    # interactive subcommands when stdin is a tty. Safe no-op otherwise.
    # Must run *before* heavy subcommand work so the user sees the
    # disclosure before any model load logs scroll past.
    _just_collected_consent = False
    if getattr(args, "command", None) is not None:
        from fusion_mlx.telemetry import maybe_prompt_for_consent
        from fusion_mlx.telemetry.state import set_cli_kill_switch

        # ``--no-telemetry`` is a per-run override; thread it into the
        # process-level kill switch so every emit site sees it without
        # having to plumb the flag through every signature.
        set_cli_kill_switch(getattr(args, "no_telemetry", False))

        _just_collected_consent = maybe_prompt_for_consent(
            args.command,
            cli_no_telemetry=getattr(args, "no_telemetry", False),
        )

    # Telemetry session lifecycle — emit session_start once we know what
    # subcommand we're dispatching, register an atexit hook for
    # session_end so the duration covers the whole interactive run
    # (including ``fusion-mlx chat`` REPLs and ``serve`` processes that
    # only exit on Ctrl-C). emit.* helpers are individually guarded by
    # ``is_enabled()`` — when telemetry is off the calls are cheap
    # no-ops, no payload constructed.
    #
    # The ``telemetry`` subcommand itself is excluded: ``telemetry
    # disable`` / ``reset`` would otherwise queue an event on the way to
    # turning telemetry OFF — a small but ugly "phone home before
    # silencing the phone" surprise that codex round 1 caught. ``status``
    # / ``preview`` / ``enable`` are excluded for consistency; their
    # observability value is near zero.
    #
    # ``_just_collected_consent`` skips the run that JUST collected
    # first-time opt-in (round 3 codex catch): the disclosure copy
    # promises "nothing from before this prompt or from a session you
    # opted out of", and the current invocation's argv was determined
    # BEFORE the user said yes. The next run starts the contract clean.
    #
    # ``_session_models_requested`` is hoisted outside the conditional so
    # the alias-resolution block below can append to it unconditionally
    # without a NameError when telemetry was skipped. The closure
    # passed to ``session_end`` reads the same list, so populate-then-
    # emit is naturally ordered.
    #
    # Round 19 codex catch on the naming: this list captures models
    # the user's invocation REQUESTED -- the alias passed argparse
    # validation -- NOT models the loader confirmed it loaded. A
    # declined auto-pull or a load failure later in the subcommand
    # handler still leaves the entry here, which the lifecycle event
    # surfaces verbatim. Phase 2.2 will replace this with confirmed
    # load events emitted from ``fusion_mlx/engine/loader.py``; until
    # then, the field semantics is "alias the session was for" and the
    # helper docstring spells this out.
    _session_models_requested: list[str] = []
    if (
        getattr(args, "command", None) is not None
        and args.command != "telemetry"
        and not _just_collected_consent
    ):
        import atexit as _atexit
        import sys as _sys
        import time as _time

        from fusion_mlx.telemetry import emit as _telemetry_emit

        _session_subcommand = args.command
        _session_started_at = _time.monotonic()
        # Round 19 codex catch: extract flag names HERE so raw argv
        # tokens (which include flag VALUES) never cross into the
        # telemetry helper signatures. The disclosure promise "values
        # are never even read" is now literally true at the function-
        # call boundary.
        from fusion_mlx.telemetry.redact import (
            hash_flag_names as _telemetry_extract_flag_names,
        )

        _session_flag_names = _telemetry_extract_flag_names(_sys.argv[1:])
        # Round 19 codex NIT: session_start sees an empty IMMUTABLE
        # snapshot of models_loaded so it does not depend on whether
        # ``emit.session_start()`` eagerly copies its input. The closure-
        # captured list keeps mutating until session_end takes its own
        # snapshot below.
        _telemetry_emit.session_start(
            subcommand=_session_subcommand,
            flag_names=_session_flag_names,
            models_loaded=(),
        )

        def _emit_session_end() -> None:
            try:
                # Snapshot the closure-captured list to an immutable
                # tuple so the payload reflects the exact state at this
                # call (round 19 NIT).
                _models_snapshot = tuple(_session_models_requested)
                _telemetry_emit.session_end(
                    subcommand=_session_subcommand,
                    duration_seconds=int(_time.monotonic() - _session_started_at),
                    models_loaded=_models_snapshot,
                )
                # Round 5 codex review caught that the atexit handler
                # for the queue's ``shutdown`` is registered inside
                # ``session_start`` (LIFO → runs after this handler),
                # but relying on that ordering is fragile. Force a
                # synchronous drain here so ``session_end`` actually
                # lands regardless of atexit ordering quirks. Idempotent
                # — the queue's own ``shutdown`` will be a no-op when
                # it runs later.
                #
                # ``session_end`` is best-effort by design (round 7
                # codex catch): the queue's own ``SHUTDOWN_BUDGET_S``
                # (2 s) caps user-visible exit latency. A slow or
                # blackholed collector drops the event — that is the
                # right trade-off, because making the user wait
                # ~12 s on every ``serve`` Ctrl-C just to file a
                # better stat is hostile UX.
                #
                # Round 19 codex review closed the previous round-17
                # SIGTERM gap: ``register_session_end_hook`` is wired
                # below so the FastAPI lifespan shutdown in
                # ``fusion_mlx.server`` calls this same function on
                # SIGTERM (systemd / Docker / Kubernetes graceful
                # stop). The latch inside the emit module makes the
                # second invocation a no-op so the event lands exactly
                # once regardless of which path fires first.
                #
                # ``_queue is None`` (telemetry was disabled, so
                # ``session_end`` no-op'd and never instantiated the
                # singleton) skips ``get_queue()`` — round 7 catch —
                # otherwise we'd spawn a daemon thread during
                # interpreter shutdown for nothing.
                try:
                    if _telemetry_emit._queue is not None:
                        _telemetry_emit._queue.shutdown()
                except BaseException:
                    pass
            except BaseException:
                # atexit handlers are run during interpreter shutdown;
                # anything that fires here — including a stray
                # ``KeyboardInterrupt`` or ``SystemExit`` raised inside
                # redaction / queue code mid-teardown — is purely noise
                # at this point because the process is already exiting.
                # Round 9 codex review caught the previous ``Exception``
                # catch as too narrow for an atexit context.
                return

        # Register the same callable for both teardown paths. The
        # latch in ``fire_session_end_hook`` ensures it runs once
        # regardless of which path (FastAPI lifespan shutdown OR cli
        # atexit fallback) fires first.
        _telemetry_emit.register_session_end_hook(_emit_session_end)
        _atexit.register(_telemetry_emit.fire_session_end_hook)

    # Resolve model aliases before dispatch.
    #
    # The doctor subcommand is exempt for historical reasons (and as a
    # belt-and-suspenders guard now that doctor doesn't take ``--model``):
    # an env-health probe should never trigger an alias→path lookup.
    if (
        hasattr(args, "model")
        and args.model
        and getattr(args, "command", None) != "doctor"
    ):
        from fusion_mlx.model_aliases import resolve_model

        resolved = resolve_model(args.model)
        if resolved != args.model:
            print(f"  Alias: {args.model} → {resolved}")
            args._original_alias = args.model
            args.model = resolved
        elif "/" not in args.model and not os.path.exists(args.model):
            # R8-M5 (Bo 0.8.9 dogfood): short audio aliases (``kokoro``,
            # ``whisper``, ``parakeet``, ``chatterbox``, ``vibevoice``,
            # ``voxcpm``) and their full-form siblings (``kokoro-82m-
            # 8bit``) are NOT in ``aliases.json`` — the resolver returns
            # them unchanged, then this fail-fast branch trips with
            # "is not a known alias or HuggingFace path" BEFORE
            # ``serve_command`` can run the audio boot guard. On a
            # fresh ``pip install fusion-mlx`` (no ``[audio]`` extra)
            # that means the operator sees a generic "unknown alias"
            # instead of the actionable "install fusion-mlx[audio]"
            # hint, and on a healthy install with ``[audio]`` the
            # short alias resolves at request time inside the audio
            # routes (``TTS_MODEL_ALIASES`` / ``STT_MODEL_ALIASES``)
            # but the CLI exits before serve_command ever runs.
            #
            # Skip the fail-fast for audio aliases so:
            #   - missing-extra installs reach the audio boot guard
            #     in ``serve_command`` (rc=2 + install hint).
            #   - healthy installs reach the audio routes' alias
            #     resolution and serve correctly.
            # The substring check matches the same alias surface the
            # serve-command boot guard uses (``_AUDIO_ALIAS_TOKENS``)
            # so a name that trips one trips the other — no risk of a
            # text/vision alias accidentally bypassing the fail-fast.
            from .audio.probe import is_audio_model_alias

            if not is_audio_model_alias(args.model):
                # Not an alias, not a HuggingFace org/name path, not a
                # local directory, not an audio alias — fail fast with
                # suggestions instead of letting the request hit
                # HuggingFace and 404 with a 30-line stack trace.
                print(
                    f"\n  Error: '{args.model}' is not a known alias or HuggingFace path."
                )
                _print_unknown_model_help(
                    args.model, full_path_example="mlx-community/Qwen3.5-9B-4bit"
                )
                sys.exit(1)
        # Round 16 codex catch: record the resolved (or already-canonical)
        # model so ``session_end`` can report what this invocation loaded.
        # ``normalize_model_path`` inside the emit helper redacts local
        # paths to the literal ``<local>`` token, so we don't need to
        # filter here. Captured after the error-fail path so we never
        # record a model that failed validation.
        _session_models_requested.append(args.model)

    # --- BEGIN B2: auto-pull confirmation gate -------------------------
    # For subcommands that may trigger a first-time download of a large
    # repo (chat/run/serve/pull/bench), warn the user before kicking off
    # a multi-GB transfer. Cached repos and small downloads pass through
    # invisibly. Env override: FUSION_MLX_AUTO_PULL=1. See
    # ``fusion_mlx/_download_gate.py`` for the policy.
    #
    # Codex round 1 surfaced two ordering issues:
    #   (a) the chat REPL spawns its own ``serve`` subprocess after the
    #       parent already gated; without FUSION_MLX_CHAT_SPAWN=1 in the
    #       child env, the second main() would re-prompt (or worse,
    #       deadlock on a non-TTY child stdin path that doesn't reach
    #       the early-return).
    #   (b) the env / TTY checks belong *before* the 5-second HF
    #       metadata fetch — otherwise every CI run that sets
    #       FUSION_MLX_AUTO_PULL=1 still pays the network round-trip.
    # Single-use marker: pop the env var as soon as we observe it so a
    # grandchild ``fusion-mlx`` spawn (e.g. a nested invocation from a
    # user hook, a doctor self-probe, or some future hub helper) does
    # NOT inherit the bypass. Codex round-2 BLOCKING #2.
    _chat_spawn_child = os.environ.pop("FUSION_MLX_CHAT_SPAWN", "") == "1"

    _GATED_COMMANDS = {"chat", "run", "serve", "pull", "bench"}
    if (
        getattr(args, "command", None) in _GATED_COMMANDS
        and hasattr(args, "model")
        and args.model
        and "/" in args.model  # only HF-style repo ids; local paths skip
        and not os.path.exists(args.model)
        and not _chat_spawn_child
    ):
        # Cheap checks first: env override and non-TTY both short-circuit
        # without touching the HF API. ``confirm_or_abort`` re-checks
        # both internally; we mirror them here so we can skip the size
        # estimate as well.
        _env_val = os.environ.get("FUSION_MLX_AUTO_PULL", "").strip().lower()
        _auto_yes = _env_val in {"1", "true", "yes"}
        _interactive = sys.stdin.isatty()
        if not _auto_yes and _interactive:
            from fusion_mlx._download_gate import (
                confirm_or_abort,
                estimate_repo_size_bytes,
                is_repo_cached,
            )

            if not is_repo_cached(args.model):
                confirm_or_abort(
                    args.model,
                    estimate_repo_size_bytes(args.model),
                )
    # --- END B2 --------------------------------------------------------

    if args.command == "serve":
        serve_command(args)
    elif args.command == "bench":
        bench_command(args)
    elif args.command == "convert":
        from fusion_mlx.cli_convert import convert_command

        sys.exit(convert_command(args))
    elif args.command == "models":
        models_command(args)
    elif args.command == "ls":
        # `ls` is a top-level alias for `models --cached`. Synthesize the
        # missing flag so models_command's branch fires without having to
        # know which command name it was invoked under.
        args.cached = True
        models_command(args)
    elif args.command == "version":
        print(f"fusion-mlx {_version}")
    elif args.command == "help":
        target = getattr(args, "subcommand", None)
        if not target:
            parser.print_help()
        elif target in subparsers.choices:
            subparsers.choices[target].print_help()
        else:
            import difflib

            print(f"Unknown subcommand: {target}")
            matches = difflib.get_close_matches(
                target, list(subparsers.choices.keys()), n=3, cutoff=0.6
            )
            if matches:
                print(f"  Did you mean: {', '.join(matches)}?")
            print("Run `fusion-mlx help` for the list of subcommands.")
            sys.exit(1)
    elif args.command == "pull":
        pull_command(args)
    elif args.command == "rm":
        rm_command(args)
    elif args.command == "ps":
        ps_command(args)
    elif args.command in ("start", "stop", "restart"):
        from fusion_mlx.cli_lifecycle import lifecycle_command

        sys.exit(lifecycle_command(args))
    elif args.command == "upgrade":
        upgrade_command(args)
    elif args.command in ("chat", "run"):
        # ``run`` is exposed as a subparser alias for Ollama compatibility;
        # argparse routes via ``aliases=`` but reports the user-typed name
        # on ``args.command``. Both names land here.
        chat_command(args)
    elif args.command == "info":
        info_command(args)
    elif args.command == "agents":
        agents_command(args)
    elif args.command == "doctor":
        doctor_command(args)
    elif args.command == "telemetry":
        telemetry_command(args)
    elif args.command == "share":
        from fusion_mlx.share.cli import share_command

        share_command(args)
    elif args.command == "launch":
        from fusion_mlx.launch.cli import launch_command

        launch_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
