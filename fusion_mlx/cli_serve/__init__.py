# SPDX-License-Identifier: Apache-2.0
"""cli_serve package — serve/bench CLI commands, split from the original
cli_serve.py monolith (139K) into responsibility-focused submodules.

Public surface preserved: ``serve_command``, ``bench_command`` and the
helpers downstream code/tests import.
"""

from .audio_mode import (
    _display_host,
    _load_embedding_model_or_exit,
    _serve_audio_mode,
)
from .bench_command import bench_command
from .config_resolve import (
    _add_pflash_args,
    _apply_mtp_cli_model_type_reconciliation,
    _autoconfig_parsers,
    _boot_guard_checks,
    _build_benchmark_context,
    _print_profile_banner,
    _print_startup_banner,
    _serve_from_model_dir,
    _stage_server_config,
)
from .model_download import (
    _ensure_model_downloaded,
    _try_mirror_prefetch,
)
from .preflight import (
    _check_disk_space,
    _check_memory_capacity,
    _gather_kv_cache_dtype_inputs,
)
from .serve_command import serve_command
from .submit_flow import (
    _run_submit_flow,
    _run_tier_submit_flow,
)

__all__ = [
    "bench_command",
    "serve_command",
    "_add_pflash_args",
    "_build_benchmark_context",
    "_check_disk_space",
    "_check_memory_capacity",
    "_display_host",
    "_ensure_model_downloaded",
    "_gather_kv_cache_dtype_inputs",
    "_load_embedding_model_or_exit",
    "_apply_mtp_cli_model_type_reconciliation",
    "_print_profile_banner",
    "_print_startup_banner",
    "_run_submit_flow",
    "_run_tier_submit_flow",
    "_serve_audio_mode",
    "_serve_from_model_dir",
    "_stage_server_config",
    "_try_mirror_prefetch",
]
