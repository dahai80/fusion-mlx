# SPDX-License-Identifier: Apache-2.0
# O5.6 CLI param grouping: reorganize argparse --help output into
# category groups WITHOUT touching the 111 add_argument calls.
#
# Approach: post-process parser._action_groups after all args are defined.
# Move each optional action into a category group based on its dest/option
# string. Actions not in the mapping stay in the default "optional" group.
# Zero risk to argument parsing — only reorganizes help rendering.

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)

# dest -> category. The 5 top-frequency args land in "Core" so they appear
# first (argparse renders groups in insertion order, Core is created first).
_SERVE_CATEGORIES: dict[str, str] = {
    # Core (high-frequency, shown first)
    "model": "Core",
    "port": "Core",
    "host": "Core",
    "api_key": "Core",
    "log_level": "Core",
    # Memory
    "memory_tier": "Memory",
    "custom_limit_mb": "Memory",
    "per_engine_pct": "Memory",
    "ssd_cache_enabled": "Memory",
    "ssd_cache_dir": "Memory",
    "ssd_cache_max_bytes": "Memory",
    "gpu_memory_utilization": "Memory",
    "cache_memory_mb": "Memory",
    "cache_memory_percent": "Memory",
    # Scheduler
    "max_num_seqs": "Scheduler",
    "max_num_batched_tokens": "Scheduler",
    "prefill_batch_size": "Scheduler",
    "completion_batch_size": "Scheduler",
    "prefill_step_size": "Scheduler",
    "chunked_prefill_tokens": "Scheduler",
    "max_concurrent_requests": "Scheduler",
    "max_waiting": "Scheduler",
    "policy": "Scheduler",
    # KV cache
    "kv_cache_quantization": "KV cache",
    "kv_cache_quantization_bits": "KV cache",
    "kv_cache_quantization_group_size": "KV cache",
    "kv_cache_min_quantize_tokens": "KV cache",
    "kv_cache_turboquant": "KV cache",
    "kv_cache_turboquant_bits": "KV cache",
    "kv_cache_turboquant_group_size": "KV cache",
    "kv_cache_turboquant_mode": "KV cache",
    "kv_cache_dtype": "KV cache",
    "use_paged_cache": "KV cache",
    "paged_cache_block_size": "KV cache",
    "max_cache_blocks": "KV cache",
    "use_memory_aware_cache": "KV cache",
    # Prefix cache
    "enable_prefix_cache": "Prefix cache",
    "prefix_cache_size": "Prefix cache",
    "prefix_cache_index": "Prefix cache",
    # Speculative decode
    "spec_decode": "Speculative decode",
    "enable_mtp": "Speculative decode",
    "mtp_num_draft_tokens": "Speculative decode",
    "mtp_optimistic": "Speculative decode",
    "mtp_sidecar": "Speculative decode",
    "mtp_model_type": "Speculative decode",
    "enable_suffix_decoding": "Speculative decode",
    "suffix_max_draft": "Speculative decode",
    "suffix_max_suffix_len": "Speculative decode",
    "suffix_min_confidence": "Speculative decode",
    "suffix_min_draft_len": "Speculative decode",
    "enable_dspark": "Speculative decode",
    "enable_dflash2": "Speculative decode",
    "dflash2_block_size": "Speculative decode",
    "dflash2_draft_bits": "Speculative decode",
    "dflash_drafter_path": "Speculative decode",
    "dflash2_drafter_path": "Speculative decode",
    "dspark_drafter_path": "Speculative decode",
    "dspark_draft_quant_bits": "Speculative decode",
    # Sampling
    "default_max_tokens": "Sampling",
    "default_temperature": "Sampling",
    "default_top_p": "Sampling",
    "default_top_k": "Sampling",
    "default_min_p": "Sampling",
    "default_repetition_penalty": "Sampling",
    "default_presence_penalty": "Sampling",
    "default_frequency_penalty": "Sampling",
    "thinking_token_budget": "Sampling",
    "no_thinking": "Sampling",
    "pin_system_prompt": "Sampling",
    # Tool calling
    "enable_auto_tool_choice": "Tool calling",
    "tool_call_parser": "Tool calling",
    # Networking
    "cloud_router_enabled": "Networking",
    "cloud_router_model": "Networking",
    "cloud_router_api_key": "Networking",
    "cloud_router_threshold": "Networking",
    "cloud_fallback_consent": "Networking",
    "cluster_advertise": "Networking",
    "cluster_lb_enabled": "Networking",
    "cluster_peers": "Networking",
    "cluster_weights": "Networking",
    "cluster_lb_health_interval": "Networking",
    "cluster_lb_health_max_missed": "Networking",
    "platform": "Networking",
    "uds": "Networking",
    # Behavior
    "gc_control": "Behavior",
    "core_behavioral_prompt": "Behavior",
    "profile": "Behavior",
    "kv_disk_checkpoint_interval": "Behavior",
}


def regroup_serve_help(parser: argparse.ArgumentParser) -> None:
    # Move optional actions into category groups based on _SERVE_CATEGORIES.
    # Positional arguments stay in the positional group. Unknown dests stay
    # in the default optional group (rendered last as "Other options").
    if not hasattr(parser, "_action_groups"):
        return
    default_optional = None
    for grp in parser._action_groups:
        title = getattr(grp, "title", "")
        if title in ("optional arguments", "options"):
            default_optional = grp
            break
    if default_optional is None:
        return
    cat_groups: dict[str, argparse._ArgumentGroup] = {}
    category_order = [
        "Core",
        "Memory",
        "Scheduler",
        "KV cache",
        "Prefix cache",
        "Speculative decode",
        "Sampling",
        "Tool calling",
        "Networking",
        "Behavior",
    ]
    for cat in category_order:
        cat_groups[cat] = parser.add_argument_group(f"{cat} options")
    moved = 0
    remaining = []
    for action in list(default_optional._group_actions):
        dest = getattr(action, "dest", "")
        cat = _SERVE_CATEGORIES.get(dest)
        if cat and cat in cat_groups:
            cat_groups[cat]._group_actions.append(action)
            default_optional._group_actions.remove(action)
            moved += 1
        else:
            remaining.append(dest)
    if remaining:
        other = parser.add_argument_group("Other options")
        for action in list(default_optional._group_actions):
            other._group_actions.append(action)
            default_optional._group_actions.remove(action)
    logger.debug(
        "regroup_serve_help: moved %d/%d args into categories",
        moved,
        moved + len(remaining),
    )


__all__ = ["regroup_serve_help"]
