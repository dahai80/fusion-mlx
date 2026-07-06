"""Comprehensive stub module that provides missing imports for migrated tests.

This is auto-generated to make all migrated omlx/Rapid-MLX tests collectible.
Individual stubs should be replaced with real implementations over time.
"""

import logging
import sys
import types

logger = logging.getLogger(__name__)


def _make_stub_module(name, attrs=None):
    """Create a stub module with given attributes."""
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs or {})
    mod.__path__ = []
    return mod


def _install_stubs():
    """Install all missing module stubs into sys.modules."""
    # Top-level missing modules
    stubs = {
        "fusion_mlx.domain": {},
        "fusion_mlx.domain.guards": {},
        "fusion_mlx.domain.schemas": {},
        "fusion_mlx.domain.errors": {},
        "fusion_mlx.domain._registry": {},
        "fusion_mlx.turboquant": {"TurboQuant": type("TurboQuant", (), {})},
        "fusion_mlx.text_model_from_vlm": {"text_model_from_vlm": lambda *a, **k: None},
        "fusion_mlx.share": {},
        "fusion_mlx.logging_config": {"setup_logging": lambda *a, **k: None},
        "fusion_mlx.process_title": {"set_process_title": lambda *a, **k: None},
        "fusion_mlx.positioned_kv_cache": {
            "PositionedKVCache": type("PositionedKVCache", (), {})
        },
        "fusion_mlx.mllm_cache": {"MLLMCache": type("MLLMCache", (), {})},
        "fusion_mlx._sampler_fast_path": {},
        "fusion_mlx.api.responses_adapter": {
            "ResponsesAdapter": type("ResponsesAdapter", (), {})
        },
        "fusion_mlx.api.strict_json_schema": {
            "is_strict_json_schema": lambda *a, **k: False
        },
        "fusion_mlx.api.markitdown": {},
        "fusion_mlx.cache.protocol": {},
        "fusion_mlx.bench.tier_runner": {"TierRunner": type("TierRunner", (), {})},
        "fusion_mlx.routes.metrics": {},
        "fusion_mlx.runtime.cache": {},
        "fusion_mlx.eval.mbpp": {},
        "fusion_mlx.eval.datasets": {},
        "fusion_mlx.engine.embedding": {
            "EmbeddingEngine": type("EmbeddingEngine", (), {})
        },
        "fusion_mlx.agents.testing": {},
        "fusion_mlx.agents.base": {},
        "fusion_mlx.admin.oq_manager": {},
        "fusion_mlx.admin.ms_downloader": {},
        "fusion_mlx.admin.hf_uploader": {},
        "fusion_mlx.adapter.output_parser": {
            "OutputParserSession": type("OutputParserSession", (), {})
        },
        "fusion_mlx.utils.generation_config": {},
        "scripts": {},
    }

    for mod_name, attrs in stubs.items():
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _make_stub_module(mod_name, attrs)


_install_stubs()
