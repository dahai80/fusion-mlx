# SPDX-License-Identifier: Apache-2.0
import logging
import sys
import types

logger = logging.getLogger(__name__)


def _make_stub_module(name, attrs=None):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs or {})
    mod.__path__ = []
    return mod


def _install_stubs():
    stubs = {
        "fusion_mlx.domain": {},
        "fusion_mlx.domain.guards": {},
        "fusion_mlx.domain.schemas": {},
        "fusion_mlx.domain.errors": {},
        "fusion_mlx.domain._registry": {},
        "fusion_mlx.text_model_from_vlm": {"text_model_from_vlm": lambda *a, **k: None},
        "fusion_mlx.logging_config": {"setup_logging": lambda *a, **k: None},
        "fusion_mlx.process_title": {"set_process_title": lambda *a, **k: None},
        "fusion_mlx.positioned_kv_cache": {
            "PositionedKVCache": type("PositionedKVCache", (), {})
        },
        "fusion_mlx.mllm_cache": {"MLLMCache": type("MLLMCache", (), {})},
        "fusion_mlx._sampler_fast_path": {},
        "fusion_mlx.api.markitdown": {},
        "fusion_mlx.engine.embedding": {
            "EmbeddingEngine": type("EmbeddingEngine", (), {})
        },
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
