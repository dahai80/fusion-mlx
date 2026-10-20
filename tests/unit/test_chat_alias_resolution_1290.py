# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from fusion_mlx import model_aliases
from fusion_mlx.server import resolve_model_id

logger = logging.getLogger(__name__)


@pytest.fixture
def alias_registry(monkeypatch):
    aliases = {
        "qwen3.5-9b-4bit": {"hf_path": "mlx-community/Qwen3.5-9B-4bit"},
        "plain-4b": "mlx-community/Qwen3.5-4B-MLX-4bit",
    }
    monkeypatch.setattr(model_aliases, "_load_aliases", lambda: aliases)
    return aliases


def test_dynamic_alias_dict_entry_resolves(alias_registry):
    logger.info("checking dict-form alias entry")
    assert resolve_model_id("qwen3.5-9b-4bit") == "mlx-community/Qwen3.5-9B-4bit"


def test_dynamic_alias_str_entry_resolves(alias_registry):
    logger.info("checking str-form alias entry")
    assert resolve_model_id("plain-4b") == "mlx-community/Qwen3.5-4B-MLX-4bit"


def test_exact_hf_path_passthrough(alias_registry):
    logger.info("checking exact HF id is not rewritten")
    assert (
        resolve_model_id("mlx-community/Qwen3.5-9B-4bit")
        == "mlx-community/Qwen3.5-9B-4bit"
    )


def test_unknown_id_passthrough(alias_registry):
    logger.info("checking unknown id is not rewritten")
    assert resolve_model_id("no-such-model") == "no-such-model"


def test_static_alias_still_wins(alias_registry):
    from fusion_mlx.config import DEFAULT_ALIASES

    name, target = next(iter(DEFAULT_ALIASES.items()))
    logger.info("checking static alias %s wins", name)
    assert resolve_model_id(name) == target
