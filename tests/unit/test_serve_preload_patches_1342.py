# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace

import pytest

from fusion_mlx.engines.batched import BatchedEngine
from fusion_mlx.engines.vlm import VLMBatchedEngine


class _StopBeforeLoad(Exception):
    pass


class _DispatchRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, model_name, model_settings=None, for_vlm=False):
        self.calls.append(
            {
                "model_name": model_name,
                "model_settings": model_settings,
                "for_vlm": for_vlm,
            }
        )
        raise _StopBeforeLoad


@pytest.fixture
def dispatch(monkeypatch):
    recorder = _DispatchRecorder()
    from fusion_mlx.utils import model_loading

    monkeypatch.setattr(
        model_loading, "maybe_apply_pre_load_patches", recorder, raising=True
    )
    return recorder


def _settings(mtp_enabled=False):
    return SimpleNamespace(mtp_enabled=mtp_enabled)


async def test_serve_llm_dispatches_pre_load_patches_once(tmp_path, dispatch):
    settings = _settings(mtp_enabled=True)
    engine = BatchedEngine(model_name=str(tmp_path), model_settings=settings)

    with pytest.raises(_StopBeforeLoad):
        await engine.start()

    assert len(dispatch.calls) == 1
    call = dispatch.calls[0]
    assert call["model_name"] == str(tmp_path)
    assert call["model_settings"] is settings
    assert call["for_vlm"] is False


async def test_serve_vlm_dispatches_with_for_vlm_true(tmp_path, dispatch):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_moe", "vision_config": {}})
    )
    settings = _settings()
    engine = VLMBatchedEngine(model_name=str(tmp_path), model_settings=settings)

    with pytest.raises(_StopBeforeLoad):
        await engine.start()

    assert len(dispatch.calls) == 1
    call = dispatch.calls[0]
    assert call["model_name"] == str(tmp_path)
    assert call["model_settings"] is settings
    assert call["for_vlm"] is True


async def test_text_only_serve_never_passes_for_vlm_true(tmp_path, dispatch):
    # Pool force_text / --text-only loads a VLM-family checkpoint through
    # BatchedEngine (mlx_lm), so VLM-only patches must stay off.
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_moe", "vision_config": {}})
    )
    engine = BatchedEngine(model_name=str(tmp_path), model_settings=_settings())

    with pytest.raises(_StopBeforeLoad):
        await engine.start()

    assert len(dispatch.calls) == 1
    assert dispatch.calls[0]["for_vlm"] is False


def test_oq_text_measure_forwards_model_settings(tmp_path, dispatch):
    from fusion_mlx.oq.streaming import _measure_sensitivity

    settings = _settings(mtp_enabled=True)
    config = {"model_type": "qwen3_5"}

    with pytest.raises(_StopBeforeLoad):
        _measure_sensitivity(
            str(tmp_path),
            config,
            4,
            model_settings=settings,
        )

    assert len(dispatch.calls) == 1
    call = dispatch.calls[0]
    assert call["model_name"] == str(tmp_path)
    assert call["model_settings"] is settings
    assert call["for_vlm"] is False


def test_oq_quantized_measure_forwards_model_settings_and_vlm_flag(tmp_path, dispatch):
    from fusion_mlx.oq.streaming import _measure_sensitivity_from_quantized_model

    settings = _settings()
    config = {"model_type": "qwen3_5_moe", "vision_config": {}}

    with pytest.raises(_StopBeforeLoad):
        _measure_sensitivity_from_quantized_model(
            str(tmp_path),
            config,
            4,
            model_settings=settings,
        )

    assert len(dispatch.calls) == 1
    call = dispatch.calls[0]
    assert call["model_name"] == str(tmp_path)
    assert call["model_settings"] is settings
    assert call["for_vlm"] is True


def test_oq_streaming_forwards_model_settings_to_measurement(tmp_path, monkeypatch):
    safetensors = pytest.importorskip("safetensors.numpy")
    numpy = pytest.importorskip("numpy")
    from fusion_mlx import oq as oq_pkg
    from fusion_mlx.oq.streaming import quantize_oq_streaming

    settings = _settings(mtp_enabled=True)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    safetensors.save_file(
        {"model.embed_tokens.weight": numpy.zeros((2, 2), dtype=numpy.float16)},
        str(tmp_path / "model.safetensors"),
    )
    captured = {}

    def _fake_measure(model_path, config, oq_level, **kwargs):
        captured.update(kwargs)
        raise _StopBeforeLoad

    monkeypatch.setattr(oq_pkg, "_measure_sensitivity", _fake_measure)

    with pytest.raises(_StopBeforeLoad):
        quantize_oq_streaming(
            str(tmp_path),
            str(tmp_path / "out"),
            oq_level=4,
            model_settings=settings,
        )

    assert captured["model_settings"] is settings
