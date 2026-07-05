# SPDX-License-Identifier: Apache-2.0
"""Regression for F-061 / F-062 — corrupt-image handling in the MLLM batch
generator."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from fusion_mlx.mllm_batch_generator import MLLMBatchGenerator, MLLMBatchRequest


class _StubModel:

    def __init__(self):
        self.language_model = object()

        class _Cfg:
            image_token_index = None

        self.config = _Cfg()


def _make_generator() -> MLLMBatchGenerator:
    return MLLMBatchGenerator(
        model=_StubModel(),
        processor=object(),
        mm_processor=None,
        enable_vision_cache=False,
    )


def _make_request(images: list[str] | None) -> MLLMBatchRequest:
    return MLLMBatchRequest(
        uid=0,
        request_id="r0",
        prompt="describe",
        images=images or [],
        max_tokens=8,
    )


def _bypass_process_image(monkeypatch):
    from fusion_mlx.models import mllm as mllm_models

    def _identity(img):
        return img

    monkeypatch.setattr(mllm_models, "process_image_input", _identity)


def _install_prepare_inputs_stub(monkeypatch, raiser):
    from fusion_mlx import mllm_batch_generator as gen_mod

    # Ensure mlx_vlm.utils is in sys.modules (conftest mocks mlx_vlm but
    # not the submodule). The source does ``from mlx_vlm.utils import
    # prepare_inputs`` which fails if the submodule isn't registered.
    if "mlx_vlm.utils" not in sys.modules:
        sys.modules["mlx_vlm.utils"] = MagicMock()

    monkeypatch.setattr(sys.modules["mlx_vlm.utils"], "prepare_inputs", raiser)
    if hasattr(gen_mod, "prepare_inputs"):
        monkeypatch.setattr(gen_mod, "prepare_inputs", raiser)


def test_preprocess_wraps_pil_oserror_as_failed_to_process_image(monkeypatch):
    _bypass_process_image(monkeypatch)

    def _raise_oserror(*args, **kwargs):
        raise OSError("broken data stream when reading image file")

    _install_prepare_inputs_stub(monkeypatch, _raise_oserror)

    gen = _make_generator()
    req = _make_request(images=["data:image/png;base64,AAAA"])

    with pytest.raises(ValueError) as exc_info:
        gen._preprocess_request(req)
    assert str(exc_info.value).startswith("Failed to process image")
    assert "broken data stream" in str(exc_info.value)


def test_preprocess_wraps_pil_unidentified_image_as_failed_to_process_image(
    monkeypatch,
):
    _bypass_process_image(monkeypatch)

    from PIL import UnidentifiedImageError

    def _raise_unidentified(*args, **kwargs):
        raise UnidentifiedImageError("cannot identify image file 'X'")

    _install_prepare_inputs_stub(monkeypatch, _raise_unidentified)

    gen = _make_generator()
    req = _make_request(images=["data:image/png;base64,SGVsbG8="])

    with pytest.raises(ValueError) as exc_info:
        gen._preprocess_request(req)
    assert str(exc_info.value).startswith("Failed to process image")
    assert "cannot identify image file" in str(exc_info.value)


def test_preprocess_normalizes_failed_to_load_image_to_failed_to_process_image(
    monkeypatch,
):
    _bypass_process_image(monkeypatch)

    def _raise_failed_to_load(*args, **kwargs):
        raise ValueError(
            "Failed to load image from /tmp/xyz.png: cannot identify image file '/tmp/xyz.png'"
        )

    _install_prepare_inputs_stub(monkeypatch, _raise_failed_to_load)

    gen = _make_generator()
    req = _make_request(images=["data:image/png;base64,AAAA"])

    with pytest.raises(ValueError) as exc_info:
        gen._preprocess_request(req)
    msg = str(exc_info.value)
    assert msg.startswith(
        "Failed to process image"
    ), f"matcher would miss this message: {msg!r}"
    assert "cannot identify image file" in msg


def test_preprocess_preserves_canonical_message_unchanged(monkeypatch):
    _bypass_process_image(monkeypatch)

    def _raise_canonical(*args, **kwargs):
        raise ValueError("Failed to process image: 404 Client Error")

    _install_prepare_inputs_stub(monkeypatch, _raise_canonical)

    gen = _make_generator()
    req = _make_request(images=["data:image/png;base64,AAAA"])

    with pytest.raises(ValueError) as exc_info:
        gen._preprocess_request(req)
    msg = str(exc_info.value)
    assert (
        msg == "Failed to process image: 404 Client Error"
    ), f"canonical message must pass through unchanged, got {msg!r}"


def test_preprocess_propagates_internal_bugs_unchanged(monkeypatch):
    _bypass_process_image(monkeypatch)

    sentinel = AttributeError("'NoneType' object has no attribute 'image_token_index'")

    def _raise_attribute_error(*args, **kwargs):
        raise sentinel

    _install_prepare_inputs_stub(monkeypatch, _raise_attribute_error)

    gen = _make_generator()
    req = _make_request(images=["data:image/png;base64,AAAA"])

    with pytest.raises(AttributeError) as exc_info:
        gen._preprocess_request(req)
    assert exc_info.value is sentinel


def test_preprocess_propagates_typeerror_unchanged(monkeypatch):
    _bypass_process_image(monkeypatch)

    sentinel = TypeError(
        "prepare_inputs() got an unexpected keyword argument 'image_token_index'"
    )

    def _raise_type_error(*args, **kwargs):
        raise sentinel

    _install_prepare_inputs_stub(monkeypatch, _raise_type_error)

    gen = _make_generator()
    req = _make_request(images=["data:image/png;base64,AAAA"])

    with pytest.raises(TypeError) as exc_info:
        gen._preprocess_request(req)
    assert exc_info.value is sentinel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
