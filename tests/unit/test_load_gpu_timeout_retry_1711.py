import logging

import pytest

from fusion_mlx.utils.model_loading import (
    is_metal_command_buffer_timeout,
    load_with_gpu_timeout_retry,
)


def _flaky_load(ok, calls):
    def _load(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1 and not ok:
            raise RuntimeError(
                "RuntimeError: [METAL] Command buffer execution failed: "
                "Caused GPU Timeout Error "
                "(00000002:kIOGPUCommandBufferCallbackErrorTimeout)."
            )
        return "model", "tokenizer"

    return _load


def test_retry_on_gpu_timeout():
    calls = []
    model, tok = load_with_gpu_timeout_retry(_flaky_load(False, calls))
    assert model == "model"
    assert len(calls) == 2


def test_no_retry_when_first_load_ok():
    calls = []
    load_with_gpu_timeout_retry(_flaky_load(True, calls))
    assert len(calls) == 1


def test_other_runtime_error_propagates():
    def _load():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        load_with_gpu_timeout_retry(_load)


def test_predicate():
    assert is_metal_command_buffer_timeout(
        RuntimeError(
            "[METAL] Command buffer execution failed: Caused GPU Timeout "
            "Error (00000002:kIOGPUCommandBufferCallbackErrorTimeout)."
        )
    )
    assert is_metal_command_buffer_timeout(
        RuntimeError("Command buffer execution failed")
    )
    assert not is_metal_command_buffer_timeout(RuntimeError("boom"))


def test_retry_logs_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="fusion_mlx.utils.model_loading"):
        load_with_gpu_timeout_retry(_flaky_load(False, []))
    assert "#1711" in caplog.text
    assert "retrying once" in caplog.text
