# SPDX-License-Identifier: Apache-2.0
# #741 Rule 12: SkyReels ControlNet per-step residual failure must propagate,
# not silently degrade to T2V. Static assertions mirror the file's existing
# test_surface_b_skyreels_encode_control.py pattern (no real-model load).
import inspect

from fusion_mlx.video.skyreels_v3.pipelines import SkyReelsBasePipeline


def _denoise_source():
    src = inspect.getsource(SkyReelsBasePipeline._denoise_sample)
    # isolate the ControlNet residual block
    start = src.index("# ControlNet: 计算每步残差")
    end = src.index("# 模型前向", start)
    return src[start:end]


def test_residual_failure_no_silent_swallow():
    # The bare `except Exception ... cn_residuals = None` that swallowed a
    # per-step residual failure into a silent unconditioned forward MUST
    # be gone (issue #741, Rule 12). The only remaining `cn_residuals = None`
    # is the benign pre-init before the adapter check.
    block = _denoise_source()
    assert "except Exception as exc" not in block
    assert "except Exception" not in block


def test_residual_none_raises_runtime_error():
    # Option A (strict): a None residual MUST raise RuntimeError refusing
    # silent T2V degradation — matching #653 Wan2 C1 (encode_control) ruling.
    block = _denoise_source()
    assert "raise RuntimeError" in block
    assert "#741" in block
    assert "degrade to T2V" in block


def test_residual_block_propagates_adapter_call_failure():
    # modify_denoise_step / get_residuals are now called outside any
    # try/except, so an exception they raise propagates and aborts the run
    # rather than being caught and logged as a warning.
    src = inspect.getsource(SkyReelsBasePipeline._denoise_sample)
    assert "ControlNet: step %d residual failed" not in src
