# SPDX-License-Identifier: Apache-2.0
"""Tests for the R15 Phase 4 TurboQuant K8V4 upgrade.

Adapted from Rapid-MLX. Most K8V4-specific symbols (TurboQuantConfig,
walsh_hadamard_transform, turboquant_k8_encode/decode, etc.) do not exist
in fusion_mlx.turboquant_kv — the module was redesigned with a different
API surface. Tests requiring those symbols are skipped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import mlx.core as mx
import numpy as np
import pytest

try:
    from fusion_mlx.turboquant_kv import TurboQuantKVCache
    _HAS_TQ = True
except ImportError:
    _HAS_TQ = False

pytestmark = pytest.mark.skipif(
    not _HAS_TQ,
    reason="fusion_mlx.turboquant_kv not available",
)


@pytest.mark.skip(reason="rapid-mlx-only: TurboQuantConfig not in fusion_mlx.turboquant_kv")
class TestV4BackwardCompat:
    pass


@pytest.mark.skip(reason="rapid-mlx-only: walsh_hadamard_transform not in fusion_mlx.turboquant_kv")
class TestWalshHadamardRotation:
    pass


@pytest.mark.skip(reason="rapid-mlx-only: turboquant_k8_encode not in fusion_mlx.turboquant_kv")
class TestK8Roundtrip:
    pass


@pytest.mark.skip(reason="rapid-mlx-only: TurboQuantConfig not in fusion_mlx.turboquant_kv")
class TestK8V4Cache:
    pass


@pytest.mark.skip(reason="rapid-mlx-only: TurboQuantConfig not in fusion_mlx.turboquant_kv")
class TestConfigValidation:
    pass


@pytest.mark.skip(reason="rapid-mlx-only: fused_kernel_status not in fusion_mlx.turboquant_kv")
class TestFusedKernel:
    pass


@pytest.mark.skip(reason="rapid-mlx-only: is_incompatible_with_turboquant not in fusion_mlx.turboquant_kv")
class TestSkipList:
    pass


@pytest.mark.skip(reason="rapid-mlx-only: CLI flags differ in fusion-mlx")
class TestCLIFlag:
    pass


@pytest.mark.skip(reason="rapid-mlx-only: routes.metrics not in fusion-mlx")
class TestMetrics:
    pass


@pytest.mark.skip(reason="rapid-mlx-only: resolve_turboquant_mode_default not in fusion_mlx.turboquant_kv")
class TestResolveTurboquantModeDefault:
    pass


@pytest.mark.skip(reason="rapid-mlx-only: TurboQuantConfig not in fusion_mlx.turboquant_kv")
def test_codec_preserves_input_dtype():
    pass


_ = (subprocess, sys)
