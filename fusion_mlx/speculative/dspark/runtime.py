# SPDX-License-Identifier: Apache-2.0
# DSpark runtime wrapper - loads the vendored DSparkGenerator.
#
# dspark-metal (DeepSeek DeepSpec MLX port) is vendored under
# fusion_mlx.speculative.dspark.engine, so no external pip install is
# needed. DSparkGenerator is self-contained: it loads its own target
# model AND the converted MLX draft, runs the propose->verify->accept
# loop, and exposes generate_from_tokens / stream_from_tokens with
# lossless rejection-sampling output. The import stays local to
# load_runtime so the heavy mlx_lm dependency only loads on demand.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DSparkRuntime:
    # The live DSparkGenerator instance (built by load_runtime).
    generator: Any = None
    # Local path to the converted MLX draft (dspark-metal-convert output).
    draft_path: str = ""
    # Quant bits used for the draft (8 is the validated sweet spot:
    # 8B 1.69x, 14B 1.52x at q8 vs 1.59x/1.44x at q4 — see dspark-metal
    # benchmarks/results-qwen3.md).
    draft_quant_bits: int = 8
    # Target HF repo / local path the generator loads.
    target_repo: str = ""
    # Rolling avg-acceptance samples for /healthz telemetry.
    _accept_lens: list = field(default_factory=list)

    def record_accept(self, avg_accept: float | None) -> None:
        if avg_accept and avg_accept > 0:
            self._accept_lens.append(float(avg_accept))

    def accept_lens_snapshot(self) -> list:
        return list(self._accept_lens)


def load_runtime(
    target_repo: str,
    draft_path: str,
    draft_quant_bits: int = 8,
) -> DSparkRuntime:
    # Local import: the vendored engine pulls mlx_lm at load time; keep
    # it deferred so CLI/help paths stay light.
    from .engine import DSparkGenerator

    logger.info(
        "loading DSparkGenerator target=%s draft=%s draft_quant_bits=%d",
        target_repo,
        draft_path,
        draft_quant_bits,
    )
    gen = DSparkGenerator(
        target_model=target_repo,
        draft_model=draft_path,
        draft_quant_bits=draft_quant_bits,
    )
    logger.info(
        "DSparkGenerator ready target=%s draft_quant=%s",
        getattr(gen.target, "requested_model", target_repo),
        getattr(gen, "draft_quantization", None),
    )
    return DSparkRuntime(
        generator=gen,
        draft_path=draft_path,
        draft_quant_bits=draft_quant_bits,
        target_repo=target_repo,
    )
