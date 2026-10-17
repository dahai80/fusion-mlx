from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx


@dataclass
class LoRAWeights:
    # #1217: lora_A/lora_B are optional — ComfyUI turbo extraction format
    # (e.g. Wan2.2-TI2V-5B-Turbo delta LoRA) also ships per-module
    # weight/bias deltas for non-Linear targets (RMSNorm/LayerNorm) that
    # have no low-rank pair at all.
    lora_A: mx.array | None = None  # noqa: N815 - matches checkpoint key convention
    lora_B: mx.array | None = None  # noqa: N815 - matches checkpoint key convention
    rank: int = 0
    alpha: float = 0.0
    module_name: str = ""
    # Direct deltas from the ComfyUI extraction format: ``{module}.diff_b``
    # (Linear/LayerNorm bias delta) and ``{module}.diff`` (norm weight
    # delta). Applied as ``param += strength * delta``.
    weight_diff: mx.array | None = None
    bias_diff: mx.array | None = None

    @property
    def scale(self) -> float:
        if not self.rank:
            return 0.0
        return self.alpha / self.rank


@dataclass
class LoRAConfig:
    path: Path
    strength: float = 1.0
    target_modules: list[str] | None = None

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"LoRA file not found: {self.path}")
        if self.strength < 0:
            raise ValueError(f"LoRA strength must be non-negative, got {self.strength}")


@dataclass
class AppliedLoRA:
    weights: LoRAWeights
    strength: float

    def compute_delta(self) -> mx.array | None:
        # #1217: diff-only modules (norm targets) have no low-rank pair;
        # their weight_diff is consumed by the apply paths directly.
        if self.weights.lora_B is None or self.weights.lora_A is None:
            return None
        scale = self.weights.scale
        delta = self.weights.lora_B @ self.weights.lora_A
        return scale * self.strength * delta
