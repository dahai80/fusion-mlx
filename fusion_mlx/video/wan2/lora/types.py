from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx


@dataclass
class LoRAWeights:

    lora_A: mx.array  # noqa: N815 - matches checkpoint key convention
    lora_B: mx.array  # noqa: N815 - matches checkpoint key convention
    rank: int
    alpha: float
    module_name: str

    @property
    def scale(self) -> float:
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

    def compute_delta(self) -> mx.array:
        scale = self.weights.scale
        delta = self.weights.lora_B @ self.weights.lora_A
        return scale * self.strength * delta
