"""Hardware type definitions adapted from whichllm."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GPUInfo:
    name: str
    vendor: str  # "nvidia" | "amd" | "apple" | "intel"
    vram_bytes: int
    usable_vram_bytes: int | None = None
    compute_capability: tuple[int, int] | None = None  # NVIDIA only
    memory_bandwidth_gbps: float | None = None
    shared_memory: bool = False


@dataclass
class HardwareInfo:
    gpus: list[GPUInfo] = field(default_factory=list)
    cpu_name: str = "Unknown"
    cpu_cores: int = 0
    ram_bytes: int = 0
    disk_free_bytes: int = 0
    os: str = "darwin"
