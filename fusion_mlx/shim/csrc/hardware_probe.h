// fusion_mlx/shim/csrc/hardware_probe.h
// Tier-1 #4: runtime hardware feature probe. M1/M2 lack hardware BF16/FP8
// MMA; M3-M5 have it. Condition-compile branch prevents perf cliff on old
// hardware (v2 doc §5.1 rule 4, §1 principle 8).
#pragma once

#include <cstdint>
#include <string>

namespace fusion_mlx::shim {

struct HardwareProbe {
    // Raw architecture string from MLX metal::Device::get_architecture().
    std::string architecture;
    // Generation number from get_architecture_gen(). M1=1? actual MLX
    // values are probed at runtime; we derive gen from the arch string as
    // a fallback.
    int gen;
    // Derived: true on M3 and later (hardware BF16 simdgroup matrix MMA).
    bool has_bf16_mma;
    // Derived: true where FP8 (e4m3/e5m2) throughput is non-emulated.
    // Conservative: only flag true on M4+ until per-part benchmarks land.
    bool has_fp8_mma;
    // GPU core count from MTLDevice if available, else 0.
    int gpu_core_count;
    // Device name string (e.g. "Apple M4 Pro").
    std::string device_name;
};

// Probe the active Metal device. Called once at shim init and cached by
// the Python fast.py layer. Safe to call repeatedly (idempotent read).
// Returns a stack HardwareProbe; nanobind wraps it into a Python dict in
// bindings.cpp.
HardwareProbe probe_hardware();

} // namespace fusion_mlx::shim
