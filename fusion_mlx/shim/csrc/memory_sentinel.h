// fusion_mlx/shim/csrc/memory_sentinel.h
// Tier-1 #8: macOS memory-pressure SIGKILL prevention. macOS jetsam
// SIGKILLs the process on memory pressure WITHOUT raising bad_alloc, so
// a polling enforcer is too slow. This C++ sentinel installs a
// dispatch_source on DISPATCH_SOURCE_TYPE_MEMORYPRESSURE and invokes a
// Python callback at warning/critical levels for preemptive cache release
// + batch downsize (v2 doc §5.5).
//
// This bridges to (does NOT replace) the existing Python
// ProcessMemoryEnforcer. The C++ sentinel fires the immediate reactive
// signal; ProcessMemoryEnforcer owns the periodic reclaim policy.
#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace fusion_mlx::shim {

// Memory pressure levels mirroring dispatch_source_memorypressure_flags_t.
enum class MemoryPressure : int {
    Normal = 0,
    Warning = 1,
    Critical = 2,
};

// Start the sentinel. callback is invoked (on a dispatch queue, NOT the
// main thread) whenever pressure transitions. Pass nullptr to use the
// default callback that only logs. Returns true if the dispatch_source
// was installed successfully; false if unavailable (non-macOS / already
// running).
//
// callback signature: (level:int, level_name:str) -> void
using PressureCallback = std::function<void(int, std::string)>;

bool start_memory_sentinel(PressureCallback callback);

// Stop the sentinel and release the dispatch source. Idempotent.
void stop_memory_sentinel();

// Last observed pressure level (thread-safe read).
MemoryPressure last_memory_pressure();

} // namespace fusion_mlx::shim
