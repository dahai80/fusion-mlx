// fusion_mlx/shim/csrc/memory_sentinel.cpp
// macOS dispatch_source MEMORYPRESSURE sentinel. Guards on
// __APPLE__ so the rest of the shim builds on non-Apple hosts (headless
// CI) — there start_memory_sentinel returns false and the Python
// ProcessMemoryEnforcer polling path stays authoritative.
#include "memory_sentinel.h"

#include <atomic>
#include <mutex>

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

namespace fusion_mlx::shim {

namespace {

std::mutex g_mtx;
#if defined(__APPLE__)
dispatch_source_t g_source = nullptr;
dispatch_queue_t g_queue = nullptr;
#endif
PressureCallback g_callback;
std::atomic<int> g_last_level{static_cast<int>(MemoryPressure::Normal)};

const char* level_name(MemoryPressure p) {
    switch (p) {
        case MemoryPressure::Normal: return "normal";
        case MemoryPressure::Warning: return "warning";
        case MemoryPressure::Critical: return "critical";
    }
    return "unknown";
}

#if defined(__APPLE__)
void dispatch_callback(int level) {
    g_last_level.store(level, std::memory_order_release);
    PressureCallback cb;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        cb = g_callback;
    }
    if (cb) {
        try {
            cb(level, std::string(level_name(static_cast<MemoryPressure>(level))));
        } catch (...) {
            // A callback exception must not escape into the dispatch queue.
        }
    }
}
#endif

} // namespace

bool start_memory_sentinel(PressureCallback callback) {
#if defined(__APPLE__)
    std::lock_guard<std::mutex> lk(g_mtx);
    if (g_source) {
        // Already running — replace the callback.
        g_callback = std::move(callback);
        return true;
    }
    g_queue = dispatch_queue_create("fusion_mlx.shim.memory_sentinel",
                                    DISPATCH_QUEUE_SERIAL);
    if (!g_queue) {
        return false;
    }
    g_source = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
        DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL,
        g_queue);
    if (!g_source) {
        return false;
    }
    g_callback = std::move(callback);
    dispatch_source_set_event_handler(g_source, ^{
        unsigned long flags = dispatch_source_get_data(g_source);
        int level = static_cast<int>(MemoryPressure::Normal);
        if (flags & DISPATCH_MEMORYPRESSURE_CRITICAL) {
            level = static_cast<int>(MemoryPressure::Critical);
        } else if (flags & DISPATCH_MEMORYPRESSURE_WARN) {
            level = static_cast<int>(MemoryPressure::Warning);
        }
        dispatch_callback(level);
    });
    dispatch_source_set_cancel_handler(g_source, ^{
        if (g_queue) {
            dispatch_release(g_queue);
            g_queue = nullptr;
        }
    });
    dispatch_resume(g_source);
    return true;
#else
    // Non-Apple host: sentinel unavailable. Python polling enforcer stays
    // authoritative. This is not an error condition.
    (void)callback;
    return false;
#endif
}

void stop_memory_sentinel() {
#if defined(__APPLE__)
    std::lock_guard<std::mutex> lk(g_mtx);
    if (g_source) {
        dispatch_source_t src = g_source;
        g_source = nullptr;
        // Cancel + release the +1 retain from dispatch_source_create. The
        // cancel handler (set in start) releases the queue asynchronously.
        // The source stays alive until the cancel handler completes.
        dispatch_source_cancel(src);
        dispatch_release(src);
    }
    g_callback = nullptr;
    g_last_level.store(static_cast<int>(MemoryPressure::Normal),
                       std::memory_order_release);
#else
    // no-op
#endif
}

MemoryPressure last_memory_pressure() {
    return static_cast<MemoryPressure>(
        g_last_level.load(std::memory_order_acquire));
}

} // namespace fusion_mlx::shim
