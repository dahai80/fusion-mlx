// fusion_mlx/shim/csrc/engine_runner.h
// PR-J: C++ EngineRunner (Tier-2 prototype, v2 doc §5.4).
//
// A dedicated decode thread pinned to P-cores via pthread QoS. The Python
// generate loop submits decode-step callables; the runner executes them on
// the exclusive thread (GIL acquired), so the main thread is free to
// stream tokens + handle I/O without contending with decode work.
//
// Every submitted callable is wrapped in the C-ABI exception envelope
// (c_abi_envelope.h): a C++/Metal exception inside the decode thread is
// caught and recorded as a ShimResult — NEVER propagated as a raw C++
// exception across the CPython boundary (that segfaults, §5.4 rule 5).
//
// Default OFF (FUSION_ENGINE_RUNNER=1). When OFF or native-unavailable,
// the Python EngineRunner wrapper runs callables inline on the calling
// thread — zero behavior change.
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "c_abi_envelope.h"

namespace fusion_mlx::shim {

// QoS class for the decode thread. QOS_CLASS_USER_INTERACTIVE pins to
// P-cores (highest priority, low latency). Exposed so tests can read it.
enum class DecodeQoS : int {
    UserInteractive = 0,  // P-core, interactive (default)
    UserInitiated = 1,    // P-core, non-interactive
    Utility = 2,          // E-core OK
};

struct EngineStats {
    int submitted = 0;
    int completed = 0;
    int failed = 0;
    int thread_started = 0;
    int thread_stopped = 0;
    int qos_class = static_cast<int>(DecodeQoS::UserInteractive);
};

class EngineRunner {
public:
    explicit EngineRunner(DecodeQoS qos = DecodeQoS::UserInteractive);
    // int overload for nanobind binding (nb::init<int>).
    explicit EngineRunner(int qos_int)
        : EngineRunner(static_cast<DecodeQoS>(
              qos_int < 0 || qos_int > 2 ? 0 : qos_int)) {}
    ~EngineRunner();

    EngineRunner(const EngineRunner&) = delete;
    EngineRunner& operator=(const EngineRunner&) = delete;

    // Start the dedicated decode thread. Idempotent. Returns true if the
    // thread is now running (newly started or already running).
    bool start();

    // Stop the thread. Idempotent. Any unit pending or picked up but not
    // finished is DROPPED (not drained): a submit() blocked on it is
    // released with a {Stopped, ...} result instead of hanging forever.
    void stop();

    bool is_running() const;

    // Submit a Python callable (wrapped as std::function) for execution on
    // the decode thread. Blocks until the callable completes. Returns the
    // ShimResult from the C-ABI envelope (Ok on success). The callable's
    // return value is delivered back to the caller via the GIL-protected
    // Python frame that called submit — nanobind handles the return.
    //
    // If the thread is not running, runs inline on the calling thread.
    ShimResult submit(std::function<void()> fn);

    EngineStats stats() const;

private:
    void _run();

    DecodeQoS _qos;
    std::thread _thread;
    mutable std::mutex _mtx;
    std::condition_variable _cv;
    std::condition_variable _done_cv;
    std::queue<std::function<void()>> _queue;
    std::atomic<bool> _running{false};
    std::atomic<bool> _stop_flag{false};
    std::atomic<bool> _work_ready{false};
    std::function<void()> _current;
    // Per-submit result slot. submit() points this at its own stack
    // ShimResult before staging; the worker writes the unit's envelope
    // result (or the drop-path Stopped) through it under _mtx. A shared
    // _last_result was racy across submitters: a late reader could observe
    // a NEWER unit's result (e.g. another submit's drop-path Stopped)
    // instead of its own.
    ShimResult* _pending_result{nullptr};

    // Stats (atomic for cross-thread reads).
    std::atomic<int> _submitted{0};
    std::atomic<int> _completed{0};
    std::atomic<int> _failed{0};
    std::atomic<int> _thread_started{0};
    std::atomic<int> _thread_stopped{0};
    std::atomic<int> _actual_qos{0};
};

} // namespace fusion_mlx::shim
