// fusion_mlx/shim/csrc/engine_runner.cpp
// PR-J: C++ EngineRunner implementation.

#include "engine_runner.h"

#include <chrono>

#if defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
#endif

namespace fusion_mlx::shim {

EngineRunner::EngineRunner(DecodeQoS qos) : _qos(qos) {}

EngineRunner::~EngineRunner() { stop(); }

bool EngineRunner::start() {
    std::unique_lock<std::mutex> lk(_mtx);
    if (_running.load()) {
        return true;
    }
    _stop_flag.store(false);
    _thread = std::thread(&EngineRunner::_run, this);
    // Wait until the thread has set _running (so callers see a consistent
    // state after start() returns).
    _cv.wait(lk, [this] { return _running.load() || _stop_flag.load(); });
    return _running.load();
}

void EngineRunner::stop() {
    {
        std::lock_guard<std::mutex> lk(_mtx);
        if (!_running.load() && _thread.get_id() == std::thread::id()) {
            return;
        }
        _stop_flag.store(true);
        _work_ready.store(true);
        _cv.notify_all();
    }
    if (_thread.joinable()) {
        _thread.join();
    }
    _thread_started.store(0);
}

bool EngineRunner::is_running() const {
    return _running.load();
}

ShimResult EngineRunner::submit(std::function<void()> fn) {
    _submitted.fetch_add(1);
    if (!_running.load()) {
        // Thread not running: run inline on the calling thread.
        return envelope([&fn] { fn(); });
    }
    std::unique_lock<std::mutex> lk(_mtx);
    _current = std::move(fn);
    _work_ready.store(true);
    _cv.notify_one();
    // Wait for the worker to finish this unit. The result comes from
    // _last_result (worker TLS is invisible here).
    _done_cv.wait(lk, [this] { return !_work_ready.load(); });
    auto res = _last_result;
    lk.unlock();
    if (res.code == ShimError::Ok) {
        _completed.fetch_add(1);
    } else {
        _failed.fetch_add(1);
    }
    return res;
}

EngineStats EngineRunner::stats() const {
    EngineStats s;
    s.submitted = _submitted.load();
    s.completed = _completed.load();
    s.failed = _failed.load();
    s.thread_started = _thread_started.load();
    s.thread_stopped = _thread_stopped.load();
    s.qos_class = _actual_qos.load();
    return s;
}

void EngineRunner::_run() {
#if defined(__APPLE__)
    // Pin to P-cores via QoS. QOS_CLASS_USER_INTERACTIVE = highest priority,
    // scheduler keeps it on P-cores for low-latency decode.
    qos_class_t target;
    switch (_qos) {
        case DecodeQoS::UserInteractive:
            target = QOS_CLASS_USER_INTERACTIVE;
            break;
        case DecodeQoS::UserInitiated:
            target = QOS_CLASS_USER_INITIATED;
            break;
        case DecodeQoS::Utility:
            target = QOS_CLASS_UTILITY;
            break;
        default:
            target = QOS_CLASS_USER_INTERACTIVE;
            break;
    }
    qos_class_t actual = target;
    pthread_set_qos_class_self_np(target, 0);
    // Read back the actual class (may be downgraded by the system).
    pthread_get_qos_class_np(pthread_self(), &actual, nullptr);
    _actual_qos.store(static_cast<int>(
        actual == QOS_CLASS_USER_INTERACTIVE ? 0
        : actual == QOS_CLASS_USER_INITIATED ? 1
        : 2));
#else
    _actual_qos.store(static_cast<int>(_qos));
#endif

    _thread_started.fetch_add(1);
    {
        std::lock_guard<std::mutex> lk(_mtx);
        _running.store(true);
        _cv.notify_all();  // wake start() waiter
    }

    while (!_stop_flag.load()) {
        std::unique_lock<std::mutex> lk(_mtx);
        _cv.wait(lk, [this] { return _work_ready.load() || _stop_flag.load(); });
        if (_stop_flag.load()) {
            break;
        }
        auto unit = std::move(_current);
        _current = nullptr;
        // Wrap in the C-ABI envelope: any exception is caught and recorded
        // via set_last_error, never propagated across the CPython boundary.
        auto res = envelope([&unit] {
            if (unit) {
                unit();
            }
        });
        if (res.code != ShimError::Ok) {
            set_last_error(res.code, res.message);
        }
        _last_result = res;
        _work_ready.store(false);
        _done_cv.notify_all();
    }

    _running.store(false);
    _thread_stopped.fetch_add(1);
}

} // namespace fusion_mlx::shim
