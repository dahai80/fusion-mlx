// fusion_mlx/shim/csrc/c_abi_envelope.h
// Tier-1 #9: C-ABI exception envelope. Metal/C++ async errors MUST NOT
// propagate as raw C++ exceptions across the CPython boundary — that
// segfaults. This envelope catches and converts to an error code + message
// the Python side reads via nanobind (v2 doc §5.4 rule 5, §3 附录一 #9).
#pragma once

#include <functional>
#include <string>
#include <utility>

namespace fusion_mlx::shim {

// Error codes are stable across the C-ABI boundary (never reorder).
enum class ShimError : int {
    Ok = 0,
    MetalKernelFailure = 1,
    DeviceUnavailable = 2,
    OutOfMemory = 3,
    ShapeMismatch = 4,
    InvalidArgument = 5,
    Stopped = 6,
    Unknown = 99,
};

struct ShimResult {
    ShimError code;
    std::string message; // human-readable, logged on the Python side
};

// Run `fn` inside a try/catch that converts every exception to a
// ShimResult. Used to wrap every native op entry point so no C++ exception
// ever crosses into CPython. Returns {Ok, ""} on success.
template <typename Fn>
ShimResult envelope(Fn&& fn) {
    try {
        fn();
        return {ShimError::Ok, ""};
    } catch (const std::bad_alloc&) {
        return {ShimError::OutOfMemory, "shim: bad_alloc"};
    } catch (const std::invalid_argument& e) {
        return {ShimError::InvalidArgument, std::string("shim: ") + e.what()};
    } catch (const std::runtime_error& e) {
        return {ShimError::MetalKernelFailure, std::string("shim: ") + e.what()};
    } catch (const std::exception& e) {
        return {ShimError::Unknown, std::string("shim: ") + e.what()};
    } catch (...) {
        return {ShimError::Unknown, "shim: unknown exception"};
    }
}

// Variant that returns a value. `fn` must return T; on exception `fallback`
// is returned and the error is captured in `out_err`.
template <typename T, typename Fn>
T envelope_value(Fn&& fn, T fallback, ShimResult* out_err) {
    try {
        return fn();
    } catch (const std::bad_alloc&) {
        if (out_err) *out_err = {ShimError::OutOfMemory, "shim: bad_alloc"};
        return fallback;
    } catch (const std::exception& e) {
        if (out_err) *out_err = {ShimError::Unknown, std::string("shim: ") + e.what()};
        return fallback;
    } catch (...) {
        if (out_err) *out_err = {ShimError::Unknown, "shim: unknown exception"};
        return fallback;
    }
}

// TLS last-error accessors (definitions in c_abi_envelope.cpp). Used by
// value-returning ops that caught an exception and returned fallback —
// the Python binding reads these to surface the message.
void set_last_error(ShimError code, std::string msg);
ShimError last_error_code();
std::string last_error_message();

} // namespace fusion_mlx::shim
