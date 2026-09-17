// fusion_mlx/shim/csrc/c_abi_envelope.cpp
// The envelope is header-only (templates). This .cpp exists only to give
// the build system a translation unit for symbol export and to host the
// last-error TLS slot for the C-ABI fallback path (when a caller cannot
// thread a ShimResult* out-param).
#include "c_abi_envelope.h"

#include <mutex>
#include <string>

namespace fusion_mlx::shim {

namespace {
// Thread-local last error so the Python binding can retrieve the message
// after a value-returning op caught an exception and returned fallback.
thread_local std::string g_last_err_msg;
thread_local ShimError g_last_err_code = ShimError::Ok;
} // namespace

void set_last_error(ShimError code, std::string msg) {
    g_last_err_code = code;
    g_last_err_msg = std::move(msg);
}

ShimError last_error_code() {
    return g_last_err_code;
}

std::string last_error_message() {
    return g_last_err_msg;
}

} // namespace fusion_mlx::shim
