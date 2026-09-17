// fusion_mlx/shim/csrc/bindings.cpp
// nanobind shim for the fusion-mlx C++ Shim layer. PR-A exports the
// Tier-1 safety base only: hardware_probe + memory_sentinel + the
// C-ABI envelope last-error accessor. Metal kernel ops land in later PRs
// (PR-G fused RoPE/RMSNorm, PR-I Tree-Mask, ...).
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>

#include "c_abi_envelope.h"
#include "hardware_probe.h"
#include "memory_sentinel.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// Wrap a Python callable into the PressureCallback C++ type. nanobind
// lets us accept nb::object and call it via nb::handle.
fusion_mlx::shim::PressureCallback make_pressure_cb(nb::object py_cb) {
    if (py_cb.is_none() || !py_cb.is_valid()) {
        return nullptr;
    }
    // Hold a reference to the Python callable so it outlives the dispatch
    // queue. The callback is invoked off the main thread; the GIL is
    // acquired before calling into Python.
    return [py_cb](int level, std::string name) {
        nb::gil_scoped_acquire gil;
        try {
            py_cb(level, nb::str(name.c_str()));
        } catch (...) {
            // Swallow — sentinel must never crash the process.
        }
    };
}

nb::dict probe_to_dict(const fusion_mlx::shim::HardwareProbe& h) {
    nb::dict d;
    d["architecture"] = nb::str(h.architecture.c_str());
    d["gen"] = nb::int_(h.gen);
    d["has_bf16_mma"] = nb::bool_(h.has_bf16_mma);
    d["has_fp8_mma"] = nb::bool_(h.has_fp8_mma);
    d["gpu_core_count"] = nb::int_(h.gpu_core_count);
    d["device_name"] = nb::str(h.device_name.c_str());
    return d;
}

} // namespace

NB_MODULE(_ext, m) {
    m.doc() = "fusion-mlx C++ Shim: Tier-1 safety base + custom op host";

    // HardwareProbe: bound as a class so both the structured object and the
    // dict convenience accessor are reachable from Python.
    nb::class_<fusion_mlx::shim::HardwareProbe>(m, "HardwareProbe")
        .def_ro("architecture", &fusion_mlx::shim::HardwareProbe::architecture)
        .def_ro("gen", &fusion_mlx::shim::HardwareProbe::gen)
        .def_ro("has_bf16_mma", &fusion_mlx::shim::HardwareProbe::has_bf16_mma)
        .def_ro("has_fp8_mma", &fusion_mlx::shim::HardwareProbe::has_fp8_mma)
        .def_ro("gpu_core_count", &fusion_mlx::shim::HardwareProbe::gpu_core_count)
        .def_ro("device_name", &fusion_mlx::shim::HardwareProbe::device_name);

    // hardware_probe() -> HardwareProbe. Probes MTLDevice feature set for
    // BF16/FP8 MMA capability. Cached by fast.py on first call.
    m.def(
        "hardware_probe",
        []() {
            return fusion_mlx::shim::envelope_value<fusion_mlx::shim::HardwareProbe>(
                []() { return fusion_mlx::shim::probe_hardware(); },
                fusion_mlx::shim::HardwareProbe{},
                nullptr);
        },
        nb::rv_policy::move);

    m.def(
        "hardware_probe_dict",
        []() {
            auto h = fusion_mlx::shim::probe_hardware();
            return probe_to_dict(h);
        },
        nb::rv_policy::move);

    // Memory pressure sentinel.
    m.def(
        "start_memory_sentinel",
        [](nb::object cb) {
            return fusion_mlx::shim::start_memory_sentinel(make_pressure_cb(cb));
        },
        "callback"_a = nb::none());
    m.def("stop_memory_sentinel", &fusion_mlx::shim::stop_memory_sentinel);
    m.def(
        "last_memory_pressure",
        []() {
            return static_cast<int>(fusion_mlx::shim::last_memory_pressure());
        });

    // C-ABI envelope last-error accessor (value-returning ops record errors
    // here since they cannot thread ShimResult* out).
    m.def("last_error_code", []() {
        return static_cast<int>(fusion_mlx::shim::last_error_code());
    });
    m.def("last_error_message", []() {
        return fusion_mlx::shim::last_error_message();
    });
}
