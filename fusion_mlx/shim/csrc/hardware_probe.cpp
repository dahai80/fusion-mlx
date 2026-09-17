// fusion_mlx/shim/csrc/hardware_probe.cpp
// Runtime MTLDevice feature-set probe. Derives BF16/FP8 MMA capability
// from the GPU family rather than the marketing name where possible.
#include "hardware_probe.h"

#include <cstdlib>
#include <string>

// MLX metal device access. metal::device(default_device()) returns the
// per-device Device wrapper holding the MTL::Device.
#include "mlx/backend/metal/device.h"
#include "mlx/device.h"

// Metal-cpp gives us MTL::Device GPU-family queries without an Obj-C
// bridge. MLX ships the Metal-cpp headers via its include path.
#include <Metal/Metal.h>

namespace fusion_mlx::shim {

namespace {

// Parse "M3", "M4 Pro", "M2 Max" -> generation int. Returns 0 if unknown.
int gen_from_name(const std::string& name) {
    // Look for "M" + digit.
    for (std::size_t i = 0; i + 1 < name.size(); ++i) {
        if (name[i] == 'M' && name[i + 1] >= '1' && name[i + 1] <= '9') {
            return name[i + 1] - '0';
        }
    }
    return 0;
}

} // namespace

HardwareProbe probe_hardware() {
    HardwareProbe h{};
    h.gen = 0;
    h.has_bf16_mma = false;
    h.has_fp8_mma = false;
    h.gpu_core_count = 0;

    // Default device name from the environment / sysctl fallback so the
    // probe is still useful when MLX Metal is unavailable (headless CI).
    if (const char* cn = std::getenv("FUSION_SHIM_FORCE_CHIP")) {
        h.device_name = cn;
        h.gen = gen_from_name(cn);
    } else {
        // sysctl machdep.cpu.brand_string is the authoritative chip string
        // on Apple Silicon; reading it via popen keeps this dependency-free.
        FILE* fp = popen("/usr/sbin/sysctl -n machdep.cpu.brand_string 2>/dev/null", "r");
        if (fp) {
            char buf[128] = {0};
            if (std::fgets(buf, sizeof(buf), fp)) {
                h.device_name = std::string(buf);
                while (!h.device_name.empty() &&
                       (h.device_name.back() == '\n' || h.device_name.back() == '\r')) {
                    h.device_name.pop_back();
                }
            }
            pclose(fp);
        }
        h.gen = gen_from_name(h.device_name);
    }

    // Try MLX Metal device for architecture + GPU family. This block is
    // guarded so a headless build (no Metal) still links.
    try {
        auto& d = mlx::core::metal::device(mlx::core::default_device());
        h.architecture = d.get_architecture();
        int mlx_gen = d.get_architecture_gen();
        if (mlx_gen > 0) {
            h.gen = mlx_gen;
        }
        MTL::Device* mtl = d.mtl_device();
        if (mtl) {
            // GPU family probe. M3+ exposes MTLGPUFamilyApple9 (or
            // MTLGPUFamilyMac2 on desktop-class). BF16 simdgroup matrix
            // multiply is available from Apple9 / Mac2 onward.
            bool apple9 = mtl->supportsFamily(MTL::GPUFamilyApple9);
            bool mac2 = mtl->supportsFamily(MTL::GPUFamilyMac2);
            bool apple10 = mtl->supportsFamily(MTL::GPUFamilyApple10);
            bool apple11 = mtl->supportsFamily(MTL::GPUFamilyApple11);
            bool apple12 = mtl->supportsFamily(MTL::GPUFamilyApple12);
            bool apple13 = mtl->supportsFamily(MTL::GPUFamilyApple13);
            if (apple9 || mac2) {
                h.has_bf16_mma = true;
            }
            // FP8 throughput: conservative — only M4+ (Apple10+) until
            // per-part benchmarks confirm non-emulated e4m3/e5m2.
            if (apple10 || apple11 || apple12 || apple13 || mac2) {
                h.has_fp8_mma = true;
            }
            // registry->deviceName returns the marketing name.
            h.gpu_core_count = 0; // core count needs IORegistry, deferred
            if (h.device_name.empty()) {
                h.device_name = h.architecture;
            }
        }
    } catch (...) {
        // Headless / no Metal: keep the sysctl-derived values. The Python
        // fallback layer (fast.py) re-derives capability from
        // utils/hardware.py when _ext is absent or this probe fails.
        if (h.architecture.empty()) {
            h.architecture = h.device_name;
        }
    }

    // Final derivation by generation if the family probe did not fire.
    if (!h.has_bf16_mma && h.gen >= 3) {
        h.has_bf16_mma = true;
    }
    if (!h.has_fp8_mma && h.gen >= 4) {
        h.has_fp8_mma = true;
    }
    if (h.architecture.empty()) {
        h.architecture = h.device_name;
    }

    return h;
}

} // namespace fusion_mlx::shim
