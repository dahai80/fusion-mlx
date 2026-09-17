// fusion_mlx/shim/csrc/alignas.h
// Tier-1 #7: CPU-GPU shared structs MUST use alignas(128) to avoid
// False-Sharing Cache Bouncing. v2 doc §5.4.
// This header provides the alignment macro + the ring-buffer slot struct
// used by the Unified-Memory double-buffered queue (PR-E GBNF / PR-J
// EngineRunner). PR-A ships the definitions only; consumers land later.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace fusion_mlx::shim {

// 128-byte alignment: matches two Apple Silicon cache lines (64B each) so
// a CPU writer and a GPU-reader-polling flag never land on the same line.
// alignas on a stack/heap object requires the allocation to honor it; for
// heap allocation use aligned_alloc below.
constexpr std::size_t SHIM_CACHELINE = 128;

// Ring-buffer slot for CPU->GPU token/mask handoff. The CPU side writes
// `ready_seq` (release order) AFTER filling payload; the GPU/reader side
// spins on `ready_seq` (acquire order) before consuming. Three-slot
// rotation avoids RAW write-after-read hazards (v2 doc §5.4 rule 3).
struct alignas(SHIM_CACHELINE) RingSlot {
    // Monotonically increasing sequence number written by producer.
    // Reader accepts a slot only when observed_seq < ready_seq.
    std::atomic<std::uint64_t> ready_seq{0};
    std::uint64_t observed_seq{0};
    // Payload: up to N token ids + their DFA acceptance mask offsets.
    // Fixed-size to keep the slot cache-line friendly; variable-length
    // drafts spill to a side buffer indexed by payload_offset.
    std::int32_t token_ids[32]{};
    std::int32_t payload_offset{0};
    std::int32_t payload_len{0};
    std::uint8_t flags{0}; // bit0: grammar_mask_valid, bit1: last_slot
};

// Aligned allocation helper. macOS posix_memalign honors alignment >=
// sizeof(void*); 128 is well above. Returned pointer must be free()'d.
inline void* aligned_alloc_128(std::size_t bytes) {
    void* p = nullptr;
    if (posix_memalign(&p, SHIM_CACHELINE, bytes) != 0) {
        return nullptr;
    }
    return p;
}

} // namespace fusion_mlx::shim
