// MoE FFN Fused Metal Kernel
// Fuses: gate_up projection (W4A8) -> SiLU activation -> down projection (W4A8)
//
// Architecture:
//   Phase 1: x[M,K] @ w_gate_up[2*inter_dim, K] -> gate[M, inter_dim] + up[M, inter_dim]
//            Apply SiLU(gate) * up, store intermediate in threadgroup
//   Phase 2: hidden[M, inter_dim] @ w_down[K_out, inter_dim] -> output[M, K_out]
//
// Key optimization:
//   - Intermediate activation stays in threadgroup (no device memory round-trip)
//   - Two-level scaling from W4A8 applied per-phase
//   - Expert routing via expert_idx parameter
//
// Threadgroup memory reuse:
//   Phase 1 and Phase 2 weight buffers share the same threadgroup space
//   (they are never needed simultaneously).

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

// ---------------------------------------------------------------------------
// FP8 / FP4 type definitions
// ---------------------------------------------------------------------------

struct fp8_e4m3 {
    uint8_t bits;
    explicit fp8_e4m3(uint8_t b) : bits(b) {}
    fp8_e4m3(float x) {
        float clamped = clamp(x, -448.0f, 448.0f);
        uint32_t u = as_type<uint32_t>(clamped);
        uint32_t sign = (u >> 31) & 1;
        int32_t exp_fp32 = int32_t((u >> 23) & 0xFF);
        uint32_t man_fp32 = u & 0x7FFFFF;
        int32_t exp_fp8 = exp_fp32 - 127 + 7;
        uint32_t man_fp8 = man_fp32 >> 20;
        if (exp_fp8 < 0) { exp_fp8 = 0; man_fp8 = 0; }
        if (exp_fp8 > 15) { exp_fp8 = 15; man_fp8 = 7; }
        bits = uint8_t((sign << 7) | (uint32_t(exp_fp8 & 0xF) << 3) | (man_fp8 & 0x7));
    }
    operator float() const {
        uint32_t sign = (bits >> 7) & 1;
        uint32_t exp8 = (bits >> 3) & 0xF;
        uint32_t man8 = bits & 0x7;
        int32_t exp32 = int32_t(exp8) - 7 + 127;
        uint32_t u = (sign << 31) | (uint32_t(exp32) << 23) | (man8 << 20);
        return as_type<float>(u);
    }
};

struct fp8_e8m0 {
    uint8_t bits;
    explicit fp8_e8m0(uint8_t b) : bits(b) {}
    fp8_e8m0(float x) {
        float log2x = log2(max(x, 1e-30f));
        int32_t rounded = int32_t(round(log2x));
        bits = uint8_t(clamp(rounded + 127, 0, 254));
    }
    operator float() const {
        int32_t exp_val = int32_t(bits) - 127;
        return exp2(float(exp_val));
    }
};

static inline float dequant_fp4(uint8_t nibble) {
    float sign = (nibble & 0x8) ? -1.0f : 1.0f;
    uint32_t exp4 = (nibble >> 1) & 0x3;
    uint32_t man4 = nibble & 0x1;
    float val = sign * float(1 + man4 * 0.5) * exp2(float(int32_t(exp4) - 1));
    return val;
}

static inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constant constexpr const int SIMD_SIZE = 32;

// ---------------------------------------------------------------------------
// FP8 activation loader
// ---------------------------------------------------------------------------

template <short BM, short BK, short BK_PAD, short tgp_size>
struct FP8ActivationLoader {
    const device uint8_t* src_fp8;
    threadgroup uint8_t* dst;
    const int src_ld;
    const short thread_idx;
    const short bi;
    const short bj;

    static constant constexpr const short vec_size = 4;
    static constant constexpr const short TCOLS = BK / vec_size;
    static constant constexpr const short TROWS = tgp_size / TCOLS;

    FP8ActivationLoader(
        const device uint8_t* src_fp8_,
        const int src_ld_,
        threadgroup uint8_t* dst_,
        ushort simd_group_id [[simdgroup_index_in_threadgroup]],
        ushort simd_lane_id [[thread_index_in_simdgroup]])
        : src_fp8(src_fp8_),
          src_ld(src_ld_),
          dst(dst_),
          thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id),
          bi(thread_idx / TCOLS),
          bj(vec_size * (thread_idx % TCOLS)) {
        dst += bi * BK_PAD + bj;
        src_fp8 += bi * src_ld + bj;
    }

    METAL_FUNC void load_unsafe() const {
        struct alignas(4) Vec4 { uint8_t v[4]; };
        for (short i = 0; i < BM; i += TROWS) {
            *((threadgroup Vec4*)(&dst[i * BK_PAD])) =
                *((const device Vec4*)(&src_fp8[i * src_ld]));
        }
    }

    METAL_FUNC void load_safe(short valid_rows, short valid_cols) const {
        for (short i = 0; i < BM; i += TROWS) {
            if (bi + i < valid_rows) {
                for (short j = 0; j < vec_size; j++) {
                    dst[i * BK_PAD + j] = (bj + j < valid_cols)
                        ? src_fp8[i * src_ld + j]
                        : uint8_t(0);
                }
            } else {
                for (short j = 0; j < vec_size; j++) {
                    dst[i * BK_PAD + j] = uint8_t(0);
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// FP4 weight loader
// ---------------------------------------------------------------------------

template <short BN, short BK, short BK_PAD, short tgp_size, short group_size = 32>
struct FP4WeightLoader {
    const device uint8_t* w_packed;
    const device uint8_t* w_scales;
    threadgroup uint8_t* dst_packed;
    threadgroup uint8_t* dst_scales;
    const int w_ld;
    const int s_ld;
    const short thread_idx;

    static constant constexpr const short pack_factor = 2;
    static constant constexpr const short K_PACKED = BK / pack_factor;
    static constant constexpr const short K_PACKED_PAD = BK_PAD / pack_factor;
    static constant constexpr const short S_COLS = BK / group_size;

    FP4WeightLoader(
        const device uint8_t* w_packed_,
        const int w_ld_,
        const device uint8_t* w_scales_,
        const int s_ld_,
        threadgroup uint8_t* dst_packed_,
        threadgroup uint8_t* dst_scales_,
        ushort simd_group_id [[simdgroup_index_in_threadgroup]],
        ushort simd_lane_id [[thread_index_in_simdgroup]])
        : w_packed(w_packed_),
          w_scales(w_scales_),
          dst_packed(dst_packed_),
          dst_scales(dst_scales_),
          w_ld(w_ld_),
          s_ld(s_ld_),
          thread_idx(simd_group_id * SIMD_SIZE + simd_lane_id) {}

    METAL_FUNC void load_unsafe() const {
        const int total_packed = BN * K_PACKED;
        for (int idx = thread_idx; idx < total_packed; idx += tgp_size) {
            int row = idx / K_PACKED;
            int col = idx % K_PACKED;
            dst_packed[row * K_PACKED_PAD + col] = w_packed[row * w_ld + col];
        }
        const int total_scales = BN * S_COLS;
        for (int idx = thread_idx; idx < total_scales; idx += tgp_size) {
            int row = idx / S_COLS;
            int col = idx % S_COLS;
            dst_scales[row * S_COLS + col] = w_scales[row * s_ld + col];
        }
    }

    METAL_FUNC void load_safe(short valid_rows, short valid_cols) const {
        const int total_packed = valid_rows * K_PACKED;
        for (int idx = thread_idx; idx < total_packed; idx += tgp_size) {
            int row = idx / K_PACKED;
            int col = idx % K_PACKED;
            dst_packed[row * K_PACKED_PAD + col] = (col < (valid_cols / pack_factor))
                ? w_packed[row * w_ld + col]
                : uint8_t(0);
        }
        const int total_scales = valid_rows * S_COLS;
        for (int idx = thread_idx; idx < total_scales; idx += tgp_size) {
            int row = idx / S_COLS;
            int col = idx % S_COLS;
            dst_scales[row * S_COLS + col] = (col < (valid_cols / group_size))
                ? w_scales[row * s_ld + col]
                : uint8_t(0);
        }
    }
};

// ---------------------------------------------------------------------------
// MoE FFN Fused Kernel
//
// Fuses: gate_up projection -> SiLU activation -> down projection
// Single expert per kernel launch (expert_idx parameter).
//
// Dimensions:
//   x: [M, K_in]        (fp8_e4m3 activations)
//   w_gate_up: [2*inter_dim, K_in]  (fp4 weights, gate and up stacked)
//   w_down: [K_out, inter_dim]       (fp4 weights)
//   output: [M, K_out]
//
// Threadgroup memory is shared between phases via union:
//   Phase 1: As_fp8 + Bs_gate_up (packed + scales)
//   Intermediate: hidden[BM][inter_dim] in fp32
//   Phase 2: Bs_down (packed + scales), A is hidden buffer
// ---------------------------------------------------------------------------

template <
    typename T,
    short BM, short BN, short BK,
    short WM, short WN,
    short group_size = 32>
[[kernel, max_total_threads_per_threadgroup(WM * WN * SIMD_SIZE)]]
void moe_ffn_fused(
    const device uint8_t* x_fp8             [[buffer(0)]],
    const device float* scales_a            [[buffer(1)]],
    const device uint8_t* w_gate_up_packed  [[buffer(2)]],
    const device uint8_t* w_gate_up_scales  [[buffer(3)]],
    const device float* scales_b_gate_up    [[buffer(4)]],
    const device uint8_t* w_down_packed     [[buffer(5)]],
    const device uint8_t* w_down_scales     [[buffer(6)]],
    const device float* scales_b_down       [[buffer(7)]],
    device float* out                       [[buffer(8)]],
    const constant int& M                   [[buffer(9)]],
    const constant int& K_in                [[buffer(10)]],
    const constant int& inter_dim           [[buffer(11)]],
    const constant int& K_out               [[buffer(12)]],
    const constant int& lda                 [[buffer(13)]],
    const constant int& ldw_gate_up         [[buffer(14)]],
    const constant int& lds_gate_up         [[buffer(15)]],
    const constant int& ldw_down            [[buffer(16)]],
    const constant int& lds_down            [[buffer(17)]],
    const constant int& ldo                 [[buffer(18)]],
    const constant int& expert_idx          [[buffer(19)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {

    constexpr short BK_PAD = BK + 16;
    constexpr short BN_PAD = BN + 4;
    constexpr short tgp_size = WM * WN * SIMD_SIZE;
    constexpr short S_COLS = BK / group_size;

    // Threadgroup memory: union to share between phases
    threadgroup union {
        struct {
            uint8_t As_fp8[BM * BK_PAD];
            uint8_t Bs_packed[BN * (BK_PAD / 2)];
            uint8_t Bs_scales[BN * S_COLS];
        } p1;
        struct {
            float hidden[BM * BN_PAD];
            uint8_t Bs2_packed[BM * (BN_PAD / 2)];
            uint8_t Bs2_scales[BM * (BN / group_size)];
        } p2;
    } tgp;

    const int row_block = int(tid.y) * BM;
    const int col_block_inter = int(tid.x) * BN;
    const short valid_m = short(min(int(BM), M - row_block));

    if (col_block_inter >= inter_dim) return;

    // Warp tiling constants
    constexpr short SM = BM / WM;
    constexpr short SN = BN / WN;
    constexpr short kMmaTilesM = SM / 8;
    constexpr short kMmaTilesN = SN / 8;
    const short warp_row = simd_group_id / WN;
    const short warp_col = simd_group_id % WN;

    float scale_a = (row_block < M) ? scales_a[row_block] : 1.0f;

    // ==================================================================
    // Phase 1: gate_up projection
    // x[M, K_in] @ w_gate[inter_dim, K_in]^T -> gate[M, inter_dim]
    // x[M, K_in] @ w_up[inter_dim, K_in]^T   -> up[M, inter_dim]
    // Then: hidden = SiLU(gate) * up
    // ==================================================================

    simdgroup_matrix<float, 8, 8> frag_C_gate[kMmaTilesM * kMmaTilesN];
    simdgroup_matrix<float, 8, 8> frag_C_up[kMmaTilesM * kMmaTilesN];

    for (int i = 0; i < kMmaTilesM * kMmaTilesN; i++) {
        for (int j = 0; j < 8; j++) {
            frag_C_gate[i].thread_elements()[j] = 0.0f;
            frag_C_up[i].thread_elements()[j] = 0.0f;
        }
    }

    const int K_IT = K_in / BK;

    // Expert-aware weight pointers
    const device uint8_t* x_tile = x_fp8 + size_t(row_block) * lda;
    const device uint8_t* gate_packed = w_gate_up_packed
        + size_t(expert_idx) * 2 * inter_dim * (K_in / 2)
        + size_t(col_block_inter) * (ldw_gate_up / 2);
    const device uint8_t* gate_scales = w_gate_up_scales
        + size_t(expert_idx) * 2 * inter_dim * (K_in / group_size)
        + size_t(col_block_inter) * (lds_gate_up / group_size);
    const device uint8_t* up_packed = w_gate_up_packed
        + size_t(expert_idx) * 2 * inter_dim * (K_in / 2)
        + size_t(inter_dim + col_block_inter) * (ldw_gate_up / 2);
    const device uint8_t* up_scales = w_gate_up_scales
        + size_t(expert_idx) * 2 * inter_dim * (K_in / group_size)
        + size_t(inter_dim + col_block_inter) * (lds_gate_up / group_size);

    float scale_b_gu = scales_b_gate_up[expert_idx * inter_dim + col_block_inter];
    float combined_scale_gu = scale_a * scale_b_gu;

    for (int k_it = 0; k_it < K_IT; k_it++) {
        // Load A (fp8 activation) - shared between gate and up
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FP8ActivationLoader<BM, BK, BK_PAD, tgp_size> loader_a(
            x_tile + k_it * BK, lda,
            tgp.p1.As_fp8,
            simd_group_id, simd_lane_id);
        if (valid_m == BM) {
            loader_a.load_unsafe();
        } else {
            loader_a.load_safe(valid_m, BK);
        }

        // Load gate weights
        FP4WeightLoader<BN, BK, BK_PAD, tgp_size, group_size> loader_gate(
            gate_packed + k_it * BK / 2, ldw_gate_up / 2,
            gate_scales + k_it * BK / group_size, lds_gate_up / group_size,
            tgp.p1.Bs_packed, tgp.p1.Bs_scales,
            simd_group_id, simd_lane_id);
        loader_gate.load_unsafe();

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Gate MMA
        simdgroup_matrix<float, 8, 8> frag_a;
        simdgroup_matrix<float, 8, 8> frag_b;

        for (short k = 0; k < BK; k += 8) {
            for (short m = 0; m < kMmaTilesM; m++) {
                const short a_row_base = warp_row * SM + m * 8;
                for (short j = 0; j < 2; j++) {
                    const short qid = simd_lane_id / 4;
                    const short r = (qid & 4) + ((simd_lane_id / 2) % 4);
                    const short c = k + (qid & 2) * 2 + (simd_lane_id % 2) * 2 + j;
                    const short gr = a_row_base + r;
                    frag_a.thread_elements()[j] = (gr < BM && c < BK)
                        ? float(fp8_e4m3(tgp.p1.As_fp8[gr * BK_PAD + c]))
                        : 0.0f;
                }
                for (short n = 0; n < kMmaTilesN; n++) {
                    const short b_col_base = warp_col * SN + n * 8;
                    for (short j = 0; j < 2; j++) {
                        const short qid = simd_lane_id / 4;
                        const short r = k + (qid & 4) + ((simd_lane_id / 2) % 4);
                        const short c = b_col_base + (qid & 2) * 2 + (simd_lane_id % 2) * 2 + j;
                        if (r < BK && c < BN) {
                            short pc = r / 2;
                            uint8_t packed = tgp.p1.Bs_packed[c * (BK_PAD / 2) + pc];
                            uint8_t nib = (r % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
                            float val = dequant_fp4(nib);
                            float sc = float(fp8_e8m0(tgp.p1.Bs_scales[c * S_COLS + r / group_size]));
                            frag_b.thread_elements()[j] = val * sc;
                        } else {
                            frag_b.thread_elements()[j] = 0.0f;
                        }
                    }
                    simdgroup_multiply_accumulate(
                        frag_C_gate[m * kMmaTilesN + n], frag_a, frag_b,
                        frag_C_gate[m * kMmaTilesN + n]);
                }
            }
        }

        // Load up weights (reuse same Bs_packed/Bs_scales space)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FP4WeightLoader<BN, BK, BK_PAD, tgp_size, group_size> loader_up(
            up_packed + k_it * BK / 2, ldw_gate_up / 2,
            up_scales + k_it * BK / group_size, lds_gate_up / group_size,
            tgp.p1.Bs_packed, tgp.p1.Bs_scales,
            simd_group_id, simd_lane_id);
        loader_up.load_unsafe();

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Up MMA
        for (short k = 0; k < BK; k += 8) {
            for (short m = 0; m < kMmaTilesM; m++) {
                const short a_row_base = warp_row * SM + m * 8;
                for (short j = 0; j < 2; j++) {
                    const short qid = simd_lane_id / 4;
                    const short r = (qid & 4) + ((simd_lane_id / 2) % 4);
                    const short c = k + (qid & 2) * 2 + (simd_lane_id % 2) * 2 + j;
                    const short gr = a_row_base + r;
                    frag_a.thread_elements()[j] = (gr < BM && c < BK)
                        ? float(fp8_e4m3(tgp.p1.As_fp8[gr * BK_PAD + c]))
                        : 0.0f;
                }
                for (short n = 0; n < kMmaTilesN; n++) {
                    const short b_col_base = warp_col * SN + n * 8;
                    for (short j = 0; j < 2; j++) {
                        const short qid = simd_lane_id / 4;
                        const short r = k + (qid & 4) + ((simd_lane_id / 2) % 4);
                        const short c = b_col_base + (qid & 2) * 2 + (simd_lane_id % 2) * 2 + j;
                        if (r < BK && c < BN) {
                            short pc = r / 2;
                            uint8_t packed = tgp.p1.Bs_packed[c * (BK_PAD / 2) + pc];
                            uint8_t nib = (r % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
                            float val = dequant_fp4(nib);
                            float sc = float(fp8_e8m0(tgp.p1.Bs_scales[c * S_COLS + r / group_size]));
                            frag_b.thread_elements()[j] = val * sc;
                        } else {
                            frag_b.thread_elements()[j] = 0.0f;
                        }
                    }
                    simdgroup_multiply_accumulate(
                        frag_C_up[m * kMmaTilesN + n], frag_a, frag_b,
                        frag_C_up[m * kMmaTilesN + n]);
                }
            }
        }
    }

    // Apply SiLU(gate) * up -> hidden, store in threadgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short m = 0; m < kMmaTilesM; m++) {
        for (short n = 0; n < kMmaTilesN; n++) {
            const short row_base = warp_row * SM + m * 8;
            const short col_base = warp_col * SN + n * 8;
            const short qid = simd_lane_id / 4;
            const short r = (qid & 4) + ((simd_lane_id / 2) % 4);
            const short c_base = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
            for (short j = 0; j < 2; j++) {
                const short gr = row_base + r;
                const short gc = col_base + c_base + j;
                if (gr < BM && gc < BN) {
                    float gate_val = combined_scale_gu * frag_C_gate[m * kMmaTilesN + n].thread_elements()[j];
                    float up_val = combined_scale_gu * frag_C_up[m * kMmaTilesN + n].thread_elements()[j];
                    tgp.p2.hidden[gr * BN_PAD + gc] = silu(gate_val) * up_val;
                }
            }
        }
    }

    // ==================================================================
    // Phase 2: down projection
    // hidden[M, inter_dim] @ w_down[K_out, inter_dim]^T -> output[M, K_out]
    // ==================================================================

    simdgroup_matrix<float, 8, 8> frag_C_down[kMmaTilesM * kMmaTilesN];
    for (int i = 0; i < kMmaTilesM * kMmaTilesN; i++) {
        for (int j = 0; j < 8; j++) {
            frag_C_down[i].thread_elements()[j] = 0.0f;
        }
    }

    // Output tiling: tid.x maps to K_out dimension
    // For inter_dim == BN, we process the full reduction in one pass
    // Load down weights for this K_out tile
    const int col_block_out = int(tid.x) * BM;
    if (col_block_out >= K_out) return;

    const device uint8_t* down_packed = w_down_packed
        + size_t(expert_idx) * K_out * (inter_dim / 2)
        + size_t(col_block_out) * (ldw_down / 2);
    const device uint8_t* down_scales = w_down_scales
        + size_t(expert_idx) * K_out * (inter_dim / group_size)
        + size_t(col_block_out) * (lds_down / group_size);

    float scale_b_dn = scales_b_down[expert_idx * K_out + col_block_out];

    // Load down weight tile
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FP4WeightLoader<BM, BN, BN_PAD, tgp_size, group_size> loader_down(
        down_packed, ldw_down / 2,
        down_scales, lds_down / group_size,
        tgp.p2.Bs2_packed, tgp.p2.Bs2_scales,
        simd_group_id, simd_lane_id);
    loader_down.load_unsafe();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2 MMA: hidden (float in tgp) x w_down (fp4 in tgp) -> output
    simdgroup_matrix<float, 8, 8> frag_a2;
    simdgroup_matrix<float, 8, 8> frag_b2;

    constexpr short S2_COLS = BN / group_size;

    for (short k = 0; k < BN; k += 8) {
        for (short m = 0; m < kMmaTilesM; m++) {
            const short a_row_base = warp_row * SM + m * 8;
            for (short j = 0; j < 2; j++) {
                const short qid = simd_lane_id / 4;
                const short r = (qid & 4) + ((simd_lane_id / 2) % 4);
                const short c = k + (qid & 2) * 2 + (simd_lane_id % 2) * 2 + j;
                const short gr = a_row_base + r;
                frag_a2.thread_elements()[j] = (gr < BM && c < BN)
                    ? tgp.p2.hidden[gr * BN_PAD + c]
                    : 0.0f;
            }

            for (short n = 0; n < kMmaTilesN; n++) {
                const short b_col_base = warp_col * SN + n * 8;
                for (short j = 0; j < 2; j++) {
                    const short qid = simd_lane_id / 4;
                    const short r = k + (qid & 4) + ((simd_lane_id / 2) % 4);
                    const short c = b_col_base + (qid & 2) * 2 + (simd_lane_id % 2) * 2 + j;
                    if (r < BN && c < BM) {
                        short pc = r / 2;
                        uint8_t packed = tgp.p2.Bs2_packed[c * (BN_PAD / 2) + pc];
                        uint8_t nib = (r % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
                        float val = dequant_fp4(nib);
                        float sc = float(fp8_e8m0(tgp.p2.Bs2_scales[c * S2_COLS + r / group_size]));
                        frag_b2.thread_elements()[j] = val * sc;
                    } else {
                        frag_b2.thread_elements()[j] = 0.0f;
                    }
                }
                simdgroup_multiply_accumulate(
                    frag_C_down[m * kMmaTilesN + n], frag_a2, frag_b2,
                    frag_C_down[m * kMmaTilesN + n]);
            }
        }
    }

    // Store phase 2 result to device memory
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* out_tile = out + size_t(row_block) * ldo + col_block_out;
    for (short m = 0; m < kMmaTilesM; m++) {
        for (short n = 0; n < kMmaTilesN; n++) {
            const short row_base = warp_row * SM + m * 8;
            const short col_base = warp_col * SN + n * 8;
            const short qid = simd_lane_id / 4;
            const short r = (qid & 4) + ((simd_lane_id / 2) % 4);
            const short c_base = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
            for (short j = 0; j < 2; j++) {
                const short gr = row_base + r;
                const short gc = col_base + c_base + j;
                if (gr < BM && gc < BM) {
                    out_tile[gr * ldo + gc] = scale_b_dn * frag_C_down[m * kMmaTilesN + n].thread_elements()[j];
                }
            }
        }
    }
}
