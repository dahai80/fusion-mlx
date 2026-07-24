// W4A8 Fused MatMul Metal Kernel
// DeepGEMM-inspired: delayed dequant + two-level post-accumulation scaling
//
// A (activation): fp8_e4m3 stored raw in threadgroup, dequantized per-K-tile in MMA loop
// B (weight):     fp4_e2m1 packed + fp8_e8m0 (MX scale), dequantized per-K-tile in MMA loop
// C (output):     fp32 accumulation, then apply scale_a * scale_b post-accumulation
//
// Key optimization over MLX fp_quantized.h:
//   - A-side: fp8 raw in threadgroup -> ~2x less shared memory than fp16
//   - B-side: fp4 packed + scale raw in threadgroup -> ~4x less shared memory
//   - Two-level scale: scale_a * scale_b applied ONCE after full K-loop, not per-tile

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

// ---------------------------------------------------------------------------
// Type definitions (matching MLX fp8.h / fp4.h)
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

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constant constexpr const int SIMD_SIZE = 32;

// ---------------------------------------------------------------------------
// BlockLoader for A (activation): fp8_e4m3 raw storage in threadgroup
// ---------------------------------------------------------------------------

template <
    typename OutT,
    short BM,
    short BK,
    short BK_PAD,
    short tgp_size>
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
// BlockLoader for B (weight): fp4 packed + MX scale in threadgroup
// ---------------------------------------------------------------------------

template <
    typename OutT,
    short BN,
    short BK,
    short BK_PAD,
    short BN_PAD,
    short tgp_size,
    short group_size = 32>
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
// W4A8 BlockMMA with delayed dequant and two-level post-accum scaling
// ---------------------------------------------------------------------------

template <
    typename T,
    short BM, short BN, short BK,
    short WM, short WN,
    short BK_PAD,
    short BN_PAD,
    short group_size = 32>
struct W4A8BlockMMA {
    // Accumulator stored as simdgroup_matrix fragments
    simdgroup_matrix<float, 8, 8> frag_C[(BM / WM / 8) * (BN / WN / 8)];

    const short simd_group_id;
    const short simd_lane_id;
    const short warp_row;
    const short warp_col;

    static constant constexpr const short SM = BM / WM;
    static constant constexpr const short SN = BN / WN;
    static constant constexpr const short kMmaTilesM = SM / 8;
    static constant constexpr const short kMmaTilesN = SN / 8;

    W4A8BlockMMA(
        ushort simd_group_id_ [[simdgroup_index_in_threadgroup]],
        ushort simd_lane_id_ [[thread_index_in_simdgroup]])
        : simd_group_id(simd_group_id_),
          simd_lane_id(simd_lane_id_),
          warp_row(simd_group_id / WN),
          warp_col(simd_group_id % WN) {
        for (int i = 0; i < kMmaTilesM * kMmaTilesN; i++) {
            for (int j = 0; j < 8; j++) {
                frag_C[i].thread_elements()[j] = 0.0f;
            }
        }
    }

    // Inline dequant fp8 -> fp32 fragment from threadgroup
    METAL_FUNC void load_frag_a(
        thread simdgroup_matrix<float, 8, 8>& frag,
        threadgroup uint8_t* As,
        const short k_offset) const {
        // simdgroup_matrix<8,8> layout: each thread holds 2 elements
        // Element coords from simd_lane_id
        const short qid = simd_lane_id / 4;
        const short row_in_8x8 = (qid & 4) + ((simd_lane_id / 2) % 4);
        const short col_in_8x8 = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

        const short global_row = warp_row * SM + row_in_8x8;
        const short global_col = k_offset + col_in_8x8;

        for (short j = 0; j < 2; j++) {
            short c = global_col + j;
            if (global_row < BM && c < BK) {
                frag.thread_elements()[j] = float(fp8_e4m3(As[global_row * BK_PAD + c]));
            } else {
                frag.thread_elements()[j] = 0.0f;
            }
        }
    }

    // Inline dequant fp4 packed -> fp32 fragment with MX scale
    METAL_FUNC void load_frag_b(
        thread simdgroup_matrix<float, 8, 8>& frag,
        threadgroup uint8_t* Bs_packed,
        threadgroup uint8_t* Bs_scales,
        const short k_offset) const {
        const short qid = simd_lane_id / 4;
        const short row_in_8x8 = (qid & 4) + ((simd_lane_id / 2) % 4);
        const short col_in_8x8 = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

        const short global_row = k_offset + row_in_8x8;
        const short global_col = warp_col * SN + col_in_8x8;

        for (short j = 0; j < 2; j++) {
            short c = global_col + j;
            if (global_row < BK && c < BN) {
                short packed_col = global_row / 2;
                uint8_t packed = Bs_packed[c * (BK_PAD / 2) + packed_col];
                uint8_t nibble = (global_row % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
                float val = dequant_fp4(nibble);
                short scale_idx = global_row / group_size;
                float scale = float(fp8_e8m0(Bs_scales[c * S_COLS + scale_idx]));
                frag.thread_elements()[j] = val * scale;
            } else {
                frag.thread_elements()[j] = 0.0f;
            }
        }
    }

    static constant constexpr const short S_COLS = BK / group_size;

    // Full MMA step over BK tile
    METAL_FUNC void mma_step(
        threadgroup uint8_t* As_fp8,
        threadgroup uint8_t* Bs_packed,
        threadgroup uint8_t* Bs_scales) {

        simdgroup_matrix<float, 8, 8> frag_a;
        simdgroup_matrix<float, 8, 8> frag_b;

        // Step through BK in 8-element chunks
        for (short k = 0; k < BK; k += 8) {
            for (short m = 0; m < kMmaTilesM; m++) {
                // Temporarily adjust warp offset for this MMA tile
                // Load A fragment with row offset
                const short a_row_base = warp_row * SM + m * 8;
                for (short j = 0; j < 2; j++) {
                    const short qid = simd_lane_id / 4;
                    const short r = (qid & 4) + ((simd_lane_id / 2) % 4);
                    const short c = k + (qid & 2) * 2 + (simd_lane_id % 2) * 2 + j;
                    const short gr = a_row_base + r;
                    if (gr < BM && c < BK) {
                        frag_a.thread_elements()[j] = float(fp8_e4m3(As_fp8[gr * BK_PAD + c]));
                    } else {
                        frag_a.thread_elements()[j] = 0.0f;
                    }
                }

                for (short n = 0; n < kMmaTilesN; n++) {
                    // Load B fragment
                    const short b_col_base = warp_col * SN + n * 8;
                    for (short j = 0; j < 2; j++) {
                        const short qid = simd_lane_id / 4;
                        const short r = k + (qid & 4) + ((simd_lane_id / 2) % 4);
                        const short c = b_col_base + (qid & 2) * 2 + (simd_lane_id % 2) * 2 + j;
                        if (r < BK && c < BN) {
                            short packed_col = r / 2;
                            uint8_t packed = Bs_packed[c * (BK_PAD / 2) + packed_col];
                            uint8_t nibble = (r % 2 == 0) ? (packed & 0x0F) : (packed >> 4);
                            float val = dequant_fp4(nibble);
                            short scale_idx = r / group_size;
                            float scale = float(fp8_e8m0(Bs_scales[c * S_COLS + scale_idx]));
                            frag_b.thread_elements()[j] = val * scale;
                        } else {
                            frag_b.thread_elements()[j] = 0.0f;
                        }
                    }

                    // C += A * B
                    simdgroup_multiply_accumulate(
                        frag_C[m * kMmaTilesN + n], frag_a, frag_b,
                        frag_C[m * kMmaTilesN + n]);
                }
            }
        }
    }

    // Apply two-level scaling and store result
    METAL_FUNC void store_result(
        device T* dst,
        const int dst_ld,
        const float scale_a,
        const float scale_b) const {
        const float combined_scale = scale_a * scale_b;

        for (short m = 0; m < kMmaTilesM; m++) {
            for (short n = 0; n < kMmaTilesN; n++) {
                const short row_base = warp_row * SM + m * 8;
                const short col_base = warp_col * SN + n * 8;

                // simdgroup_matrix elements per thread = 2
                const short qid = simd_lane_id / 4;
                const short r = (qid & 4) + ((simd_lane_id / 2) % 4);
                const short c_base = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

                for (short j = 0; j < 2; j++) {
                    const short gr = row_base + r;
                    const short gc = col_base + c_base + j;
                    if (gr < BM && gc < BN) {
                        dst[gr * dst_ld + gc] = T(combined_scale * frag_C[m * kMmaTilesN + n].thread_elements()[j]);
                    }
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Kernel entry point
// ---------------------------------------------------------------------------

template <
    typename T,
    short BM, short BN, short BK,
    short WM, short WN,
    short group_size = 32>
[[kernel, max_total_threads_per_threadgroup(WM * WN * SIMD_SIZE)]]
void w4a8_fused_matmul(
    const device uint8_t* x_fp8       [[buffer(0)]],
    const device float* scales_a       [[buffer(1)]],
    const device uint8_t* w_packed     [[buffer(2)]],
    const device uint8_t* w_scales     [[buffer(3)]],
    const device float* scales_b       [[buffer(4)]],
    device T* out                       [[buffer(5)]],
    const constant int& M              [[buffer(6)]],
    const constant int& K              [[buffer(7)]],
    const constant int& N              [[buffer(8)]],
    const constant int& lda            [[buffer(9)]],
    const constant int& ldw            [[buffer(10)]],
    const constant int& lds            [[buffer(11)]],
    const constant int& ldo            [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {

    constexpr short BK_PAD = BK + 16 / sizeof(uint8_t);
    constexpr short BN_PAD = BN + 4;
    constexpr short tgp_size = WM * WN * SIMD_SIZE;
    constexpr short S_COLS = BK / group_size;

    // Threadgroup memory: A in fp8 raw, B in fp4 packed + scales
    threadgroup uint8_t As_fp8[BM * BK_PAD];
    threadgroup uint8_t Bs_packed[BN * (BK_PAD / 2)];
    threadgroup uint8_t Bs_scales[BN * S_COLS];

    const int row_block = int(tid.y) * BM;
    const int col_block = int(tid.x) * BN;
    const int K_IT = K / BK;

    const short valid_m = short(min(int(BM), M - row_block));
    const short valid_n = short(min(int(BN), N - col_block));

    // Get scale_a for this row block (assuming per-tile scale)
    float scale_a = (row_block < M) ? scales_a[row_block] : 1.0f;
    // Get scale_b for this col block
    float scale_b = (col_block < N) ? scales_b[col_block] : 1.0f;

    // Initialize MMA
    W4A8BlockMMA<T, BM, BN, BK, WM, WN, BK_PAD, BN_PAD, group_size>
        mma_op(simd_group_id, simd_lane_id);

    // Pointers for this tile
    const device uint8_t* x_tile = x_fp8 + size_t(row_block) * lda;
    const device uint8_t* w_tile = w_packed + size_t(col_block) * (ldw / 2);
    const device uint8_t* s_tile = w_scales + size_t(col_block) * (lds / group_size);

    // GEMM loop over K
    for (int k_it = 0; k_it < K_IT; k_it++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load A tile (fp8 raw)
        FP8ActivationLoader<T, BM, BK, BK_PAD, tgp_size> loader_a(
            x_tile + k_it * BK,
            lda,
            As_fp8,
            simd_group_id,
            simd_lane_id);

        if (valid_m == BM) {
            loader_a.load_unsafe();
        } else {
            loader_a.load_safe(valid_m, BK);
        }

        // Load B tile (fp4 packed + scales)
        FP4WeightLoader<T, BN, BK, BK_PAD, BN_PAD, tgp_size, group_size>
            loader_b(
                w_tile + k_it * BK / 2,
                ldw / 2,
                s_tile + k_it * BK / group_size,
                lds / group_size,
                Bs_packed,
                Bs_scales,
                simd_group_id,
                simd_lane_id);

        if (valid_n == BN) {
            loader_b.load_unsafe();
        } else {
            loader_b.load_safe(valid_n, BK);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MMA step with delayed dequant
        mma_op.mma_step(As_fp8, Bs_packed, Bs_scales);
    }

    // Apply two-level scaling and store
    device T* out_tile = out + size_t(row_block) * ldo + col_block;
    mma_op.store_result(out_tile, ldo, scale_a, scale_b);
}
