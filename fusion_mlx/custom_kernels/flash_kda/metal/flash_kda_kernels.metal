// SPDX-License-Identifier: Apache-2.0
// FlashKDA Metal compute kernels for Apple Silicon.
//
// KDA recurrence: h_t = g_t * h_{t-1} + beta_t * (k_t ⊗ v_t), o_t = q_t^T * h_t
// CHUNK = 16, K = V = 128
//
// K1: Token-parallel gate computation + inverse
//   - Each SIMD group processes one (token, head) pair
//   - Computes gate values from g, A_log, dt_bias via sigmoid + exp + clamp
//   - Produces per-chunk gate products and inverse for recurrence
//
// K2: Head-parallel recurrence
//   - Each SIMD group owns one head's state [V, K] = [128, 128]
//   - Uses simdgroup_matrix (8x8 bf16) for outer product k⊗v and query q^T*h
//   - State stored in threadgroup memory, bf16 on-chip
//   - Sequential scan across chunks within the head

#include <metal_stdlib>
using namespace metal;

constant int CHUNK [[function_constant(0)]];
constant int D [[function_constant(1)]];
constant int H [[function_constant(2)]];
constant int T [[function_constant(3)]];
constant float SCALE [[function_constant(4)]];
constant float LOWER_BOUND [[function_constant(5)]];

// K1: Gate computation kernel
// Input: g [T, H, D] gate logits, A_log [H], dt_bias [H, D]
// Output: gate_values [T, H, D], gate_inverse [T/CHUNK, H, D]
kernel void kda_k1_gate(
    device const bfloat<1>* g_ptr [[buffer(0)]],
    device const float* A_log_ptr [[buffer(1)]],
    device const float* dt_bias_ptr [[buffer(2)]],
    device bfloat<1>* gate_out [[buffer(3)]],
    device bfloat<1>* gate_inv_out [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    // Placeholder: gate computation
    // Full implementation: sigmoid(g + A_log + dt_bias) -> exp -> clamp
    int t = tid.x * CHUNK + lid.x;
    int h = tid.y;
    if (t >= T || h >= H) return;

    int idx = t * H * D + h * D + lid.x;
    float g_val = float(g_ptr[idx]);

    if (A_log_ptr) {
        g_val += A_log_ptr[h];
    }
    if (dt_bias_ptr) {
        g_val += dt_bias_ptr[h * D + lid.x];
    }

    float gate_sigmoid = 1.0f / (1.0f + exp(-g_val));
    float gate_val = exp(-exp(gate_sigmoid) * LOWER_BOUND);
    gate_val = clamp(gate_val, 0.0f, 1.0f);

    gate_out[idx] = bfloat<1>(gate_val);
}

// K2: Recurrence kernel
// Sequential scan per head with SIMD matrix operations
kernel void kda_k2_recur(
    device const bfloat<1>* q_ptr [[buffer(0)]],
    device const bfloat<1>* k_ptr [[buffer(1)]],
    device const bfloat<1>* v_ptr [[buffer(2)]],
    device const bfloat<1>* gate_ptr [[buffer(3)]],
    device const bfloat<1>* beta_ptr [[buffer(4)]],
    device bfloat<1>* out_ptr [[buffer(5)]],
    device bfloat<1>* state_out [[buffer(6)]],
    device const bfloat<1>* state_in [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    // Placeholder: full recurrence implementation
    // Will use simdgroup_matrix<float, 8, 8> for:
    //   1. Gate * state elementwise
    //   2. beta * (k ⊗ v) outer product via simd_matrix_multiply
    //   3. q^T * h via simd_matrix_multiply
    // State lives in threadgroup memory [128, 128] bf16 per head
}
