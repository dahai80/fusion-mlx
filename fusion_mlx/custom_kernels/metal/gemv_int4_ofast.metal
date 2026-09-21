#include <metal_stdlib>
using namespace metal;

#define SIMD_SIZE 32
#define NUM_SIMD 2
#define RES_PER_SIMD 4
#define VPT 16
#define BLOCK_SIZE 512
#define GROUP_SIZE 128
#define NPACKS (VPT / 4)

[[kernel]] void custom_kernel_gemv_uint32_t_float_float_float16_t_floatc_float16_t(
  const device uint32_t* w [[buffer(0)]],
  const device float* scales [[buffer(1)]],
  const device float* biases [[buffer(2)]],
  const device half* x [[buffer(3)]],
  const constant float* meta [[buffer(4)]],
  device half* out [[buffer(5)]],
  uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
  uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
  uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
    uint M = uint(meta[0]); uint K = uint(meta[1]);
    uint n_groups = K / GROUP_SIZE; uint K_u16 = K / 4;
    uint tg_y = threadgroup_position_in_grid.x;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;
    const int out_row = int(tg_y) * (NUM_SIMD * RES_PER_SIMD) + int(simd_gid) * RES_PER_SIMD;
    if (out_row >= int(M)) return;
    const device uint16_t* ws = (const device uint16_t*)w;
    ws += out_row * K_u16 + simd_lid * (VPT / 4);
    scales += out_row * n_groups + simd_lid / 8;
    biases += out_row * n_groups + simd_lid / 8;
    x += simd_lid * VPT; out += out_row;
    thread float x_thread[VPT];
    thread float result[RES_PER_SIMD];
    #pragma unroll
    for (int r = 0; r < RES_PER_SIMD; r++) result[r] = 0.0f;
    for (uint k = 0; k < K; k += BLOCK_SIZE) {
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < VPT; i += 4) {
            float xv0 = float(x[i]), xv1 = float(x[i+1]), xv2 = float(x[i+2]), xv3 = float(x[i+3]);
            sum += xv0 + xv1 + xv2 + xv3;
            x_thread[i] = xv0; x_thread[i+1] = xv1 / 16.0f;
            x_thread[i+2] = xv2 / 256.0f; x_thread[i+3] = xv3 / 4096.0f;
        }
        #pragma unroll
        for (int row = 0; row < RES_PER_SIMD; row++) {
            const device uint16_t* wl = ws + row * K_u16;
            float sc = scales[row * n_groups]; float bi = biases[row * n_groups];
            float accum = 0.0f;
            #pragma unroll
            for (int i = 0; i < NPACKS; i++) {
                uint16_t wv = wl[i];
                accum += (x_thread[4*i] * float(wv & 0x000f)
                        + x_thread[4*i + 1] * float(wv & 0x00f0)
                        + x_thread[4*i + 2] * float(wv & 0x0f00)
                        + x_thread[4*i + 3] * float(wv & 0xf000));
            }
            result[row] += sc * accum + sum * bi;
        }
        ws += BLOCK_SIZE / 4; scales += BLOCK_SIZE / GROUP_SIZE;
        biases += BLOCK_SIZE / GROUP_SIZE; x += BLOCK_SIZE;
    }
    #pragma unroll
    for (int row = 0; row < RES_PER_SIMD; row++) {
        result[row] = simd_sum(result[row]);
        if (simd_lid == 0) out[row] = half(result[row]);
    }
}
