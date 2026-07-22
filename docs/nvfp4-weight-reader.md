# NVFP4 Weight Reader (#179)

Software dequantizer that lets FusionMLX load NVIDIA **NVFP4** weight
checkpoints without a separate conversion step. Lives in
`fusion_mlx/custom_kernels/nvfp4.py` and is wired into the SkyReels-V3
safetensors loader.

## Why

MLX 0.32 has no native `float4_e2m1` / `from_nv_fp4` op, so an NVFP4
checkpoint (4-bit E2M1 weights + E4M3 block scales) cannot execute
directly. This reader is a **format-compatibility bridge**: it detects
NVFP4 tensors at load time and dequantizes them to `bfloat16` in memory.

This is **not a speed path**. The 4-bit storage/bandwidth benefit is
consumed at load time and is not retained at inference. The goal is
compatibility - run externally-quantized NVFP4 DiT checkpoints as-is.

## NVFP4 layout

NVIDIA TensorRT-LLM / cutlass spec:

- **Weights**: 4-bit E2M1 floats, 2 packed per `uint8` byte, little-endian
  nibble order (element `[2k]` in the low nibble, `[2k+1]` in the high
  nibble).
- **Block scales**: E4M3 float8, one scale per `NVFP4_BLOCK_SIZE = 16`
  weights, row-major (typical Linear/DiT shape `(out, in/16)`).
- **E2M1 magnitudes** (3-bit code `0..7`, sign in bit 3):

  `{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`

  The full 16-value LUT (`_NVFP4_LUT`) is the signed set indexed by the
  raw 4-bit code.

## Detection (conservative)

`dequant_nvfp4_weights` scans a `{key: mx.array}` weights dict and only
fires when **all** hold:

1. The weight tensor is `uint8` (packed NVFP4; bf16/fp32 checkpoints are
   untouched).
2. A sibling block-scale tensor exists under the weight key plus one of
   `_NVFP4_SCALE_SUFFIXES` (covers TensorRT-LLM, llm-compressor /
   compressed-tensors, and HF nvfp4 export conventions):
   `_weight_scale_inv`, `_weight_scale`, `_scale_inv`, `_scale`,
   `_block_scale`, `_weight_block_scale`, `_scaling_factor`.
3. The size relation is exact: `scale.size == (weight.size * 2) / 16`
   (one scale per 16 weights). This rules out coincidental `uint8 +
   _scale` pairs in non-NVFP4 checkpoints (e.g. raw fp8 bytes stored as
   uint8 with a per-channel float scale).

When the size relation does not match, the tensor is skipped with a
debug log - never a crash. On a normal bf16/fp32 checkpoint the pass is
a silent no-op (no `uint8` weights).

## Dequant

```
low  = packed & 0x0F
high = (packed >> 4) & 0x0F
codes   = interleave(low, high)            # element order
values  = _NVFP4_LUT[codes]                # float32
scale_f = mx.from_fp8(scales, float32)     # E4M3 -> float32 (uint8 scales)
out     = values * repeat(scale_f, 16)     # one scale per 16 weights
return  reshape(out, shape).astype(bfloat16)
```

E4M3 scale decode reuses MLX's native `mx.from_fp8`. E2M1 decode is a
software LUT gather (no native op in MLX 0.32).

## Integration

Single chokepoint: `_load_safetensors_dir` in
`fusion_mlx/video/skyreels_v3/weights.py`. Both the single-file and
directory return paths route the merged weights dict through
`dequant_nvfp4_weights`. Because every SkyReels-V3 loader
(`load_dit_weights`, `load_vae_weights`, `load_text_encoder_weights`,
`load_all_weights`) flows through this function, NVFP4 support is
inherited by all three backbones with one edit.

## Shape inference

- 2D scale `(out, in/16)` -> weight shape `(out, in)` (common
  Linear/DiT layout).
- Otherwise the flattened numel is used (`(numel,)`).
- Callers may pass explicit `shape_hints` to override.

## Testing

`tests/unit/test_nvfp4_reader.py` - 10 unit tests:

- E2M1 LUT known-answer (all 16 codes, independent of the reader's LUT).
- E4M3 block-scale multiply, sign-bit negatives.
- Multi-block 2D reshape, pack/dequant round-trip self-consistency.
- Non-block-aligned input raises.
- Dict transform: scale key consumed, sibling tensors untouched.
- 2D shape inference via the real compressed-tensors `weight_scale`
  naming.
- Non-NVFP4 dict is a no-op; size-mismatch skips dequant.

Known-answer vectors come from the NVIDIA E2M1 value set, so a wrong LUT
fails the assertion rather than hiding behind it.

## Limitations / future

- Dequant-at-load only; the 4-bit win is not retained at inference.
- `_scale` suffix is greedy but guarded by the size relation, so it
  cannot misfire on real non-NVFP4 checkpoints.
- A native MLX `from_nv_fp4` op (if added upstream) would supersede the
  software LUT path.
