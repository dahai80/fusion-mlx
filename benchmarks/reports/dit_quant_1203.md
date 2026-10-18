# #1203 — DiT linear-layer quantization: quality regression report

Harness: `benchmarks/dit_quant_1203_bench.py`
Raw data: `benchmarks/reports/dit_quant_1203_flux1_dev.json`

Gate: ship a quantized config only if SSIM >= 0.90 against the
full-precision reference and the no-reference sharpness/colorfulness
proxies retain >= 90%.

## Image DiT — FLUX.1-dev (PASS)

Black-forest-labs/FLUX.1-dev, 768x768, 8 Euler steps, seed 20261018, one
image per config. Reference = the same run with `quantize=None` (bf16
weights). INT4/INT8 use `mlx.nn.quantize` (group_size default) via the
engine's `quantize` argument, i.e. the same path `FUSION_FLUX_QUANT=w4`
selects in production.

| config | s/step | speed vs bf16 | PSNR | SSIM | sharpness retained | colorfulness retained | gate |
|--------|--------|---------------|------|------|--------------------|-----------------------|------|
| bf16   | 3.92   | 1.00x         | —    | —    | 1.00               | 1.00                  | ref  |
| INT4   | 3.22   | 1.22x         | 23.9 | 0.954 | 1.38              | 1.03                  | PASS |
| INT8   | 3.28   | 1.20x         | 38.5 | 0.998 | 1.07              | 1.00                  | PASS |

Notes:

- SSIM/PSNR here are same-seed but different-weight comparisons; the
  diffusion sampler amplifies small weight differences, so absolute PSNR
  is not the gate — the structural/no-reference floors are.
- Sharpness retention > 1 means the quantized render is slightly crisper
  than the 8-step reference (quantization noise acts like extra texture at
  low step counts); it is not a blur regression.
- Memory is not reported per config: the image worker runs in a
  subprocess whose peak RSS is not visible to the harness. Weight memory
  scales with bits (FLUX.1-dev DiT ~24 GB bf16 → ~7 GB INT4), which is the
  point of the INT4 path; the timing here is compute-bound at 768x768.

Verdict: FLUX.1-dev may ship INT4 (`FUSION_FLUX_QUANT=w4`) and INT8. INT8
remains the safer default when quality matters (SSIM 0.998).

## Video DiT — ltx2 / ltx2_5 (blocked on offline conversion)

Video DiT quantization in this codebase is **checkpoint-driven, not a
runtime knob**:

- `fusion_mlx/video/ltx2/utils.py:apply_quantization` and
  `fusion_mlx/video/ltx2_5/ltx2_5_model.py` quantize only when the
  checkpoint already contains `.scales` keys, with `quantization_bits` /
  `quantization_group_size` read from `split_model.json`
  (`_read_quant_config`, default q8).
- A bf16 checkpoint therefore cannot be quantized at load time: the
  quantized module needs scales/biases that only an offline conversion
  produces.

What ships today: LTX-2.5 q8 (`dgrauet/ltx-2.5-mlx-q8`) is the supported
quantized video path and is what the server loads.

Why the INT4 quality bench is not in this report: it needs
(a) an offline INT4 conversion of the LTX-2.5 DiT and (b) a
full-precision baseline — the bf16 checkpoint is 187 GB, above the 128 GB
unified memory of the target machine, so the baseline cannot be run
locally at all. Both are prerequisites, not measurements.

Status: `ltx2/ltx2_5` INT4 is marked **not shipped** here with the reason
above. The harness refuses `--modality video` with that message rather
than fabricating a number. When an INT4 checkpoint exists, point
`--model` at it and the same gate applies.

## Reproduce

```bash
python benchmarks/dit_quant_1203_bench.py \
    --modality image \
    --model black-forest-labs/FLUX.1-dev \
    --bits 4 8 --steps 8 --width 768 --height 768 \
    --out benchmarks/reports/dit_quant_1203_flux1_dev.json
```

The model must be present locally (HF cache layout under
`~/.fusion-mlx/models`); a repo id resolves to its newest snapshot, or
pass a directory.
