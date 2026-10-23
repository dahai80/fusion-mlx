# GDN hybrid spec-decode verification — CORRECTED (2026-10-23)

Branch `research/gdn-spec-verify`. **This file supersedes the first version
of this doc, which mis-attributed the measured speedup to vendored MTP.**

## What actually happened

The original A/B served the 27B with `--text-only --enable-mtp
--mtp-num-draft-tokens 2`. Follow-up reproduction (2026-10-23 17:11) shows
that serve actually resolved **n-gram suffix**, not MTP: profile "full" is
spec-default-on, `--enable-mtp` (a boolean, not a method choice) is cleared
by the spec-auto resolver, auto routes family=qwen3_8 to dflash2, and with
no dflash2 drafter path installed it falls back to n-gram suffix. The
"zero `no spec method loaded` lines = MTP engaged" inference in the first
version was wrong.

## Corrected measurements (same day, port 11433, prod CC load in background)

- `--spec-decode none` (true baseline): 22.7 / 23.7 tok/s
- default auto → suffix: 20.3 / 32.2 tok/s

n-gram suffix helps repetitive content (math steps: +36%) and is roughly
neutral on code. The first version's "+65-90%" gap (16-19.6 baseline vs
27.9-33) is not reproducible; the baseline numbers were most likely
contaminated by concurrent prod load (image gen drops CC decode to ~1
tok/s; see #1333). The greedy byte-identity result (4/4 prompts identical
between the two serves) remains valid — n-gram verify is also greedy-exact
— but it is no longer evidence about recurrent-state handling in an MTP
verify path, because no MTP path ran.

## Native MTP on this checkpoint: blocked, needs a sidecar

The checkpoint `mlx-community/Qwen3.8-27B-4bit` (both snapshot revisions)
declares `text_config.mtp_num_hidden_layers=1` but contains **zero
`mtp.*` tensors** (mlx-lm converters strip them). The vendored MTP
injector therefore refuses:

```
[mtp.inject] inject_mtp_support called without mtp_sidecar and
allow_random_init=False; refusing to ship a random-init MTP head. Pass
mtp_sidecar='mlx-community/Qwen3.5-9B-MTP-4bit' (or equivalent)
```

The `model_settings.mtp_enabled=True` route fails the same way, loudly:
"converted weights are missing the mtp.* tensors". (Note: with
mtp_enabled=True the single-model serve additionally stalls for the full
300s load timeout instead of failing fast — the sanitize ValueError does
not propagate through the load executor; separate issue.)

The VLM-vs-text question is also resolved: `--text-only`/`--no-mllm` is
staged in `_pending_single_model["force_text"]` but **never read** — dead
flag on the single-model path. The model loads via the same
`mlx_lm.load()` either way, and the MTP injector rejects the
`Qwen3_5ForConditionalGeneration` wrapper class regardless.

## What a real MTP rollout needs

1. Download an MTP sidecar head (e.g. `mlx-community/Qwen3.5-9B-MTP-4bit`)
   and serve with `--spec-decode mtp --mtp-sidecar <dir>` — then re-run
   the greedy-equivalence A/B to validate the recurrent-state verify path
   for real.
2. Pool mode (`--model-dir`, what prod runs) additionally needs wiring:
   `_serve_from_model_dir`'s SchedulerConfig never carries
   `mtp_model_type`/`mtp_sidecar`, so no pool engine can engage MTP today.

Until (1) is done, the only spec decode actually available on the 27B is
n-gram suffix, which the default profile already enables pool-wide.

## Retained valid observations

- Roofline: 27B-4bit ≈ 15GB / ~400GB/s ≈ 26 tok/s theoretical; measured
  20-24 idle — no headroom without spec decode.
- 4B (Qwen3.5-4B-MLX-4bit) also declares `text_config.mtp_num_hidden_layers=1`
  — same sidecar story would apply.
- MTP eligibility probe and reconciliation machinery
  (`_detect_mtp_eligibility_verbose`, config_resolve reconciliation) work;
  the gap is checkpoint weights + pool wiring.
