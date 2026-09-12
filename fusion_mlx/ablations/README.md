# Ablations Isolation Zone

Experimental, half-validated, or falsified-but-kept-for-repro code lives here,
env-gated OFF by default. The main trunk never imports these modules directly —
it calls `load_ablation(name)`, which returns `None` unless
`FUSION_ABLATION_<NAME>=1` is set.

## Rules

1. **Env-gated OFF**: default behavior = ablation not loaded, trunk unchanged.
2. **Fail-visible**: `load_ablation` logs a loud `ABLATION ACTIVE` warning when
   an ablation is enabled, so operators know non-trunk code is running.
3. **No direct imports**: trunk code calls `load_ablation("name")`, never
   `from fusion_mlx.ablations.name import ...`.
4. **Promote out when graduated**: when an ablation passes soak + bench, delete
   the env gate and move the code to its real module. This is a quarantine,
   not a permanent home.

## Candidate ablations

- **speculative_denoise**: falsified (0% acceptance). Kept here when revived
  for re-evaluation. Currently cleaned from trunk (see git history).
- **Future spikes**: any new feature merged behind an env gate before its
  72h soak / bench passes.
