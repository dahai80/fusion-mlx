## Fixed

**Speculative decode corruption on hybrid recurrent models.** On Qwen3.6-27B-mxfp8 and other hybrid architectures (48 GatedDeltaNet recurrent layers + 16 full-attention layers, `ArraysCache`), speculative decoding produced incoherent repetition (`"useruseruser"`, `"Thinking1000000..."`) and overshot `max_tokens`.

**Root cause:** the batched verify forward `model([D1..DK], cache)` computes recurrent state in parallel, but `ArraysCache` state must update sequentially — each token depends on the prior state — so batching derails it into a repetition fixed-point. Both n-gram and draft-model spec share this verify forward, so both corrupted identically.

**Fix:** a boot-level gate in `engine_core.py` probes the loaded model's cache via the shared `model_has_recurrent_cache()` helper and skips both spec paths when recurrent layers are present, falling back to coherent pure decode. Pure-attention models keep speculative decode unchanged. The same probe powers `enrich_model_config`, so the boot gate and the config gate cannot drift apart.

## Benchmark

- **Coherent decode: ~18 tok/s overall / ~20 tok/s decode** (Apple M5 Max, 128 GB). This is the hardware ceiling for this 26.7 GB model — raw `mlx_lm.stream_generate` caps at 18.4 tok/s on the same hardware.
- **v0.4.0's reported 29.8 tok/s was a corrupt artifact, not real performance.** The 0.4.0 benchmark measured speed only (HTTP API, `max_tokens=100`, 3 runs) and never coherence-tested the output. Every 0.4.0 spec-enabled run returned `completion_tokens=199` for `max_tokens=100` — a 99-token overshoot that is the signature of speculative decode — and the emitted text was incoherent repetition. 0.4.0 running *coherently* (spec disabled) was also ~18 tok/s.
- **There is no real performance regression**: 0.4.1's coherent ~18 tok/s equals 0.4.0's true coherent throughput; 0.4.1 additionally fixes the corruption 0.4.0 shipped silently.

## Integrated (from v0.4.1-wip migration)

Rapid-MLX tool parsers, omlx model patches, oQ quantizer, telemetry, MCP, middleware.

## Tests

New `tests/unit/test_spec_recurrent_gate.py` (7 cases) covering the recurrent-cache probe and config enrichment. Net +7 passing, 0 regressions versus HEAD. 89 pre-existing failures from the rapid-mlx merge are documented test debt (broken `fusion_mlx.spec_decode` test imports), unrelated to this fix.

## macOS app

**Standalone `FusionMLX-0.4.1-macos26-tahoe.dmg` (565 MB) is attached** — a self-contained app bundling CPython 3.11 + the MLX framework layer + `fusion_mlx`, so no separate Python/donor install is needed. Drag to /Applications, then right-click → Open on first launch (or `xattr -cr /Applications/FusionMLX.app`).

Verified end-to-end on the bundled Python 3.11: the app launches, the embedded server (`python -m fusion_mlx.cli serve`) binds `127.0.0.1:11434`, `/health` / `/api/status` / `/v1/models` return 200, and a `Qwen3-0.6B-4bit` chat completion loads and generates coherently. The spec-decode gate is embedded, so recurrent models decode coherently inside the app.

Two build/run fixes landed for the bundle: `openclaw_routes.py` had a PEP 701 multi-line f-string the dev `.venv` (3.14) accepted but the bundled 3.11 rejected at import — hoisted to a local variable; and `package_dmg.sh` DMG padding was raised 64 → 512 MB so 2 GB+ bundles package reliably.
