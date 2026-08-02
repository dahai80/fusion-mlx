# Phase-2: Speculative Decode Enhancement Plan

## Current State (Phase-1 Completed)
- HiddenStateCapture installed on last layer (31/32), captures hidden states
- d2t logits bug fixed (`mx.full(-1e9)` instead of `mx.zeros`)
- Eagle3 receives real hidden states → acceptance rate 0% → ~2% D1 / 20% token-level
- `Eagle3Model.fc = nn.Linear(3 * target_hidden_size, hidden_size)` exists but only 2x used (embed+hidden via FirstLayer)

## Phase-2 Changes (4 items, ordered by impact)

### 1. Multi-Layer Capture (HIGH IMPACT)
**Problem**: Currently capturing only layer 31 (last). Eagle3's `fc` layer takes `3 * hidden_size` input, suggesting the architecture was designed for `embed + 2 hidden layers` (3 × 4096 = 12288). Only using 2x (embed+1 hidden) wastes the fc layer's capacity.

**Change**:
- `engine_core.py`: Capture layers `[num_layers-1, num_layers//2]` (e.g., layers 31 and 15 for 32-layer model)
- `hidden_capture.py`: Already supports multiple `layer_ids` — no code change needed
- `spec_decode.py`: Extract ALL captured layers, concatenate them, pass multi-layer hidden state to Eagle3
- `eagle3/model.py`:
  - Modify `forward_standalone()` to accept `hidden_states: dict[int, mx.array]` (multiple layers)
  - Concatenate captured hidden states → feed into `self.fc` which already has `3*hidden_size` input
  - Pass fc output as `hidden` into FirstLayer (replaces the current single-layer path)

**Files**: `engine_core.py`, `spec_decode.py`, `eagle3/model.py`, `eagle3/speculator.py`

### 2. Model-Matching Guard (SAFETY)
**Problem**: No guard prevents running Eagle3-LLaMA3 with a Qwen target model, which would produce garbage.

**Change**:
- `eagle3/speculator.py`: Add `is_compatible(target_model_name: str) -> bool` that checks `target_family` against target model name
- `engine_core.py`: After loading Eagle3, call `is_compatible()` against the loaded model name. If incompatible, log warning and skip Eagle3 (fall back to no spec decode)
- Add `target_family` matching: llama3 → "llama", qwen3 → "qwen"

**Files**: `eagle3/speculator.py`, `engine_core.py`

### 3. Adaptive Pause/Resume Tuning (EFFICIENCY)
**Problem**: Current adaptive pause threshold is 10% (SPEC_MIN_ACCEPT_RATE=0.10). With multi-layer capture, acceptance should improve. Need to tune parameters and add resume-check interval.

**Change**:
- `spec_decode.py`: Add `SPEC_RESUME_CHECK_INTERVAL` env var (default 10 steps). When paused, re-enable spec every N steps to probe acceptance rate
- `spec_decode.py`: Lower default `SPEC_MIN_ACCEPT_RATE` to 0.05 (5%) — even low acceptance helps throughput
- `spec_decode.py`: Add logging when pausing/resuming with current acceptance rate

**Files**: `spec_decode.py`

### 4. Draft Token Quality: Temperature Sampling (QUALITY)
**Problem**: Current `temperature=0.0` (greedy argmax) in Eagle3. For speculative decode, a small temperature can help draft diversity match the target model's distribution better.

**Change**:
- `eagle3/speculator.py`: Change default `temperature` from 0.0 to 0.1 (configurable via `FUSION_EAGLE3_DRAFT_TEMP`)
- This is a one-line config change, already supported by code

**Files**: `eagle3/speculator.py`

## Implementation Order
1. **Model-matching guard** — safety first, small change
2. **Multi-layer capture** — biggest impact, core of Phase-2
3. **Adaptive pause/resume tuning** — efficiency improvement
4. **Temperature sampling** — minor quality tuning

## Verification
- Start fusion-mlx with `FUSION_SPEC_METHOD=eagle3`
- Send test prompt, check logs for:
  - Model compatibility check passes
  - HiddenStateCapture installed on 2 layers
  - Multi-layer hidden states captured and passed to Eagle3
  - fc layer receives concatenated hidden states (shape [B, L, 3*hidden_size])
  - Acceptance rate improves vs Phase-1 (~2% D1)
