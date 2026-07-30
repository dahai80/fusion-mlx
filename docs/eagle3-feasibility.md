# EAGLE-3 Feasibility Assessment for fusion-mlx

## Summary

**Verdict: NOT FEASIBLE in the near term (6+ months)**

EAGLE-3 requires extracting hidden states (intermediate layer activations) from the target model's forward pass to train and run a drafter model. This is fundamentally incompatible with MLX's current architecture.

## What EAGLE-3 Does

EAGLE-3 (Extrapolation Algorithm for Greater Language-model Efficiency v3) is a speculative decoding drafter that:
1. Extracts hidden states from the target LLM at specific transformer layers
2. Trains a lightweight autoregressive drafter on these hidden states
3. At inference, feeds the target's hidden state into the drafter to predict N tokens ahead
4. Verifies draft tokens against the target in a single forward pass

Key requirement: **access to intermediate layer activations** during target model forward().

## Why It Doesn't Work on MLX

### 1. No Hidden State Extraction API
MLX's `nn.Module.forward()` is a monolithic call — there is no hook/callback mechanism to extract intermediate layer outputs. PyTorch has `register_forward_hook()`; MLX does not.

### 2. Architecture Coupling
Hidden state dimensions and layer positions are model-specific. EAGLE-3 needs:
- Layer index selection (typically layer[-2] or similar)
- Activation tensors at that layer for every token position
- These activations as input features for the drafter network

### 3. Training Requirement
EAGLE-3 requires training the drafter on the target model's hidden states. This means:
- Running the full target model on a corpus
- Extracting and saving hidden states
- Training the drafter (requires a training loop, optimizer, loss)
- MLX training infrastructure exists but is less mature than PyTorch

### 4. No Existing Code
There is zero EAGLE-3 code in fusion-mlx or the MLX ecosystem. The only implementation is the original PyTorch/CUDA repo.

## What Would Be Needed

| Component | Effort | Risk |
|-----------|--------|------|
| Hook system for MLX nn.Module | 2-3 weeks | Medium — API design, upstream MLX buy-in |
| Hidden state extraction per model family | 1-2 weeks | Low — straightforward once hooks exist |
| Drafter architecture port | 1-2 weeks | Low — standard transformer |
| Training pipeline for drafter | 2-3 weeks | Medium — needs corpus + compute |
| Integration with speculative decoder | 1-2 weeks | Low — existing framework in place |
| **Total** | **7-12 weeks** | **High** (MLX hook system is blocker) |

## Alternatives

1. **DFlash/DDTree** — Already implemented. Uses existing drafter models without hidden state extraction.
2. **Medusa heads** — Multi-token prediction heads added to target model. Simpler than EAGLE-3 but still needs training.
3. **Prompt-lookup decoding** — No model changes, matches n-grams from prompt. Trivial to implement but limited gains.

## Recommendation

Defer EAGLE-3 until MLX adds forward hooks (track apple/mlx roadmap). Invest in DFlash/DDTree improvements instead — already functional and provides real speedups for supported models.
