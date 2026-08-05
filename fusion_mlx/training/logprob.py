import gc
import logging
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LogprobResult:
    logprob: float
    token_count: int
    per_token: list = field(default_factory=list)

    def to_dict(self):
        return {
            "logprob": self.logprob,
            "token_count": self.token_count,
            "per_token": self.per_token,
        }


def compute_logprob(model, prompt_ids, completion_ids):
    # Score sum log p(completion | prompt) via teacher forcing.
    # Single forward pass over prompt+completion (causal attention), then
    # gather log_softmax at predicting positions (inlined _score_fn from
    # mlx_lm.evaluate). logits[i] predicts token i+1, so the predictor for
    # completion[0] is logits[len(prompt)-1].
    if len(completion_ids) == 0:
        logger.warning("compute_logprob: empty completion, returning 0")
        return 0.0, 0, []

    if not isinstance(prompt_ids, mx.array):
        prompt_ids = mx.array(prompt_ids)
    if not isinstance(completion_ids, mx.array):
        completion_ids = mx.array(completion_ids)

    prompt_ids = prompt_ids.reshape(-1)
    completion_ids = completion_ids.reshape(-1)
    n_prompt = int(prompt_ids.shape[0])
    n_comp = int(completion_ids.shape[0])

    full = mx.concatenate([prompt_ids, completion_ids])
    out = model(full[None, :])
    # mlx_lm models return logits (array) when cache=None, or (logits, cache)
    # when a cache is supplied. Handle both.
    logits = out[0] if isinstance(out, tuple) else out
    # (1, L, V) -> take predictor rows for each completion token
    start = n_prompt - 1
    predictor_logits = logits[0, start : start + n_comp, :].astype(mx.float32)
    log_probs = nn.log_softmax(predictor_logits)
    targets = completion_ids[..., None]
    per_token_arr = mx.take_along_axis(log_probs, targets, axis=-1)[..., 0]
    per_token = per_token_arr.tolist()

    total = float(sum(per_token))
    logger.info(
        "compute_logprob: prompt=%d completion=%d sum_logprob=%.6f",
        n_prompt,
        n_comp,
        total,
    )
    return total, n_comp, per_token


def score_text(model_path, prompt, completion, adapter_path=None):
    # Load model (optionally with adapter), tokenize, score, evict.
    # Standalone load-and-evict path; caller handles any pool coordination.
    import mlx_lm.utils as mlx_utils

    logger.info(
        "score_text: model=%s adapter=%s prompt_len=%d completion_len=%d",
        model_path,
        adapter_path,
        len(prompt),
        len(completion),
    )
    model, tokenizer = mlx_utils.load(model_path, adapter_path=adapter_path)

    try:
        prompt_ids = tokenizer.encode(prompt)
        completion_ids = tokenizer.encode(completion)
        total, n, per_token = compute_logprob(model, prompt_ids, completion_ids)
        return LogprobResult(logprob=total, token_count=n, per_token=per_token)
    finally:
        del model
        del tokenizer
        gc.collect()
        mx.clear_cache()
        logger.info("score_text: model evicted")
