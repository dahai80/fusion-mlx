# SPDX-License-Identifier: Apache-2.0
"""Draft model decoder for speculative decoding.

Uses a small LM (e.g. Qwen3-0.6B-4bit) to generate K draft tokens
that are verified against the target model in a single forward pass.

The draft model runs on the same GPU stream as the target model,
producing tokens sequentially (no parallelism — the draft model is
small enough that sequential generation is fast).
"""

import logging
import time

import mlx.core as mx

logger = logging.getLogger(__name__)

DRAFT_MODEL_PATH = __import__("os").environ.get(
    "FUSION_DRAFT_MODEL_PATH",
    "/Users/dahai/.omlx/models/mlx-community/Qwen3-0.6B-4bit",
)
DRAFT_NUM_TOKENS = int(__import__("os").environ.get("FUSION_SPEC_DRAFT_TOKENS", "3"))
DRAFT_TEMPERATURE = float(__import__("os").environ.get("FUSION_SPEC_DRAFT_TEMP", "0.0"))


class DraftModelDecoder:
    """Small LM that drafts tokens for speculative decode verification."""

    def __init__(
        self, model_path: str = DRAFT_MODEL_PATH, num_draft: int = DRAFT_NUM_TOKENS
    ):
        self.model_path = model_path
        self.num_draft = num_draft
        self.model = None
        self.tokenizer = None
        self._draft_cache = None
        self._prev_token = None
        self._draft_tokens = []
        self._total_drafts = 0
        self._total_accepted = 0
        self._loaded = False

    def load(self) -> bool:
        """Load the draft model. Returns True on success."""
        if self._loaded:
            return True
        try:
            t0 = time.perf_counter()
            import mlx_lm

            self.model, self.tokenizer = mlx_lm.load(self.model_path)
            dt = time.perf_counter() - t0
            self._loaded = True
            logger.info(
                "draft_model: loaded %s in %.1fs, num_draft=%d",
                self.model_path,
                dt,
                self.num_draft,
            )
            return True
        except Exception as e:
            logger.warning("draft_model: failed to load %s: %s", self.model_path, e)
            return False

    def reset(self):
        """Reset draft cache state."""
        self._draft_cache = None
        self._prev_token = None
        self._draft_tokens = []

    def on_new_request(self, request_id: str, prompt_tokens: list[int]):
        """Reset draft state for a new request."""
        self.reset()
        logger.info(
            "draft_model: on_new_request req=%s prompt_tokens=%d",
            request_id[:8],
            len(prompt_tokens) if prompt_tokens else 0,
        )
        if self.model is not None and prompt_tokens:
            try:
                from mlx_lm.models.cache import KVCache

                with mx.stream(mx.default_stream(mx.gpu)):
                    input_ids = mx.array(prompt_tokens, mx.uint32)
                    self._draft_cache = [KVCache() for _ in self.model.layers]
                    self.model(input_ids[None], cache=self._draft_cache)
                    mx.eval(self._draft_cache)
                    logger.info(
                        "draft_model: prefill success, cache=%s, layers=%d",
                        type(self._draft_cache[0]).__name__,
                        len(self._draft_cache),
                    )
            except Exception as e:
                logger.warning("draft_model: prefill failed: %s", e)
                self._draft_cache = None

    def generate_draft_tokens(self, current_token: int) -> list[int]:
        """Generate K draft tokens starting from current_token.

        Uses the draft model's cache to generate tokens sequentially.
        Returns list of draft token IDs.
        """
        if self.model is None or not self._loaded:
            return []
        if self._draft_cache is None:
            logger.info(
                "draft_model: no cache, skipping draft for token=%d", current_token
            )
            return []

        drafts = []
        token = current_token

        try:
            with mx.stream(mx.default_stream(mx.gpu)):
                for _ in range(self.num_draft):
                    input_ids = mx.array([token], mx.uint32)
                    logits = self.model(input_ids[None], cache=self._draft_cache)
                    logits = logits.squeeze(0).squeeze(0)

                    if DRAFT_TEMPERATURE > 0:
                        from mlx_lm.sample_utils import make_sampler

                        sampler = make_sampler(temp=DRAFT_TEMPERATURE)
                        next_token = sampler(logits)
                    else:
                        next_token = mx.argmax(logits)

                    mx.eval(next_token)
                    token = int(next_token)
                    drafts.append(token)

            self._total_drafts += len(drafts)
            return drafts
        except Exception as e:
            logger.warning("draft_model: generate failed: %s", e)
            self.reset()
            return []

    def record_accepted(self, n_accepted: int):
        """Record how many draft tokens were accepted."""
        self._total_accepted += n_accepted

    def get_stats(self) -> dict:
        """Get draft model statistics."""
        rate = (
            self._total_accepted / self._total_drafts if self._total_drafts > 0 else 0.0
        )
        return {
            "model_path": self.model_path,
            "num_draft": self.num_draft,
            "total_drafts": self._total_drafts,
            "total_accepted": self._total_accepted,
            "acceptance_rate": rate,
            "loaded": self._loaded,
        }
