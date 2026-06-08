import logging

import mlx.core as mx
from mlx_lm.generate import (
    GenerationBatch,
    PromptProcessingBatch,
    generation_stream,
)

logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Version guard — warn when mlx-lm is outside tested range so operators
# know patches may not apply correctly.
# ---------------------------------------------------------------------------
try:
    import importlib.metadata
    _mlx_lm_ver = importlib.metadata.version("mlx-lm")
except Exception:
    _mlx_lm_ver = "unknown"

# Update these bounds when testing against new mlx-lm releases.
_MLX_LM_MIN_TESTED = "0.21"
_MLX_LM_MAX_TESTED = "0.25"

if _mlx_lm_ver != "unknown":
    def _ver_tuple(v):
        return tuple(int(x) for x in v.split(".")[:3])
    vt = _ver_tuple(_mlx_lm_ver)
    if vt < _ver_tuple(_MLX_LM_MIN_TESTED) or vt > _ver_tuple(_MLX_LM_MAX_TESTED):
        logger.warning(
            "mlx-lm %s is outside the tested range [%s, %s]. "
            "Monkeypatches may not apply correctly. "
            "Please verify after upgrading.",
            _mlx_lm_ver, _MLX_LM_MIN_TESTED, _MLX_LM_MAX_TESTED,
        )
    else:
        logger.debug("mlx-lm %s within tested range [%s, %s]",
                      _mlx_lm_ver, _MLX_LM_MIN_TESTED, _MLX_LM_MAX_TESTED)


# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.
_default_generation_stream = generation_stream


# ---------------------------------------------------------------------------
# Monkey-patch GenerationBatch._step to call grammar accept_token() after
# sampling.  In the pipelined _step(), logits processors fill the bitmask
# (constrain NEXT token) but can't know which token was just sampled.
# After _original_step returns, self._next_tokens holds the freshly sampled
# tokens.  We eval them synchronously and accept in grammar processors.
# ---------------------------------------------------------------------------
_original_generation_batch_step = GenerationBatch._step


def _patched_generation_batch_step(self):
    # Build per-batch mRoPE deltas from UID mapping before each step.
    # This handles batch size changes during prompt split/generate.
    model = self.model
    if (
        getattr(model, "_uses_mrope", False)
        and getattr(model, "_uid_rope_deltas", None)
        and self.uids
    ):
        deltas = [model._uid_rope_deltas.get(uid, 0.0) for uid in self.uids]
        model.set_batch_rope_deltas(mx.array(deltas))

    result = _original_generation_batch_step(self)

    # self._next_tokens contains the just-sampled tokens (async eval pending).
    # We need to accept them NOW so the next __call__ fills the correct bitmask.
    if any(self.logits_processors):
        from ..api.grammar import GrammarConstraintProcessor

        has_grammar = any(
            isinstance(p, GrammarConstraintProcessor)
            for procs in self.logits_processors
            for p in procs
        )
        if has_grammar:
            # Force eval of the sampled tokens so we can read them.
            mx.eval(self._next_tokens)
            sampled = self._next_tokens.tolist()
            for e in range(len(self.uids)):
                for proc in self.logits_processors[e]:
                    if isinstance(proc, GrammarConstraintProcessor):
                        proc.accept_token(sampled[e])

    return result


GenerationBatch._step = _patched_generation_batch_step


# Monkey-patch TurboQuantKVCache.merge so _merge_caches() works
try:
    from mlx_vlm.turboquant import TurboQuantKVCache as _TQCache

    from ..turboquant_kv import BatchTurboQuantKVCache as _BTQCache

    if not hasattr(_TQCache, "merge"):
        _TQCache.merge = _BTQCache.merge
except ImportError:
    pass


# Monkey-patch ChunkedKVCache for Llama-4 (Scout / Maverick): mlx_lm's
# ChunkedKVCache lacks the batch-aware methods (`merge`, `filter`, `extract`,
# `size`, `extend`) that BatchGenerator's continuous-batching code path
# expects, so any chat completion targeting a Llama-4 model raises
# `Cache corruption not recoverable: <ChunkedKVCache> does not yet support
# batching with history` and returns 500.
#
# Real continuous batching with chunked attention is unimplemented upstream;
# this patch installs batch=1 pass-throughs so serialized requests work.
# Run the server with `--max-concurrent-requests 1` to honor the assumption.
try:
    from mlx_lm.models.cache import ChunkedKVCache as _CKVCache

    _ckvcache_methods_skipped: list[str] = []

    if not hasattr(_CKVCache, "merge"):
        @classmethod
        def _ckvcache_merge_passthrough(cls, caches):
            if len(caches) == 1:
                return caches[0]
            raise NotImplementedError(
            "ChunkedKVCache.merge for batch_size > 1 is not implemented. "
            "Run with --max-concurrent-requests 1 when serving Llama-4."
            )

        _CKVCache.merge = _ckvcache_merge_passthrough
    else:
        _ckvcache_methods_skipped.append("merge")

    if not hasattr(_CKVCache, "filter"):
        def _ckvcache_filter_passthrough(self, batch_indices):
            try:
                n = len(batch_indices)
            except TypeError:
                n = int(getattr(batch_indices, "shape", (0,))[0] or 0)
            if n == 0:
                self.keys = None
                self.values = None
                self.offset = 0
                self.start_position = 0
                return
            if n == 1:
                return
            raise NotImplementedError(
                f"ChunkedKVCache.filter with batch_size={n} > 1 is not "
            "implemented. Run with --max-concurrent-requests 1 when "
            "serving Llama-4."
            )

        _CKVCache.filter = _ckvcache_filter_passthrough
    else:
        _ckvcache_methods_skipped.append("filter")

    if not hasattr(_CKVCache, "extract"):
        def _ckvcache_extract_passthrough(self, idx):
            return self

        _CKVCache.extract = _ckvcache_extract_passthrough
    else:
        _ckvcache_methods_skipped.append("extract")

    if not hasattr(_CKVCache, "size"):
        def _ckvcache_size(self):
            return max(0, self.offset - self.start_position)

        _CKVCache.size = _ckvcache_size
    else:
        _ckvcache_methods_skipped.append("size")

    if not hasattr(_CKVCache, "extend"):
        def _ckvcache_extend_passthrough(self, other):
            if other is None or other.empty():
                return
            if self.empty():
                self.keys = other.keys
                self.values = other.values
                self.offset = other.offset
                self.start_position = other.start_position
                return
            raise NotImplementedError(
            "ChunkedKVCache.extend across non-empty caches is not "
            "supported. Run with --max-concurrent-requests 1."
            )

        _CKVCache.extend = _ckvcache_extend_passthrough
    else:
        _ckvcache_methods_skipped.append("extend")

    if _ckvcache_methods_skipped:
        # Upstream may have landed implementations between mlx_lm upgrades.
        # Surface which ones so a regression in Llama-4 batching is visible
        # to operators without diffing the patch against installed mlx_lm.
        logger.info(
            "ChunkedKVCache patch: methods already present upstream, "
            "skipped: %s",
            ", ".join(_ckvcache_methods_skipped),
        )
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Monkey-patch PromptProcessingBatch.prompt to set mRoPE deltas before the
# prompt processing loop.  Without this, batched VLM prompt processing
# (e.g. the 1-token final prompt after external prefill) would use
# per-request offsets without rope_deltas, corrupting attention masks
# for concurrent VLM requests.
# ---------------------------------------------------------------------------
_original_ppb_prompt = PromptProcessingBatch.prompt


def _patched_ppb_prompt(self, tokens):
    model = self.model
    if (
        getattr(model, "_uses_mrope", False)
        and getattr(model, "_uid_rope_deltas", None)
        and self.uids
    ):
        deltas = [model._uid_rope_deltas.get(uid, 0.0) for uid in self.uids]
        model.set_batch_rope_deltas(mx.array(deltas))
    return _original_ppb_prompt(self, tokens)


PromptProcessingBatch.prompt = _patched_ppb_prompt
