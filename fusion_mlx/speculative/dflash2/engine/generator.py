# SPDX-License-Identifier: Apache-2.0
# DFlash2 in-target drafter — proposes draft blocks using the SCHEDULER's
# already-loaded target model. No duplicate 27B load, no prefill replay.
#
# The official dflash pkg (PyPI, z-lab) provides DFlash2DraftModel.propose
# which reads target hidden states (captured via _LayerHook on
# target_layer_ids) and produces a draft block. The verify+rollback loop
# runs in the scheduler's dflash2_spec_step (spec_decode.py), reusing
# _run_spec_verify / _trim_trimmable — same in-target pattern as DFlash-v1.

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def _local_draft_path(draft_repo: str):
    import os
    from pathlib import Path

    if not draft_repo:
        return None
    p = Path(draft_repo)
    candidates = [p] if p.is_absolute() else []
    model_dir = os.environ.get("FUSION_MODEL_DIR") or os.path.expanduser(
        "~/.fusion-mlx/models"
    )
    candidates.append(Path(model_dir) / draft_repo)
    for c in candidates:
        if c.is_dir() and (c / "config.json").exists():
            return c
    return None


class DFlash2InTargetDrafter:
    def __init__(
        self,
        draft_repo: str,
        block_size: int = 5,
        draft_bits: int | None = None,
    ) -> None:
        if not draft_repo:
            raise ValueError("draft_repo must be a non-empty string")
        if block_size <= 0 or block_size > 8:
            raise ValueError(f"block_size must be in [1, 8]; got {block_size}")
        if draft_bits is not None and draft_bits not in (4, 8):
            raise ValueError(f"draft_bits must be 4 or 8; got {draft_bits}")
        from dflash import model_mlx as _dflash

        logger.info("[dflash2] loading draft=%s (DFlash2DraftModel)", draft_repo)
        draft_path = _local_draft_path(draft_repo)
        if draft_path is not None:
            _orig_download = _dflash.snapshot_download
            _dflash.snapshot_download = lambda _id, **_kw: str(draft_path)
            try:
                self.draft = _dflash.load_draft(str(draft_path))
            finally:
                _dflash.snapshot_download = _orig_download
        else:
            self.draft = _dflash.load_draft(draft_repo)
        if draft_bits is not None:
            import mlx.nn as nn

            nn.quantize(self.draft, group_size=64, bits=draft_bits)
            logger.info(
                "[dflash2] draft quantized to %d bits (group_size=64)", draft_bits
            )
        self.draft_repo = draft_repo
        self.block_size = block_size
        self.mask_id = int(self.draft.config.mask_token_id)
        self.layer_ids = tuple(self.draft.config.target_layer_ids)
        self._all_sliding = all(
            t == "sliding_attention" for t in (self.draft.config.layer_types or ())
        )
        self._hidden_limit = (
            self.draft.config.sliding_window - 1 if self._all_sliding else None
        )
        self._target = None
        self._bound = False
        self._draft_cache = None
        self._last_hidden = None
        self._last_draft_indices = None
        self._last_draft_probs = None
        logger.info(
            "[dflash2] drafter ready draft=%s block_size=%d draft_bits=%s "
            "layer_ids=%s hidden_limit=%s",
            draft_repo,
            block_size,
            draft_bits,
            self.layer_ids,
            self._hidden_limit,
        )

    @property
    def loaded(self) -> bool:
        return self.draft is not None

    def bind(self, target_model) -> None:
        if self._bound:
            logger.warning("[dflash2] drafter already bound — skipping")
            return
        from dflash.model_mlx import _patch_model

        self.draft.bind(target_model)
        _patch_model(target_model, list(self.layer_ids))
        self._target = target_model
        self._bound = True
        logger.info(
            "[dflash2] drafter bound to target=%s (hooks on layers %s)",
            type(target_model).__name__,
            self.layer_ids,
        )

    def reset(self) -> None:
        self._draft_cache = self.draft.make_cache()
        self._last_hidden = None

    def align_draft_cache(self, target_offset: int) -> None:
        if self._draft_cache is None:
            self.reset()
        for c in self._draft_cache:
            c.offset = target_offset

    def get_hidden(self, model) -> mx.array:
        if self._last_hidden is not None:
            return self._last_hidden
        hs = getattr(model, "_hidden_states", None)
        if not hs or any(h is None for h in hs):
            return mx.zeros((1, 1, self.draft.config.hidden_size * len(self.layer_ids)))
        hidden = mx.concatenate(hs, axis=-1)
        if self._hidden_limit is not None:
            hidden = hidden[:, -self._hidden_limit :]
        return hidden

    def propose_block(
        self,
        current_token: int,
        hidden: mx.array,
        temperature: float,
    ) -> mx.array:
        bs = self.block_size
        block = mx.array([[current_token] + [self.mask_id] * (bs - 1)], dtype=mx.uint32)
        draft_tokens, draft_indices, draft_probs = self.draft.propose(
            block,
            hidden,
            self._draft_cache,
            temperature,
            logits_start=1,
        )
        self._last_draft_indices = draft_indices
        self._last_draft_probs = draft_probs
        return draft_tokens

    def store_verify_hidden(self, model) -> None:
        hs = getattr(model, "_hidden_states", None)
        if not hs or any(h is None for h in hs):
            return
        hidden = mx.concatenate(hs, axis=-1)
        if self._hidden_limit is not None:
            hidden = hidden[:, -self._hidden_limit :]
        self._last_hidden = hidden

    def trim_draft_cache(self, expected_offset: int) -> None:
        if self._draft_cache is None:
            return
        actual = self._draft_cache[0].offset
        trim_n = actual - expected_offset
        if trim_n <= 0:
            return
        from dflash.model_mlx import _trim_recent_cache

        _trim_recent_cache(self._draft_cache, trim_n)
