# SPDX-License-Identifier: Apache-2.0
"""DSpine drafter wrapper for the fusion-mlx spec framework.

Bundles the ported reference core (model/propose/bridge/config/target) behind
the drafter interface the scheduler expects:

- ``DSpineDrafter.load()``: load DSpine-b16 checkpoint + bind target embeddings
- ``DSpineDrafter.draft_block(...)``: one round of block-parallel drafting
  (draft forward + RefinementTail greedy), returns candidate token ids
- ``DSpineDrafter.append_context(...)`` / ``trim_context(n)``: maintain the
  draft-side context K/V and target-hidden window across verify rounds

Issue #1549. Measured on M5 Max (Qwen3-8B bf16 + DSpine-b16): 3.6-8.3x vs AR
at T=0; see fusion-research docs/logs/dspine-m5max-bench-1112.md.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from .bridge import build_serial_tables
from .config import DraftConfig
from .model import DSpineDraftModel, load_draft
from .propose import RefinementTail
from .target import HeadAccess, _find_embed

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_SIZE = 16


class DSpineDrafter:
    """High-level DSpine drafter for one target model instance."""

    def __init__(
        self,
        model_path: str,
        target_model: Any | None = None,
        target_layer_ids: list[int] | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ):
        self.model_path = model_path
        self.target_model = target_model
        self.block_size = int(block_size)
        self.config: DraftConfig | None = None
        self._draft: DSpineDraftModel | None = None
        self._tail: RefinementTail | None = None
        self._head: HeadAccess | None = None
        self._embed_tokens = None
        self._target_layer_ids = list(target_layer_ids) if target_layer_ids else None
        self._caches: list[KVCache] | None = None
        self._loaded = False
        logger.info(
            "DSpineDrafter: path=%s block_size=%d target_layers=%s",
            model_path,
            self.block_size,
            target_layer_ids,
        )

    def load(self, target_model: Any | None = None) -> None:
        if self._loaded:
            return
        if target_model is not None:
            self.target_model = target_model
        if self.target_model is None:
            raise ValueError("DSpineDrafter.load requires the target model")

        root = Path(self.model_path)
        # Production loader: strict name/shape checks + bridge prepare()
        # (packed_weight/codebook lazy init lives in prepare(); building
        # serial tables before it raised "code-bridge weights were not
        # prepared").
        draft, cfg = load_draft(root)
        if self._target_layer_ids:
            cfg.target_layer_ids = tuple(self._target_layer_ids)
        draft.bind(self.target_model)
        draft.set_block_size(self.block_size)

        self._embed_tokens = _find_embed(self.target_model)
        self._head = HeadAccess(_find_lm_head(self.target_model))
        bridge = draft.selected_prefix_bridge
        t0 = time.perf_counter()
        self._tg, self._tsv = build_serial_tables(
            head_rows=self._head.rows_range,
            vocab_size=int(cfg.vocab_size),
            norm_w=draft.norm.weight,
            write_w=bridge.feature_write_weights[-1],
            pv_w=bridge.packed_weight[bridge.rank :],
            codebook=bridge.codebook,
        )
        self._tail = RefinementTail(draft, self._head, self._tg, self._tsv)
        self.config = cfg
        self._draft = draft
        self._loaded = True
        logger.info(
            "DSpineDrafter: loaded (%d params), layers=%s vocab=%d, "
            "serial tables in %.1fs",
            draft_model_num_parameters(draft),
            cfg.target_layer_ids,
            cfg.vocab_size,
            time.perf_counter() - t0,
        )

    @property
    def draft(self) -> DSpineDraftModel:
        if self._draft is None:
            self.load()
        return self._draft

    @property
    def target_layer_ids(self) -> list[int]:
        cfg = self.draft.config
        return list(cfg.target_layer_ids)

    def reset_cache(self) -> None:
        # Draft KV caches are per-request state; the RefinementTail's serial
        # tables are bound to the loaded model and must survive resets
        # (nulling _tail left build_tail() returning None →
        # 'NoneType' object has no attribute 'greedy' on the next round).
        self._caches = None

    def make_caches(self) -> list[KVCache]:
        self._caches = self.draft.make_cache()
        return self._caches

    def build_tail(self, target_hidden: mx.array) -> RefinementTail:
        """Serial tables are built at load(); kept for interface parity."""
        return self._tail

    def draft_block(
        self,
        target_hidden: mx.array,
        anchor_token: int,
        caches: list[KVCache] | None = None,
    ) -> list[int]:
        """One drafting round: block forward + greedy refinement tail.

        target_hidden: [1, L, concat_layers*H] hidden window from the target
        (the last verify forward's captured states). Returns draft token ids
        (block_size - 1 candidates after the anchor).
        """
        d = self.draft
        caches = caches if caches is not None else self._caches
        block = mx.array(
            [[anchor_token] + [d.mask_token_id] * (self.block_size - 1)],
            dtype=mx.int32,
        )
        out, stash = d(target_hidden, block, caches)
        rows = out[0, 1:]
        if self._tail is None:
            self.build_tail(target_hidden)
        drafts, _cand = self._tail.greedy(rows, stash, anchor_token)
        return [int(x) for x in drafts.tolist()]

    def draft_block_sampled(
        self,
        target_hidden: mx.array,
        anchor_token: int,
        caches: list[KVCache] | None = None,
        *,
        inv_temp: float,
        top_k: int,
        top_p: float,
        min_p: float,
        uniforms: mx.array,
    ) -> tuple[list[int], mx.array, mx.array]:
        """One drafting round with sampled refinement tail (T>0 path).

        Same block forward as draft_block, but the tail samples each row's
        chain under the request's filters (serial_sample_chain). Returns
        (draft token ids, q rows [n, vocab] filtered proposal probs,
        candidates [n, K]). The q rows must be carried to the rejection
        verify unchanged — they are the distribution the drafts were drawn
        from, and recomputing them would break the sampler identity.
        """
        d = self.draft
        caches = caches if caches is not None else self._caches
        block = mx.array(
            [[anchor_token] + [d.mask_token_id] * (self.block_size - 1)],
            dtype=mx.int32,
        )
        out, stash = d(target_hidden, block, caches)
        rows = out[0, 1:]
        if self._tail is None:
            self.build_tail(target_hidden)
        tokens, q_rows, _cand = self._tail.sampled(
            rows,
            stash,
            anchor_token,
            inv_temp=inv_temp,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            uniforms=uniforms,
        )
        return [int(x) for x in tokens.tolist()], q_rows, _cand


def _find_lm_head(model: Any):
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
        return model.language_model.lm_head
    if hasattr(model, "model") and hasattr(model.model, "lm_head"):
        return model.model.lm_head
    raise AttributeError(f"Cannot find lm_head in {type(model).__name__}")


def draft_model_num_parameters(draft: DSpineDraftModel) -> int:
    from mlx.utils import tree_flatten

    return sum(p.size for _, p in tree_flatten(draft.parameters()))
