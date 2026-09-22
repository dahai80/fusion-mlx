from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


class FusionPagedKVPool:
    def __init__(
        self,
        block_size: int,
        num_blocks: int,
        n_kv_heads: int,
        head_dim: int,
        k_head_dim: int | None = None,
        v_head_dim: int | None = None,
        dtype: Any = mx.float32,
    ):
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.n_kv_heads = n_kv_heads
        self.k_head_dim = k_head_dim or head_dim
        self.v_head_dim = v_head_dim or head_dim
        self.dtype = dtype
        self.keys_pool = mx.zeros(
            (num_blocks, 1, n_kv_heads, block_size, self.k_head_dim), dtype=dtype
        )
        self.values_pool = mx.zeros(
            (num_blocks, 1, n_kv_heads, block_size, self.v_head_dim), dtype=dtype
        )
        self.free_list: deque[int] = deque(range(num_blocks - 1, -1, -1))
        self.in_use: dict[int, str] = {}
        # CoW pool refcount: phys block -> share count. Always-on (cheap dict
        # ops); non-CoW path never calls share_block so every block stays
        # refcount=1 and free_block behaves identically to the prior immediate
        # free. CoWPagedRequestCache raises refcount>1 via share_block +
        # ensure_writable copies on shared write.
        self._refcount: dict[int, int] = {}
        self._owners: dict[int, set[str]] = {}
        self._step: int = 0
        self._last_access: dict[str, int] = {}
        self._active_ids: set[str] | None = None
        self._evict_cb = None
        # D1 (audit): the scheduler calls set_active_ids / set_evict_callback
        # from the asyncio event-loop thread while request executor threads
        # call alloc_block / free_request. Without a lock, concurrent alloc
        # can pop the same deque element and the LRU victim selection iterates
        # in_use while another thread mutates it -> dict-changed-size RuntimeError
        # or wrong victim (silent KV corruption). RLock: alloc_block calls
        # free_request (reentrant) during eviction.
        self._lock = threading.RLock()
        logger.info(
            "paged_kv pool init: cap=%d block_size=%d n_kv=%d head_dim=%d/%d",
            num_blocks,
            block_size,
            n_kv_heads,
            self.k_head_dim,
            self.v_head_dim,
        )

    def set_active_ids(self, ids: set[str] | None) -> None:
        with self._lock:
            self._active_ids = set(ids) if ids else None
        logger.debug("paged_kv pool set_active_ids=%s", self._active_ids or set())

    def touch_active(self) -> None:
        with self._lock:
            if not self._active_ids:
                return
            self._step += 1
            for rid in self._active_ids:
                self._last_access[rid] = self._step
            n = len(self._active_ids)
            step = self._step
        logger.debug("paged_kv pool touch_active n=%d step=%d", n, step)

    def set_evict_callback(self, cb) -> None:
        with self._lock:
            self._evict_cb = cb

    def touch(self, request_id: str) -> None:
        with self._lock:
            self._step += 1
            self._last_access[request_id] = self._step
            step = self._step
        logger.debug("paged_kv pool touch request=%s step=%d", request_id, step)

    def alloc_block(self, request_id: str, *, active_ids: set | None = None) -> int:
        with self._lock:
            self._step += 1
            self._last_access[request_id] = self._step
            if active_ids is None:
                active_ids = self._active_ids if self._active_ids else {request_id}
            if not self.free_list:
                owners = set(self.in_use.values())
                evictable = owners - active_ids
                if evictable:
                    victim = min(
                        evictable,
                        key=lambda rid: self._last_access.get(rid, 0),
                    )
                    self._free_request_locked(victim)
                    if self._evict_cb is not None:
                        try:
                            self._evict_cb(victim)
                        except Exception as e:
                            logger.warning(
                                "paged_kv pool evict_cb failed for %s: %s",
                                victim,
                                e,
                            )
                    logger.warning(
                        "paged_kv LRU evicting idle request=%s available=%d",
                        victim,
                        len(self.free_list),
                    )
                else:
                    logger.error(
                        "paged_kv pool exhausted for request=%s (no evictable idle)",
                        request_id,
                    )
                    raise RuntimeError(
                        f"paged_kv pool exhausted (cap={self.num_blocks}); "
                        f"reject request or raise pool_num_blocks"
                    )
            pb = self.free_list.pop()
            self.in_use[pb] = request_id
            self._refcount[pb] = 1
            self._owners[pb] = {request_id}
            avail = len(self.free_list)
        logger.debug(
            "paged_kv pool alloc block=%d request=%s available=%d",
            pb,
            request_id,
            avail,
        )
        return pb

    def share_block(self, request_id: str, phys: int) -> None:
        """CoW: adopt an already-allocated physical block for a second owner.
        Increments refcount and registers request_id as a co-owner. The block
        is NOT copied — both owners read the same GPU slab until one writes
        (ensure_writable copies on write). Called by CoWPagedRequestCache on a
        prefix-page donation hit.
        """
        with self._lock:
            rc = self._refcount.get(phys, 0)
            if rc <= 0:
                raise RuntimeError(
                    f"paged_kv pool share_block: phys {phys} not allocated (rc={rc}) "
                    f"— refusing to share a freed/foreign block (CoW D5)"
                )
            self._refcount[phys] = rc + 1
            owners = self._owners.setdefault(phys, set())
            owners.add(request_id)
            new_rc = self._refcount[phys]
        logger.info(
            "paged_kv pool share_block phys=%d request=%s refcount=%d",
            phys,
            request_id,
            new_rc,
        )

    def ensure_writable(self, request_id: str, phys: int) -> int:
        """CoW: if phys is shared (refcount>1), copy it to a fresh slab,
        decrement the original refcount, and return the new physical id. If
        sole owner (refcount==1), return phys unchanged. Called by
        CoWPagedRequestCache before writing into a block.
        """
        with self._lock:
            rc = self._refcount.get(phys, 0)
            if rc <= 1:
                return phys
            if not self.free_list:
                owners = set(self.in_use.values())
                evictable = owners - (self._active_ids or {request_id})
                if evictable:
                    victim = min(
                        evictable,
                        key=lambda rid: self._last_access.get(rid, 0),
                    )
                    self._free_request_locked(victim)
                else:
                    raise RuntimeError(
                        "paged_kv pool ensure_writable exhausted (no free slab "
                        "for CoW copy)"
                    )
            new_phys = self.free_list.pop()
            self.keys_pool[new_phys] = self.keys_pool[phys]
            self.values_pool[new_phys] = self.values_pool[phys]
            self.in_use[new_phys] = request_id
            self._refcount[new_phys] = 1
            self._owners[new_phys] = {request_id}
            self._refcount[phys] = rc - 1
            self._owners[phys].discard(request_id)
            if not self._owners[phys]:
                self._owners.pop(phys, None)
        logger.debug(
            "paged_kv pool CoW copy phys=%d (rc=%d) -> %d request=%s",
            phys,
            rc,
            new_phys,
            request_id,
        )
        return new_phys

    def free_request(self, request_id: str) -> int:
        with self._lock:
            return self._free_request_locked(request_id)

    def _free_request_locked(self, request_id: str) -> int:
        # D1: caller MUST hold self._lock (RLock). Returns freed block count.
        # CoW-refcount-aware: a block co-owned by request_id is decremented,
        # not freed, when other owners remain. Only returns to free_list when
        # the last owner releases. Primary in_use tag is dropped only when the
        # allocator (primary owner) is the one freed.
        freed = [pb for pb, rid in self.in_use.items() if rid == request_id]
        shared_touched = 0
        for pb in freed:
            self.in_use.pop(pb, None)
            rc = self._refcount.pop(pb, 1)
            owners = self._owners.pop(pb, set())
            owners.discard(request_id)
            if not owners:
                self.free_list.append(pb)
            else:
                # still co-owned by another request: keep block live, restore
                # bookkeeping under a remaining owner (pick any).
                remaining = next(iter(owners))
                self.in_use[pb] = remaining
                self._refcount[pb] = max(rc - 1, len(owners))
                self._owners[pb] = owners
                shared_touched += 1
        # also release co-ownership on blocks where request_id is a secondary
        # owner (in_use tag held by a different request).
        for pb, owners in list(self._owners.items()):
            if request_id in owners and pb not in freed:
                owners.discard(request_id)
                rc = self._refcount.get(pb, 1)
                if not owners:
                    self._owners.pop(pb, None)
                    self._refcount.pop(pb, None)
                    self.in_use.pop(pb, None)
                    self.free_list.append(pb)
                    freed.append(pb)
                else:
                    self._refcount[pb] = max(rc - 1, len(owners))
                shared_touched += 1
        if freed:
            self._last_access.pop(request_id, None)
        logger.info(
            "paged_kv pool free request=%s blocks_freed=%d shared_decremented=%d "
            "available=%d",
            request_id,
            len(freed),
            shared_touched,
            len(self.free_list),
        )
        return len(freed)

    def free_block(self, pb: int, request_id: str | None = None) -> None:
        # D6 (audit): single-block free used by FusionPagedRequestCache.trim /
        # state.setter so they no longer bypass the lock / refcount path by
        # mutating in_use + free_list directly. CoW-aware: if request_id is
        # given and the block is shared, only decrement that owner's share
        # (block stays live for remaining owners); only return to free_list
        # when the last owner releases.
        with self._lock:
            rc = self._refcount.get(pb, 0)
            if rc <= 0:
                logger.debug("paged_kv pool free_block phys=%d already free (skip)", pb)
                return
            owners = self._owners.get(pb, set())
            if request_id is not None and owners and len(owners) > 1:
                owners.discard(request_id)
                self._refcount[pb] = max(rc - 1, len(owners))
                if request_id == self.in_use.get(pb):
                    self.in_use[pb] = next(iter(owners))
                logger.debug(
                    "paged_kv pool free_block phys=%d decremented rc=%d owners=%d",
                    pb,
                    self._refcount[pb],
                    len(owners),
                )
                return
            self.in_use.pop(pb, None)
            self._refcount.pop(pb, None)
            self._owners.pop(pb, None)
            self.free_list.append(pb)

    def available(self) -> int:
        with self._lock:
            return len(self.free_list)

    def _adapt_dtype(self, dtype: Any) -> None:
        if dtype == self.dtype:
            return
        with self._lock:
            if self.in_use:
                logger.warning(
                    "paged_kv pool cannot adapt dtype %s -> %s with blocks in_use; "
                    "keeping existing storage",
                    self.dtype,
                    dtype,
                )
                return
            logger.info(
                "paged_kv pool adapting dtype %s -> %s (model compute dtype)",
                self.dtype,
                dtype,
            )
            self.dtype = dtype
            self.keys_pool = mx.zeros(self.keys_pool.shape, dtype=dtype)
            self.values_pool = mx.zeros(self.values_pool.shape, dtype=dtype)

    def stats(self) -> dict:
        return {
            "cap": self.num_blocks,
            "available": self.available(),
            "in_use": len(self.in_use),
        }


class MergedPagedCacheView:
    # Batched view over B per-sequence FusionPagedRequestCache handles.
    # mlx_lm _merge_caches calls cache.merge([c[i] for c in caches]) to batch
    # per-request layer caches, then runs the batched forward pass
    # (prefill) and -- after split() -- generation appends via
    # update_and_fetch on the MERGED cache. The prior merge returned a
    # read-only materialized snapshot (_is_merged=True) which crashed on
    # the first generation append ("cannot update a merged cache"). This
    # view delegates update_and_fetch per-sequence to each constituent's
    # own paged blocks, so generation appends land in the right
    # per-sequence slab. filter/split only reindex the constituent list
    # (no slab reorg) -- each sequence keeps its own block_table. CoW
    # (ensure_writable) handles divergence on the underlying blocks.

    def __init__(self, constituents: list | None = None):
        self.constituents: list = list(constituents) if constituents else []
        self._offset_override: int | None = None

    @property
    def pool(self):
        return self.constituents[0].pool if self.constituents else None

    @property
    def offset(self):
        if self._offset_override is not None:
            return self._offset_override
        return max((c.offset for c in self.constituents), default=0)

    @offset.setter
    def offset(self, v):
        self._offset_override = v
        for c in self.constituents:
            c.offset = v

    def update_and_fetch(self, keys, values):
        B = keys.shape[0]
        if len(self.constituents) != B:
            raise RuntimeError(
                f"MergedPagedCacheView.update_and_fetch: batch mismatch "
                f"B={B} constituents={len(self.constituents)}"
            )
        fetched = []
        for i in range(B):
            k_i = keys[i : i + 1]
            v_i = values[i : i + 1]
            fk, fv = self.constituents[i].update_and_fetch(k_i, v_i)
            fetched.append((fk, fv))
        max_len = max(fk.shape[-2] for fk, _ in fetched)
        k_head_dim = fetched[0][0].shape[-1]
        v_head_dim = fetched[0][1].shape[-1]
        n_kv_heads = fetched[0][0].shape[1]
        dt = fetched[0][0].dtype
        out_k = mx.zeros((B, n_kv_heads, max_len, k_head_dim), dtype=dt)
        out_v = mx.zeros((B, n_kv_heads, max_len, v_head_dim), dtype=dt)
        for i, (fk, fv) in enumerate(fetched):
            L = fk.shape[-2]
            out_k[i : i + 1, :, :L, :] = fk
            out_v[i : i + 1, :, :L, :] = fv
        return out_k, out_v

    @property
    def state(self):
        if not self.constituents:
            d = 1
            return mx.zeros((1, 1, 0, d)), mx.zeros((1, 1, 0, d))
        states = [c.state for c in self.constituents]
        max_len = max(s[0].shape[-2] for s in states)
        k_head_dim = states[0][0].shape[-1]
        v_head_dim = states[0][1].shape[-1]
        n_kv_heads = states[0][0].shape[1]
        dt = states[0][0].dtype
        B = len(states)
        out_k = mx.zeros((B, n_kv_heads, max_len, k_head_dim), dtype=dt)
        out_v = mx.zeros((B, n_kv_heads, max_len, v_head_dim), dtype=dt)
        for i, (sk, sv) in enumerate(states):
            L = sk.shape[-2]
            out_k[i : i + 1, :, :L, :] = sk
            out_v[i : i + 1, :, :L, :] = sv
        return out_k, out_v

    @state.setter
    def state(self, v):
        if v is None or not self.constituents:
            return
        keys, values = v
        B = keys.shape[0]
        for i in range(min(B, len(self.constituents))):
            self.constituents[i].state = (
                keys[i : i + 1],
                values[i : i + 1],
            )

    @property
    def meta_state(self):
        return [getattr(c, "meta_state", None) for c in self.constituents]

    @meta_state.setter
    def meta_state(self, v):
        if not isinstance(v, (list, tuple)):
            return
        for i, ms in enumerate(v):
            if i < len(self.constituents):
                c = self.constituents[i]
                if hasattr(c, "meta_state"):
                    try:
                        c.meta_state = ms
                    except Exception:
                        pass

    def is_trimmable(self):
        return True

    def size(self):
        return self.offset

    def trim(self, n):
        for c in self.constituents:
            try:
                c.trim(n)
            except Exception as e:
                logger.debug("MergedPagedCacheView.trim constituent failed: %s", e)
        return n

    def filter(self, keep):
        if not keep and self.constituents:
            self.clear()
            return
        self.constituents = [self.constituents[i] for i in keep]

    def clear(self):
        for c in self.constituents:
            try:
                c.clear()
            except Exception as e:
                logger.debug("MergedPagedCacheView.clear constituent failed: %s", e)
        self.constituents = []
        self._offset_override = None

    def empty(self):
        return all(c.empty() for c in self.constituents) if self.constituents else True

    @property
    def nbytes(self):
        return sum(c.nbytes for c in self.constituents)

    def free_all(self):
        freed = 0
        for c in self.constituents:
            try:
                freed += c.free_all()
            except Exception as e:
                logger.debug("MergedPagedCacheView.free_all constituent failed: %s", e)
        self.constituents = []
        self._offset_override = None
        return freed

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def __deepcopy__(self, memo):
        import copy as _copy

        new = MergedPagedCacheView.__new__(MergedPagedCacheView)
        new.constituents = [_copy.deepcopy(c) for c in self.constituents]
        new._offset_override = self._offset_override
        return new

    def extract(self, idx):
        # mlx_lm BatchGenerator.extract_cache calls c.extract(idx) to pull the
        # idx-th sequence's cache out of the batched cache. For the view this
        # is just the idx-th constituent (already per-sequence).
        return self.constituents[idx]

    def prepare(self, lengths=None, right_padding=None, **kwargs):
        # mlx_lm calls c.prepare(lengths=, right_padding=) when batched
        # prompts have different lengths (padding). Each constituent is
        # per-sequence so it needs no padding prep; the view's
        # update_and_fetch already delegates per-sequence with correct
        # offsets. No-op keeps the batched prefill path from AttributeError.
        pass

    def to_quantized(self, *args, **kwargs):
        raise NotImplementedError(
            "MergedPagedCacheView.to_quantized not supported "
            "(pool paged-KV is non-quantized)"
        )

    def stats(self) -> dict:
        return {
            "merged": True,
            "n_constituents": len(self.constituents),
            "offset": self.offset,
            "constituent_stats": [c.stats() for c in self.constituents],
        }


class FusionPagedRequestCache:
    def __init__(self, pool: FusionPagedKVPool, request_id: str):
        self.pool = pool
        self.request_id = request_id
        self.block_table: list[int] = []
        self.offset: int = 0
        self._n_kv_heads: int = pool.n_kv_heads
        self._k_head_dim: int = pool.k_head_dim
        self._v_head_dim: int = pool.v_head_dim
        self._dtype: Any = pool.dtype
        self._B: int = 1
        self._is_merged: bool = False
        self._merged_keys: mx.array | None = None
        self._merged_values: mx.array | None = None
        self._merged_padding: list[int] = []

    def _logical_to_block(self, logical_pos: int) -> int:
        return logical_pos // self.pool.block_size

    def _pos_in_block(self, logical_pos: int) -> int:
        return logical_pos % self.pool.block_size

    def update_and_fetch(self, keys, values):
        if self._is_merged:
            raise RuntimeError("cannot update a merged FusionPagedRequestCache")
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        dtype = keys.dtype
        if dtype != self.pool.dtype:
            self.pool._adapt_dtype(dtype)
        self._B = B
        self._n_kv_heads = n_kv_heads
        self._k_head_dim = k_head_dim
        self._v_head_dim = v_head_dim
        self._dtype = dtype

        prev = self.offset
        end = prev + num_steps
        first_block = self._logical_to_block(prev)
        last_block = self._logical_to_block(end - 1)
        if num_steps <= 2:
            logger.debug(
                "paged_kv UF req=%s prev=%d end=%d nsteps=%d bt_len=%d",
                self.request_id,
                prev,
                end,
                num_steps,
                len(self.block_table),
            )

        # D3 (audit): track blocks newly allocated in THIS call so a
        # mid-operation alloc failure (pool exhausted after some blocks
        # appended) rolls them back — otherwise block_table holds dangling
        # allocated blocks with a stale offset, and free_all only reclaims
        # them if the request_id is later evicted.
        allocated_this_call: list[int] = []
        try:
            for lb in range(first_block, last_block + 1):
                while len(self.block_table) <= lb:
                    pb = self.pool.alloc_block(self.request_id)
                    self.block_table.append(pb)
                    allocated_this_call.append(pb)
                    logger.debug(
                        "paged_kv request=%s block_table grow lb=%d pb=%d",
                        self.request_id,
                        lb,
                        pb,
                    )

            for lb in range(first_block, last_block + 1):
                block_start_logical = lb * self.pool.block_size
                block_end_logical = block_start_logical + self.pool.block_size
                s_start = max(prev, block_start_logical) - prev
                s_end = min(end, block_end_logical) - prev
                n = s_end - s_start
                if n <= 0:
                    continue
                pb = self.block_table[lb]
                pos_start = self._pos_in_block(max(prev, block_start_logical))
                self.pool.keys_pool[pb, ..., pos_start : pos_start + n, :] = keys[
                    ..., s_start:s_end, :
                ]
                self.pool.values_pool[pb, ..., pos_start : pos_start + n, :] = values[
                    ..., s_start:s_end, :
                ]
        except Exception:
            # D3 (audit): rollback blocks allocated this call that sit at the
            # tail of block_table (purely appended, not overwriting pre-existing
            # entries). Frees them back to the pool so offset stays consistent
            # and no dangling allocated blocks leak.
            while self.block_table and self.block_table[-1] in allocated_this_call:
                pb = self.block_table.pop()
                self.pool.free_block(pb, self.request_id)
            logger.warning(
                "paged_kv update_and_fetch rolled back %d blocks for request=%s "
                "after alloc/write failure (offset unchanged at %d)",
                len(allocated_this_call),
                self.request_id,
                self.offset,
            )
            raise
        finally:
            self._block_table_len_before = len(self.block_table)

        self.offset = end
        fetched = self._fetch_logical(end)
        # Force eval of the written slabs so the fetch reads materialized
        # data, not a lazy view that could be overwritten by a later
        # in-place write to the same pool slab (silent KV corruption).
        try:
            mx.eval(fetched[0], fetched[1])
        except Exception:
            pass
        return fetched

    def _fetch_logical(self, length: int):
        if self._is_merged:
            return self._merged_keys, self._merged_values
        num_full = length // self.pool.block_size
        rem = length % self.pool.block_size
        k_parts = []
        v_parts = []
        for lb in range(num_full):
            pb = self.block_table[lb]
            k_parts.append(self.pool.keys_pool[pb])
            v_parts.append(self.pool.values_pool[pb])
        if rem:
            lb = num_full
            if lb < len(self.block_table):
                pb = self.block_table[lb]
                k_parts.append(self.pool.keys_pool[pb, ..., :rem, :])
                v_parts.append(self.pool.values_pool[pb, ..., :rem, :])
        if not k_parts:
            d = self._k_head_dim if self._k_head_dim else 1
            empty = mx.zeros((1, 1, 0, d), dtype=self._dtype)
            return empty, empty
        all_k = mx.concatenate(k_parts, axis=-2) if len(k_parts) > 1 else k_parts[0]
        all_v = mx.concatenate(v_parts, axis=-2) if len(v_parts) > 1 else v_parts[0]
        return all_k, all_v

    def __deepcopy__(self, memo):
        # mlx_lm BatchGenerator._copy() deepcopies prompt_cache on every
        # split() (prefill->generation transition). The pool is a shared
        # singleton with an unpicklable mlx.core.Dtype, and MUST stay shared.
        #
        # ADOPT (move) the original's physical blocks into the copy instead
        # of alloc+copy. The prior alloc+copy path allocated 336 fresh
        # blocks (4 req x 28 layer x 3 blocks) during a single split,
        # exhausting the 512-block pool and triggering LRU eviction of an
        # ACTIVE concurrent request (pool_N namespace mismatch in
        # set_active_ids left all pool requests evictable). The evict
        # callback invalidate_request then cleared the victim's
        # block_table + offset mid-deepcopy -> the copy received an empty
        # cache -> decode garbage (#955 root cause).
        #
        # Adopt is safe because the scheduler gate (6feae2da) forces every
        # prompt through external prefill -> 1-token insert -> immediate
        # all-split (indices_left empty) -> the original prompt-batch is
        # discarded right after _copy(). The original's block_table is
        # detached (emptied) so it cannot free or corrupt the adopted
        # blocks. No alloc -> no eviction -> no invalidate_request.
        cls = self.__class__
        new = cls.__new__(cls)
        new.pool = self.pool
        new.request_id = self.request_id
        new.offset = self.offset
        new._n_kv_heads = self._n_kv_heads
        new._k_head_dim = self._k_head_dim
        new._v_head_dim = self._v_head_dim
        new._dtype = self._dtype
        new._B = self._B
        new._is_merged = False
        new._merged_keys = None
        new._merged_values = None
        new._merged_padding = []
        new._block_table_len_before = getattr(self, "_block_table_len_before", 0)
        new.block_table = list(self.block_table)
        _moved = len(new.block_table)
        # Detach the original: it is discarded after split, so its
        # block_table must not reference (and thus free) the adopted blocks.
        self.block_table = []
        self.offset = 0
        logger.info(
            "paged_kv __deepcopy__ adopt req=%s moved=%d blocks (no alloc)",
            self.request_id,
            _moved,
        )
        return new

    def extract(self, idx):
        # mlx_lm BatchGenerator.extract_cache calls c.extract(idx). A
        # per-sequence FusionPagedRequestCache is already single-sequence
        # (B=1), so extract(0) returns self; idx!=0 is invalid.
        return self

    def prepare(self, lengths=None, right_padding=None, **kwargs):
        # mlx_lm calls prepare for batched padding. Per-sequence B=1 cache
        # needs no padding prep.
        pass

    def filter(self, keep):
        # mlx_lm BatchGenerator.filter reindexes the batch dim. For B==1 the
        # only keep is [0] (no-op) or [] (caller should clear instead). B>1
        # batched filter would require reorganizing the batch dim across
        # every allocated slab, which this paged-KV batched model does not
        # support — fail visibly rather than silently corrupt.
        if not keep:
            return
        if len(keep) == self._B and list(keep) == list(range(self._B)):
            return
        raise RuntimeError(
            f"FusionPagedRequestCache.filter: batched reindex not supported "
            f"(B={self._B}, keep={keep}); paged-KV pool mode requires B=1 "
            f"per cache (non-batched continuous batching)"
        )

    def clear(self):
        # mlx_lm calls clear() when all sequences leave the batch. Free ONLY
        # this cache's own block_table blocks (instance-scoped via free_block),
        # NOT pool.free_request(request_id) — all 28 layer caches of one
        # request share request_id, so free_request would free every layer's
        # blocks mid-generation and corrupt the live peers (D9: phys collision
        # from stale block_table referencing freed-then-reallocated slabs).
        if self.offset > 0 or self.block_table:
            logger.debug(
                "paged_kv clear req=%s offset=%d bt_len=%d",
                self.request_id,
                self.offset,
                len(self.block_table),
            )
        for pb in self.block_table:
            try:
                self.pool.free_block(pb, self.request_id)
            except Exception as e:
                logger.debug("paged_kv clear free_block phys=%d failed: %s", pb, e)
        self.block_table = []
        self.offset = 0
        self._is_merged = False
        self._merged_keys = None
        self._merged_values = None

    @property
    def state(self):
        if self._is_merged:
            return self._merged_keys, self._merged_values
        return self._fetch_logical(self.offset)

    @state.setter
    def state(self, v):
        if self._is_merged:
            raise RuntimeError("cannot set state on a merged FusionPagedRequestCache")
        keys, values = v
        if keys is None:
            return
        B, n_kv_heads, length, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        self._B = B
        self._n_kv_heads = n_kv_heads
        self._k_head_dim = k_head_dim
        self._v_head_dim = v_head_dim
        self._dtype = keys.dtype
        # D8 (audit): free the old block_table's physical blocks BEFORE
        # resetting, so they return to the pool free-list. The prior code
        # cleared block_table=[] and rebuilt free_list full (standalone) or
        # leaked (pool-shared) — physical block ids were silently lost.
        for pb in self.block_table:
            self.pool.free_block(pb, self.request_id)
        self.block_table = []
        self.offset = 0
        num_blocks_needed = (length + self.pool.block_size - 1) // self.pool.block_size
        for lb in range(num_blocks_needed):
            pb = self.pool.alloc_block(self.request_id)
            self.block_table.append(pb)
        for lb in range(num_blocks_needed):
            block_start_logical = lb * self.pool.block_size
            block_end_logical = block_start_logical + self.pool.block_size
            s_start = max(0, block_start_logical)
            s_end = min(length, block_end_logical)
            n = s_end - s_start
            if n <= 0:
                continue
            pb = self.block_table[lb]
            pos_start = self._pos_in_block(max(0, block_start_logical))
            self.pool.keys_pool[pb, ..., pos_start : pos_start + n, :] = keys[
                ..., s_start:s_end, :
            ]
            self.pool.values_pool[pb, ..., pos_start : pos_start + n, :] = values[
                ..., s_start:s_end, :
            ]
        self.offset = length

    @property
    def meta_state(self):
        return ",".join(
            map(str, (self.offset, self.pool.block_size, self.pool.num_blocks))
        )

    @meta_state.setter
    def meta_state(self, v):
        logger.warning(
            "meta_state setter not supported on shared-pool "
            "FusionPagedRequestCache (request=%s); pool geometry is fixed",
            self.request_id,
        )
        raise NotImplementedError(
            "meta_state setter not supported on shared-pool " "FusionPagedRequestCache"
        )

    def is_trimmable(self):
        return True

    def size(self):
        return self.offset

    def trim(self, n):
        if self._is_merged:
            raise RuntimeError("cannot trim a merged FusionPagedRequestCache")
        n = min(self.offset, n)
        self.offset -= n
        new_num_blocks = (
            self.offset + self.pool.block_size - 1
        ) // self.pool.block_size
        # D6 (audit): route through pool.free_block so the pool's lock +
        # bookkeeping stay consistent. The prior code mutated
        # pool.in_use / pool.free_list directly, bypassing the lock and
        # (if CoW were routed through the pool) the refcount path.
        while len(self.block_table) > new_num_blocks:
            pb = self.block_table.pop()
            self.pool.free_block(pb, self.request_id)
        return n

    def empty(self):
        return self.offset == 0 and not self.block_table

    @property
    def nbytes(self):
        if self._is_merged:
            return self._merged_keys.nbytes + self._merged_values.nbytes
        return len(self.block_table) * (
            self.pool.keys_pool[0].nbytes + self.pool.values_pool[0].nbytes
        )

    def free_all(self) -> int:
        if self._is_merged:
            freed = 0
            self._merged_keys = None
            self._merged_values = None
            self._is_merged = False
            self.offset = 0
            return freed
        # D9: instance-scoped free — free ONLY this cache's block_table blocks.
        # pool.free_request(request_id) would free every layer cache sharing
        # this request_id (all 28 layers of one request), corrupting live peers
        # when free_all is called mid-generation on a single layer.
        freed = 0
        for pb in self.block_table:
            try:
                self.pool.free_block(pb, self.request_id)
                freed += 1
            except Exception as e:
                logger.debug("paged_kv free_all free_block phys=%d failed: %s", pb, e)
        self.block_table = []
        self.offset = 0
        return freed

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    @classmethod
    def merge(cls, caches):
        # Return a writable batched VIEW over the per-sequence caches, NOT a
        # read-only materialized snapshot. mlx_lm runs update_and_fetch on
        # the merged cache during generation (after split); the view
        # delegates per-sequence so appends land in the right slab. The old
        # materialized-snapshot path (_is_merged=True) crashed on the first
        # generation append. filter/split reindex the constituent list.
        if not caches:
            return MergedPagedCacheView([])
        if len(caches) == 1:
            return caches[0]
        view = MergedPagedCacheView(list(caches))
        logger.info(
            "paged_kv merge(view): B=%d offsets=%s",
            len(caches),
            [c.offset for c in caches],
        )
        return view

    def stats(self) -> dict:
        return {
            "request_id": self.request_id,
            "offset": self.offset,
            "blocks_used": len(self.block_table),
            "block_size": self.pool.block_size,
            "pool_available": self.pool.available(),
            "is_merged": self._is_merged,
        }


__all__ = ["FusionPagedKVPool", "FusionPagedRequestCache"]
