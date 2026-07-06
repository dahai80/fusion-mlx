# SPDX-License-Identifier: Apache-2.0
import logging
import time
from collections.abc import Callable, Generator
from functools import partial
from typing import Any

import mlx.core as mx

from .accept_counter import get_global_counter
from .cache_patch import patch_arrays_cache_rollback_state
from .draft_k_controller_v2 import DepthController, get_or_create_controller

patch_arrays_cache_rollback_state()

logger = logging.getLogger(__name__)

_CACHE_CLEAR_INTERVAL = 256


def _apply_xtc_with_shared_draw(
    logits: mx.array,
    xtc_probability: float,
    xtc_threshold: float,
    xtc_special_tokens: list[int],
    p_draw: mx.array | None,
) -> mx.array:
    from mlx_lm.sample_utils import apply_xtc

    if p_draw is None:
        return apply_xtc(logits, xtc_probability, xtc_threshold, xtc_special_tokens)

    if not (0 <= xtc_threshold <= 0.5):
        raise ValueError(f"xtc_threshold must be in [0, 0.5]; got {xtc_threshold}")
    probs = mx.softmax(logits, axis=-1)
    mask = probs > xtc_threshold
    n_above = mask.sum(axis=-1, keepdims=True)
    mask = mx.where(n_above > 1, mask, mx.zeros_like(mask))
    if xtc_special_tokens:
        mask[..., xtc_special_tokens] = False
    return mx.where(
        p_draw > xtc_probability,
        logits,
        mx.where(mask, -mx.inf, logits),
    )


def _make_sampler_chain(
    top_p: float = 0.0,
    top_k: int = 0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    xtc_special_tokens: list[int] | None = None,
) -> tuple[list[Callable[[mx.array], mx.array]], list | None]:
    from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

    xtc_special_tokens = xtc_special_tokens or []
    xtc_cell: list | None = [None] if xtc_probability > 0.0 else None
    chain: list[Callable[[mx.array], mx.array]] = []
    if 0 < top_p < 1.0:
        chain.append(lambda x: apply_top_p(x, top_p))
    if min_p != 0.0:
        chain.append(lambda x: apply_min_p(x, min_p, min_tokens_to_keep))
    if xtc_probability > 0.0:
        def _xtc(x, _cell=xtc_cell):
            return _apply_xtc_with_shared_draw(
                x,
                xtc_probability,
                xtc_threshold,
                xtc_special_tokens,
                _cell[0],
            )
        chain.append(_xtc)
    if top_k > 0:
        chain.append(lambda x: apply_top_k(x, top_k))
    return chain, xtc_cell


def mtp_generate_step(
    prompt: mx.array,
    model: Any,
    *,
    max_tokens: int = 256,
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] | None = None,
    prompt_cache: Any | None = None,
    prefill_step_size: int = 2048,
    kv_bits: int | None = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    input_embeddings: mx.array | None = None,
    temp: float = 0.0,
    top_p: float = 0.0,
    top_k: int = 0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    xtc_special_tokens: list[int] | None = None,
    accept_counter=None,
    model_id: str | None = None,
    max_k: int = 1,
    disable_auto_k: bool = False,
    stop_tokens: set[int] | None = None,
) -> Generator[tuple[int, mx.array, bool], None, None]:
    import inspect as _inspect

    from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache
    from mlx_lm.models import cache as _cache_module
    from mlx_lm.sample_utils import categorical_sampling

    xtc_special_tokens = xtc_special_tokens or []
    if accept_counter is None:
        accept_counter = get_global_counter()

    try:
        _mtp_supports_hidden = (
            "return_hidden" in _inspect.signature(model.mtp_forward).parameters
        )
    except (TypeError, ValueError):
        _mtp_supports_hidden = False

    y = prompt.astype(mx.uint32)
    prev_tokens: mx.array | None = None

    if prompt_cache is None:
        model_cache = _cache_module.make_prompt_cache(model)
        mtp_cache = model.make_mtp_cache()
    else:
        n_main = len(model.layers)
        model_cache = prompt_cache[:n_main]
        mtp_cache = prompt_cache[n_main:] or model.make_mtp_cache()

    _is_greedy = temp == 0

    _filter_chain, _xtc_cell = (
        _make_sampler_chain(
            top_p,
            top_k,
            min_p,
            min_tokens_to_keep,
            xtc_probability,
            xtc_threshold,
            xtc_special_tokens,
        )
        if not _is_greedy
        else ([], None)
    )

    quantize_cache_fn = partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    def _process_and_sample(tokens, logits, xtc_draw=None):
        if logits_processors:
            logits = logits[None]
            for processor in logits_processors:
                logits = processor(tokens, logits)
            logits = logits.squeeze(0)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if _filter_chain:
            if _xtc_cell is not None:
                _xtc_cell[0] = xtc_draw
            masked = logprobs
            for f in _filter_chain:
                masked = f(masked)
            token = categorical_sampling(masked, temp)
            scaled = masked / temp
            lp_accept = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)
        elif _is_greedy:
            token = mx.argmax(logprobs, axis=-1)
            lp_accept = logprobs
        else:
            token = categorical_sampling(logprobs, temp)
            scaled = logprobs / temp
            lp_accept = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)
        return token, logprobs, lp_accept

    def _clear_rollback():
        for c in model_cache:
            if hasattr(c, "rollback_state"):
                c.rollback_state = None

    def _rollback_draft(n_to_drop: int = 1):
        for c in model_cache:
            if hasattr(c, "rollback_state") and c.rollback_state is not None:
                if n_to_drop != 1:
                    raise AssertionError(
                        f"_rollback_draft(n_to_drop={n_to_drop}) on SSM "
                        "cache: only single-token rollback is supported. "
                        "Chain-of-K on SSM-hybrid targets is not wired "
                        "yet — the generator should have clamped max_k=1."
                    )
                conv_snap, ssm_snap = c.rollback_state
                c[0] = conv_snap
                c[1] = ssm_snap
                c.rollback_state = None
            elif c.is_trimmable():
                c.trim(n_to_drop)

    def _step_backbone(yy, prev, n_predict=1, n_confirmed=0, xtc_draw=None):
        with mx.stream(generation_stream):
            logits, hidden = model(
                yy[None],
                cache=model_cache,
                return_hidden=True,
                n_confirmed=n_confirmed,
            )
            logits = logits[:, -n_predict:, :]
            quantize_cache_fn(model_cache)
            toks: list = []
            lps: list = []
            accept_lps: list = []
            for i in range(n_predict):
                if logits_processors:
                    prev = (
                        mx.concatenate([prev, yy[i : i + 1]])
                        if prev is not None
                        else yy[i : i + 1]
                    )
                draw = xtc_draw if i == 0 else None
                tok, lp, alp = _process_and_sample(
                    prev, logits[:, i, :].squeeze(0), draw
                )
                toks.append(tok)
                lps.append(lp)
                accept_lps.append(alp)
            return (
                mx.stack(toks),
                mx.stack(lps),
                mx.stack(accept_lps),
                hidden,
                prev,
            )

    def _step_mtp(hidden_last, main_tok, prev, *, cache_commit=None, want_hidden=False):
        if cache_commit is not None:
            align_h, align_tok = cache_commit
            hidden_last = mx.concatenate([align_h, hidden_last], axis=1)
            next_ids = mx.concatenate(
                [align_tok.reshape(1, 1), main_tok.reshape(1, 1)], axis=1
            )
        else:
            next_ids = main_tok.reshape(1, 1)
        drafter_hidden_last = None
        with mx.stream(generation_stream):
            if want_hidden:
                mtp_logits, mtp_hidden = model.mtp_forward(
                    hidden_last, next_ids, mtp_cache, return_hidden=True
                )
                drafter_hidden_last = mtp_hidden[:, -1:, :]
            else:
                mtp_logits = model.mtp_forward(hidden_last, next_ids, mtp_cache)
            quantize_cache_fn(mtp_cache)
            mtp_logits = mtp_logits[:, -1, :].squeeze(0)
            if logits_processors:
                tokens_for_proc = (
                    mx.concatenate([prev, main_tok.reshape(-1)])
                    if prev is not None
                    else main_tok.reshape(-1)
                )
            else:
                tokens_for_proc = prev
            xtc_draw = mx.random.uniform() if _xtc_cell is not None else None
            draft_tok, draft_lp, draft_accept_lp = _process_and_sample(
                tokens_for_proc, mtp_logits, xtc_draw
            )
        return draft_tok, draft_lp, draft_accept_lp, xtc_draw, drafter_hidden_last

    def _step_mtp_chain(hidden_last, main_tok, prev, K, *, cache_commit=None):
        draft_toks: list = []
        draft_lps: list = []
        draft_accept_lps: list = []
        xtc_draws: list = []
        prev_tok = main_tok
        cur_hidden = hidden_last
        cur_commit = cache_commit
        for _k in range(K):
            d_tok, d_lp, d_alp, d_xtc, d_hidden = _step_mtp(
                cur_hidden,
                prev_tok,
                prev,
                cache_commit=cur_commit,
                want_hidden=_mtp_supports_hidden and K >= 2,
            )
            mx.eval(d_tok)
            draft_toks.append(d_tok)
            draft_lps.append(d_lp)
            draft_accept_lps.append(d_alp)
            xtc_draws.append(d_xtc)
            prev_tok = d_tok
            if d_hidden is not None:
                mx.eval(d_hidden)
                cur_hidden = d_hidden
            cur_commit = None
        return draft_toks, draft_lps, draft_accept_lps, xtc_draws

    def _prefill(yy, embeddings):
        total = len(embeddings) if embeddings is not None else yy.size
        while total > 1:
            n = min(prefill_step_size, total - 1)
            if embeddings is not None:
                _, hidden = model(
                    yy[:n][None],
                    cache=model_cache,
                    return_hidden=True,
                    input_embeddings=embeddings[:n][None],
                )
                embeddings = embeddings[n:]
            else:
                _, hidden = model(yy[:n][None], cache=model_cache, return_hidden=True)
            model.mtp_forward(hidden, yy[1 : n + 1][None], mtp_cache)
            quantize_cache_fn(mtp_cache)
            quantize_cache_fn(model_cache)
            mx.eval([c.state for c in model_cache + mtp_cache if hasattr(c, "state")])
            yy = yy[n:]
            total -= n
            mx.clear_cache()
        return yy

    with mx.stream(generation_stream):
        y = _prefill(y, input_embeddings)

    ntoks = 0
    last_cache_block = 0
    pending_drafts: list | None = None

    _has_ssm_cache = any(hasattr(c, "rollback_state") for c in model_cache)
    if not disable_auto_k:
        _max_k_hw = 1 if _has_ssm_cache else max(0, max_k)
        if _has_ssm_cache and max_k > 1:
            logger.info(
                "[MTP-chain-of-K] SSM cache detected in model_cache — "
                "clamping max_k from %d to 1 (chain-of-K on SSM-hybrid "
                "targets needs per-position snapshots not yet wired). "
                "Set --mtp-max-k=1 to silence this log.",
                max_k,
            )
        max_k_effective = _max_k_hw
        _controller: DepthController | None = get_or_create_controller(
            model_id or "__default__", max_k=max_k_effective
        )
    else:
        max_k_effective = 1
        _controller = None

    next_k = _controller.pick_k() if _controller is not None else 1

    def _record_round(k_used: int, round_wall_ms: float, accepts: list[bool]) -> None:
        if _controller is None:
            return
        _controller.record(k_used, round_wall_ms, accepts)

    while ntoks < max_tokens:
        round_start_perf = time.perf_counter()
        if pending_drafts is None:
            toks, lps, accept_lps, hidden, prev_tokens = _step_backbone(
                y, prev_tokens, n_predict=1
            )
            mx.eval(toks)
            main_tok, main_lp = toks[0], lps[0]
            round_wall_ms = (time.perf_counter() - round_start_perf) * 1000.0
            _record_round(0, round_wall_ms, [])

            ntoks += 1
            yield main_tok.item(), main_lp, False
            if ntoks >= max_tokens:
                return

            next_k = _controller.pick_k() if _controller is not None else 1

            hidden_at_main = hidden[:, -1:, :]
            if next_k >= 1:
                d_toks, d_lps, d_alps, d_xtcs = _step_mtp_chain(
                    hidden_at_main, main_tok, prev_tokens, next_k
                )
                pending_drafts = list(zip(d_toks, d_lps, d_alps, d_xtcs))
            else:
                pending_drafts = None
            y = mx.array([main_tok.item()], mx.uint32)
        else:
            k_len = len(pending_drafts)
            draft_toks_arr = [rec[0] for rec in pending_drafts]
            draft_lps_arr = [rec[1] for rec in pending_drafts]
            draft_alps_arr = [rec[2] for rec in pending_drafts]
            first_xtc_draw = pending_drafts[0][3]

            drafts_arr = (
                mx.stack([d.reshape(-1) for d in draft_toks_arr])
                .reshape(-1)
                .astype(mx.uint32)
            )
            y_with_drafts = mx.concatenate([y, drafts_arr])

            toks, lps, accept_lps, hidden, prev_tokens = _step_backbone(
                y_with_drafts,
                prev_tokens,
                n_predict=k_len + 1,
                n_confirmed=k_len,
                xtc_draw=first_xtc_draw,
            )

            u = mx.random.uniform()
            drafts_i32 = drafts_arr.astype(mx.int32)

            if _is_greedy:
                accept_mask_arr = toks[:k_len].astype(mx.int32) == drafts_i32
                residual_toks_arr = toks[:k_len]
                bonus_tok_arr = toks[k_len]
            else:
                v_alps = accept_lps[:k_len]
                d_alps_stack = mx.stack(draft_alps_arr)
                idx = drafts_i32.reshape(-1, 1)
                v_at = mx.take_along_axis(v_alps, idx, axis=1).squeeze(-1)
                d_at = mx.take_along_axis(d_alps_stack, idx, axis=1).squeeze(-1)
                log_accept = v_at - d_at
                accept_mask_arr = (log_accept >= 0) | (u < mx.exp(log_accept))

                p_target = mx.exp(v_alps)
                p_draft = mx.exp(d_alps_stack)
                residual = mx.maximum(p_target - p_draft, 0.0)
                z = residual.sum(axis=-1, keepdims=True)
                dist = mx.where(z > 0, residual, p_target)
                residual_toks_arr = mx.random.categorical(mx.log(dist))
                bonus_tok_arr = toks[k_len]

            mx.eval(toks, accept_mask_arr, residual_toks_arr, bonus_tok_arr, u)

            accept_flags = accept_mask_arr.tolist()
            residual_ids = residual_toks_arr.tolist()
            bonus_id = int(bonus_tok_arr.item())
            draft_ids = drafts_arr.tolist()

            for _ in range(k_len):
                accept_counter.record_attempt()

            accepts: list[bool] = []
            accepted_count = 0
            for i in range(k_len):
                ok = bool(accept_flags[i])
                accepts.append(ok)
                if ok:
                    accepted_count += 1
                else:
                    break

            eos_cut = False
            accepts_for_record = accepts
            if stop_tokens:
                for j in range(accepted_count):
                    if int(draft_ids[j]) in stop_tokens:
                        eos_cut = True
                        accepts_for_record = accepts[: j + 1]
                        accepted_count = j + 1
                        break

            round_wall_ms = (time.perf_counter() - round_start_perf) * 1000.0
            _record_round(k_len, round_wall_ms, accepts_for_record)

            for i in range(accepted_count):
                accept_counter.record_accept(tokens_saved=1)
                ntoks += 1
                yield int(draft_ids[i]), draft_lps_arr[i], True
                if ntoks >= max_tokens:
                    return

            if eos_cut:
                return

            if accepted_count == k_len:
                _clear_rollback()
                ntoks += 1
                yield bonus_id, lps[k_len], False
                if ntoks >= max_tokens:
                    return
                last_committed_tok_id = bonus_id
                last_committed_hidden = hidden[:, k_len : k_len + 1, :]
                y = mx.array([bonus_id], mx.uint32)
            else:
                n_to_drop = k_len - accepted_count
                _rollback_draft(n_to_drop)
                accept_counter.record_reject()
                if logits_processors and prev_tokens is not None:
                    prev_tokens = prev_tokens[:-n_to_drop]

                for mc in mtp_cache:
                    if mc.is_trimmable():
                        mc.trim(n_to_drop)

                verify_tok_id = int(residual_ids[accepted_count])

                ntoks += 1
                yield verify_tok_id, lps[accepted_count], False
                if ntoks >= max_tokens:
                    return
                last_committed_tok_id = verify_tok_id
                last_committed_hidden = hidden[
                    :, accepted_count : accepted_count + 1, :
                ]
                y = mx.array([verify_tok_id], mx.uint32)

            next_k = _controller.pick_k() if _controller is not None else 1
            if next_k >= 1:
                if accepted_count == k_len:
                    align_h = hidden[:, accepted_count - 1 : accepted_count, :]
                    align_tok = draft_toks_arr[accepted_count - 1]
                    cache_commit = (align_h, align_tok)
                else:
                    cache_commit = None
                last_committed_tok = mx.array([last_committed_tok_id], mx.uint32)
                d_toks, d_lps, d_alps, d_xtcs = _step_mtp_chain(
                    last_committed_hidden,
                    last_committed_tok,
                    prev_tokens,
                    next_k,
                    cache_commit=cache_commit,
                )
                pending_drafts = list(zip(d_toks, d_lps, d_alps, d_xtcs))
            else:
                pending_drafts = None

        block = ntoks // _CACHE_CLEAR_INTERVAL
        if block > last_cache_block:
            mx.clear_cache()
            last_cache_block = block
