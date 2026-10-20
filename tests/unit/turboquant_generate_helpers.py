# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import logging

from fusion_mlx.request import Request, SamplingParams
from fusion_mlx.scheduler import Scheduler, SchedulerConfig

logger = logging.getLogger(__name__)

_MAX_STEPS = 200000
_STORE_DRAIN_TIMEOUT_S = 300.0


def _make_scheduler(
    model,
    tokenizer,
    *,
    ssd_cache_dir=None,
    block_size=256,
    turboquant_bits=None,
    max_num_seqs=8,
):
    config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        prefill_batch_size=max_num_seqs,
        completion_batch_size=max_num_seqs,
        paged_cache_block_size=block_size,
        paged_ssd_cache_dir=ssd_cache_dir,
    )
    scheduler = Scheduler(model=model, tokenizer=tokenizer, config=config)
    if turboquant_bits is not None:
        scheduler._turboquant_kv_bits = float(turboquant_bits)
        scheduler._turboquant_skip_last = True
        from fusion_mlx.patches.turboquant_attention import (
            apply_turboquant_attention_patch,
        )

        applied = apply_turboquant_attention_patch()
        logger.info("turboquant attention patch applied=%s", applied)
    logger.info(
        "tq helper scheduler: ssd_dir=%s block_size=%s tq_bits=%s seqs=%s",
        ssd_cache_dir,
        block_size,
        turboquant_bits,
        max_num_seqs,
    )
    return scheduler


def _make_request(request_id, token_ids, max_tokens):
    request = Request(
        request_id=request_id,
        prompt="",
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )
    request.prompt_token_ids = list(token_ids)
    request.num_prompt_tokens = len(token_ids)
    return request


def _drain_stores(scheduler):
    futures = list(getattr(scheduler, "_inflight_store_futures", {}).values())
    if not futures:
        return
    logger.info("waiting for %d store_cache futures", len(futures))
    done, pending = concurrent.futures.wait(futures, timeout=_STORE_DRAIN_TIMEOUT_S)
    if pending:
        raise RuntimeError(
            f"{len(pending)} store_cache futures still pending after "
            f"{_STORE_DRAIN_TIMEOUT_S}s"
        )
    for future in done:
        future.result()


def _drive(scheduler, requests):
    for request in requests:
        scheduler.add_request(request)
    finished = set()
    steps = 0
    while len(finished) < len(requests):
        output = scheduler.step()
        finished.update(output.finished_request_ids or ())
        steps += 1
        if steps > _MAX_STEPS:
            raise RuntimeError(
                f"scheduler did not finish {len(requests)} requests in {_MAX_STEPS} steps"
            )
    _drain_stores(scheduler)


def _shutdown(scheduler):
    _drain_stores(scheduler)
    manager = getattr(scheduler, "paged_ssd_cache_manager", None)
    if manager is not None and hasattr(manager, "close"):
        manager.close()
        logger.info("closed paged SSD cache manager")


def _generate_tokens(
    model,
    tokenizer,
    token_ids,
    *,
    max_tokens=16,
    ssd_cache_dir=None,
    block_size=256,
    turboquant_bits=None,
):
    scheduler = _make_scheduler(
        model,
        tokenizer,
        ssd_cache_dir=ssd_cache_dir,
        block_size=block_size,
        turboquant_bits=turboquant_bits,
    )
    request = _make_request("tq-single", token_ids, max_tokens)
    _drive(scheduler, [request])
    cached = int(getattr(request, "cached_tokens", 0) or 0)
    _shutdown(scheduler)
    logger.info(
        "tq helper single: output=%d cached=%d", len(request.output_token_ids), cached
    )
    return list(request.output_token_ids), cached


def _generate_batch(
    model,
    tokenizer,
    prompts,
    *,
    mode="concurrent",
    max_tokens=16,
    ssd_cache_dir=None,
    block_size=256,
    turboquant_bits=None,
):
    if mode not in ("concurrent", "serial"):
        raise ValueError(f"unsupported mode: {mode}")
    scheduler = _make_scheduler(
        model,
        tokenizer,
        ssd_cache_dir=ssd_cache_dir,
        block_size=block_size,
        turboquant_bits=turboquant_bits,
        max_num_seqs=max(len(prompts), 1),
    )
    requests = [
        _make_request(f"batch-{i}", prompt, max_tokens)
        for i, prompt in enumerate(prompts)
    ]
    if mode == "concurrent":
        _drive(scheduler, requests)
    else:
        for request in requests:
            _drive(scheduler, [request])
    _shutdown(scheduler)
    results = [
        (
            request.request_id,
            list(request.output_token_ids),
            int(getattr(request, "cached_tokens", 0) or 0),
        )
        for request in requests
    ]
    logger.info(
        "tq helper batch: mode=%s n=%d cached=%s",
        mode,
        len(results),
        [r[2] for r in results],
    )
    return results
