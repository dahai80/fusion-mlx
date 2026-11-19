# SPDX-License-Identifier: Apache-2.0
"""Gate A' KV quality battery (2026-11-13).

One-shot close-or-reopen quality arm for radial/polar 4-bit KV coding on
Qwen3-8B bf16. Design doc: fusion-research docs/gateap-polarquant-1113.

Arms: bf16 (baseline) / radial (gateap_radial_coder 4-bit) / turboquant
(in-tree TurboQuantKVCache bits=4 via the vendored mlx_vlm codec, with the
fusion_mlx dense-codec guard #1230 replicated inline so the broken RHT path
cannot be selected).

Reconstruction path only: cache entries are quantized after prefill and
attention reads the DECODED values (paper Python-path semantics). The fused
kernel speed question is out of scope.

Battery per (arm, context): multi-depth needle retrieval (2 sets x 3 depths),
held-out wikitext perplexity (512-token continuation), code completion
prefix continuation (edit similarity, 2 files), structured JSON validity
(2 prompts). Fixed seeds, greedy decoding, fixed corpus windows shared by
all arms (cursor advances identically per process).

Usage:
  python3 benchmarks/gateap_kv_quality_1113.py --arm bf16 --ctx 8192 \
      --model /Users/dahai/.fusion-mlx/models/Qwen--Qwen3-8B \
      --out /tmp/gateap_out/bf16_8k.json
"""

import argparse
import difflib
import gc
import hashlib
import json
import logging
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gateap_radial_coder import RadialPolarCodec, sign_sketch_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("gateap-quality")

SEED = 0
PPL_CONT_TOKENS = 512
CODE_GEN_TOKENS = 128
CODE_GT_TOKENS = 128
JSON_GEN_TOKENS = 160
NEEDLE_MAX_TOKENS = 24
PREFILL_CHUNK = 8192

NEEDLE_SETS = [
    [("ZETA-7", "KX-4492"), ("ORION-3", "QT-8156"), ("VELA-9", "MW-2730")],
    [("ARGUS-5", "FD-6194"), ("LYRA-2", "HB-9057"), ("NONUS-8", "PC-3471")],
]
NEEDLE_DEPTHS = [0.10, 0.50, 0.90]

CODE_FILES = [
    "fusion_mlx/turboquant_kv.py",
    "fusion_mlx/memory_cache.py",
]

# Dense-codec guard replicated from
# fusion_mlx/patches/turboquant_fused_rht_guard.py (#1230): the vendored
# mlx_vlm 0.5.0 RHT codec is broken end to end for power-of-two head dims;
# force use_rht=False so turboquant arm uses the dense rotation path that
# in-tree production uses.
def apply_dense_codec_guard():
    from mlx_vlm.turboquant import _TurboQuantMSECodec

    original_init = _TurboQuantMSECodec.__init__
    if getattr(original_init, "_gateap_dense_guard", False):
        return

    def _dense_guard_init(self, dim, bits, seed):
        original_init(self, dim, bits, seed)
        if getattr(self, "use_rht", False):
            self.use_rht = False

    _dense_guard_init._gateap_dense_guard = True
    _TurboQuantMSECodec.__init__ = _dense_guard_init
    logger.info("TurboQuant dense-codec guard installed (RHT path disabled)")


class _StateShim:
    def __init__(self, keys, values):
        self._state = (keys, values)

    @property
    def state(self):
        return self._state


class CorpusCursor:
    def __init__(self, seed=SEED):
        self.cursor = 0
        self.tokens = None
        # Corpus is cached locally from the 2026-11-13 prep run; offline
        # mode avoids mirror flakiness and keeps windows deterministic.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from datasets import load_dataset

        texts = []
        for split in ("validation", "test"):
            ds = load_dataset(
                "Salesforce/wikitext", "wikitext-103-raw-v1", split=split
            )
            texts.extend(ds["text"])
        self.blob = "\n".join(texts)
        logger.info("corpus chars: %d", len(self.blob))

    def tokenize(self, tokenizer, n_tokens):
        # Tokenize the whole blob once, then serve fixed windows.
        if self.tokens is None:
            self.tokens = tokenizer.encode(self.blob)
            logger.info("corpus tokens: %d", len(self.tokens))
        start = self.cursor
        end = min(start + n_tokens, len(self.tokens))
        self.cursor = end
        return self.tokens[start:end]


def model_revision(model_path):
    h = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json"):
        with open(os.path.join(model_path, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def load_model_and_tokenizer(model_path):
    from mlx_lm.utils import load

    model, tokenizer = load(model_path)
    return model, tokenizer


def make_caches(model):
    from mlx_lm.models.cache import make_prompt_cache

    return make_prompt_cache(model)


def prefill(model, caches, tokens):
    arr = mx.array(np.asarray(tokens, dtype=np.int32))[None]
    for start in range(0, arr.shape[1] - 1, PREFILL_CHUNK):
        end = min(start + PREFILL_CHUNK, arr.shape[1] - 1)
        model(arr[:, start:end], cache=caches)
    logits = model(arr[:, -1:], cache=caches)
    mx.eval(logits)
    return logits


BYTE_EVENTS = []


def quantize_caches(caches, arm, seed, codec_cache):
    stats = {"bytes": 0, "tokens": 0}
    for ci, c in enumerate(caches):
        keys, values = c.state
        if arm == "bf16":
            stats["bytes"] += keys.nbytes + values.nbytes
            stats["tokens"] = int(keys.shape[2])
            continue
        if arm == "radial":
            codec = codec_cache.get(ci)
            if codec is None:
                codec = RadialPolarCodec(keys.shape[-1], seed=seed)
                codec_cache[ci] = codec
            kq = codec.roundtrip(keys, out_dtype=keys.dtype)
            vq = codec.roundtrip(values, out_dtype=values.dtype)
            packed_bytes = 2 * (
                (keys.shape[-1] // 2) + 2
            ) * keys.shape[1] * keys.shape[2] * keys.shape[0]
            stats["bytes"] += packed_bytes
        elif arm == "turboquant":
            tq = _TurboQuantKVCache(bits=4.0, seed=seed)
            tq.update_and_fetch(keys, values)
            dk, dv = tq.dequantize()
            kq = dk.astype(keys.dtype)
            vq = dv.astype(values.dtype)
            stats["bytes"] += tq.nbytes
        else:
            raise ValueError(arm)
        stats["tokens"] = int(keys.shape[2])
        mx.eval(kq, vq)
        c.keys = kq
        c.values = vq
    BYTE_EVENTS.append(stats)
    return stats


def greedy_generate(model, caches, first_logits, max_new, eos_ids):
    logits = first_logits
    out = []
    for _ in range(max_new):
        nxt = mx.argmax(logits[:, -1, :], axis=-1)
        nid = int(nxt.item())
        if nid in eos_ids:
            break
        out.append(nid)
        logits = model(nxt[:, None], cache=caches)
    mx.eval(caches[0].state[0])
    return out


def eos_id_set(tokenizer):
    ids = set()
    v = getattr(tokenizer, "eos_token_id", None)
    if v is None:
        v = getattr(getattr(tokenizer, "tokenizer", None), "eos_token_id", None)
    if isinstance(v, int):
        ids.add(v)
    elif isinstance(v, (list, tuple)):
        ids.update(int(i) for i in v)
    return ids


def insert_needle(tokenizer, doc_tokens, needle, depth):
    sentence = (
        f"The access code for facility {needle[0]} is {needle[1]}. "
    )
    cut = int(len(doc_tokens) * depth)
    head = tokenizer.decode(doc_tokens[:cut])
    tail = tokenizer.decode(doc_tokens[cut:])
    return tokenizer.encode(head + sentence + tail)


def run_needle_set(model, tokenizer, cursor, eos_ids, arm, seed, codec_cache, ctx):
    doc_tokens = cursor.tokenize(tokenizer, ctx)
    results = []
    for set_idx, needles in enumerate(NEEDLE_SETS):
        if set_idx > 0:
            doc_tokens = cursor.tokenize(tokenizer, ctx)
        depth_results = []
        for n_idx, needle in enumerate(needles):
            full = insert_needle(tokenizer, doc_tokens, needle, NEEDLE_DEPTHS[n_idx])
            full = full[: ctx + 64]
            question = (
                f"\n\nQuestion: What is the access code for facility "
                f"{needle[0]}? Reply with the code only.\nAnswer:"
            )
            q_tokens = tokenizer.encode(question)
            caches = make_caches(model)
            prefill(model, caches, full)
            quantize_caches(caches, arm, seed, codec_cache)
            q_logits = model(mx.array(np.asarray(q_tokens, dtype=np.int32))[None],
                             cache=caches)
            out = greedy_generate(model, caches, q_logits, NEEDLE_MAX_TOKENS,
                                  eos_ids)
            text = tokenizer.decode(out)
            hit = needle[1].lower() in text.lower()
            depth_results.append(
                {"facility": needle[0], "depth": NEEDLE_DEPTHS[n_idx],
                 "output": text, "hit": bool(hit)}
            )
            caches = None
            mx.clear_cache()
            gc.collect()
        results.extend(depth_results)
        logger.info("needle set %d: %d/%d", set_idx,
                    sum(1 for r in depth_results if r["hit"]), len(depth_results))
    return results


def run_perplexity(model, tokenizer, cursor, arm, seed, codec_cache, ctx):
    ctx_tokens = cursor.tokenize(tokenizer, ctx)
    cont_tokens = cursor.tokenize(tokenizer, PPL_CONT_TOKENS)
    caches = make_caches(model)
    prefill(model, caches, ctx_tokens)
    quantize_caches(caches, arm, seed, codec_cache)
    cont = mx.array(np.asarray(cont_tokens, dtype=np.int32))[None]
    logits = model(cont, cache=caches)
    z = logits[:, :-1, :]
    zmax = mx.max(z, axis=-1, keepdims=True)
    lse = zmax + mx.log(mx.sum(mx.exp(z - zmax), axis=-1, keepdims=True))
    log_probs = z - lse
    targets = cont[:, 1:]
    nll = -mx.take_along_axis(
        log_probs, targets[:, :, None], axis=-1
    ).squeeze(-1)
    mean_nll = mx.mean(nll).item()
    caches = None
    mx.clear_cache()
    gc.collect()
    logger.info("perplexity: nll=%.4f ppl=%.2f", mean_nll, float(np.exp(mean_nll)))
    return {"mean_nll": mean_nll, "ppl": float(np.exp(mean_nll)),
            "cont_tokens": PPL_CONT_TOKENS}


def run_code_task(model, tokenizer, cursor, eos_ids, arm, seed, codec_cache,
                  ctx, repo_root, file_idx):
    path = os.path.join(repo_root, CODE_FILES[file_idx])
    with open(path, "r") as f:
        code = f.read()
    code_tokens = tokenizer.encode(code)
    filler_budget = max(int(ctx * 0.5), 256)
    filler = cursor.tokenize(tokenizer, filler_budget)
    prefix_budget = min(ctx - len(filler), len(code_tokens) - CODE_GT_TOKENS)
    if prefix_budget <= 0:
        raise ValueError("code prefix budget exhausted")
    prefix = code_tokens[:prefix_budget]
    gt = code_tokens[len(prefix): len(prefix) + CODE_GT_TOKENS]
    gt_text = tokenizer.decode(gt)
    instruction = (
        "\n\nContinue this Python code exactly as it continues in the "
        "original file. Output only the raw continuation.\n```python\n"
        + tokenizer.decode(prefix)
    )
    prompt_tokens = tokenizer.encode(instruction)
    caches = make_caches(model)
    logits = prefill(model, caches, prompt_tokens)
    quantize_caches(caches, arm, seed, codec_cache)
    out = greedy_generate(model, caches, logits, CODE_GEN_TOKENS, eos_ids)
    gen_text = tokenizer.decode(out)
    sim = difflib.SequenceMatcher(None, gen_text, gt_text).ratio()
    caches = None
    mx.clear_cache()
    gc.collect()
    logger.info("code %s: edit_sim=%.4f", CODE_FILES[file_idx], sim)
    return {"file": CODE_FILES[file_idx], "edit_sim": sim, "gen": gen_text,
            "gt": gt_text}


def fact_sheet(rng):
    facilities = [f"FAC-{i:02d}" for i in range(1, 9)]
    codes = {}
    for fac in facilities:
        codes[fac] = "%s-%04d" % (
            "".join(rng.choice(list("ABCDEFGHJKLMNPQRSTUVWXYZ"), 2)),
            rng.randint(0, 9999),
        )
    metrics = {
        "throughput": round(float(rng.uniform(80, 200)), 1),
        "latency_ms": round(float(rng.uniform(5, 60)), 2),
        "error_rate": round(float(rng.uniform(0.001, 0.05)), 4),
        "uptime_days": int(rng.randint(10, 400)),
    }
    return facilities, codes, metrics


def run_json_task(model, tokenizer, cursor, eos_ids, arm, seed, codec_cache,
                  ctx, prompt_idx):
    rng = np.random.RandomState(SEED + 700 + prompt_idx)
    facilities, codes, metrics = fact_sheet(rng)
    doc_tokens = cursor.tokenize(tokenizer, ctx)
    sheet_lines = [f"Access code for facility {f} is {codes[f]}." for f in facilities]
    sheet_lines.append(
        "System metrics: throughput {t} tok/s, latency {l} ms, "
        "error rate {e}, uptime {u} days.".format(
            t=metrics["throughput"], l=metrics["latency_ms"],
            e=metrics["error_rate"], u=metrics["uptime_days"])
    )
    sheet = "\n".join(sheet_lines) + "\n"
    cut = int(len(doc_tokens) * 0.3)
    head = tokenizer.decode(doc_tokens[:cut])
    tail = tokenizer.decode(doc_tokens[cut:])
    full_doc = tokenizer.encode(head + sheet + tail)[: ctx + 128]
    if prompt_idx == 0:
        keys = ["alpha", "beta", "gamma", "delta"]
        targets = [codes[facilities[0]], codes[facilities[3]],
                   codes[facilities[6]], codes[facilities[7]]]
        ask = ("alpha: access code of facility %s; beta: access code of "
               "facility %s; gamma: access code of facility %s; delta: "
               "access code of facility %s" % (
                   facilities[0], facilities[3], facilities[6], facilities[7]))
    else:
        keys = ["throughput", "latency_ms", "error_rate", "uptime_days"]
        targets = [str(metrics["throughput"]), str(metrics["latency_ms"]),
                   str(metrics["error_rate"]), str(metrics["uptime_days"])]
        ask = ("throughput: the numeric throughput value; latency_ms: the "
               "numeric latency value; error_rate: the numeric error rate "
               "value; uptime_days: the numeric uptime value")
    instruction = (
        "\n\nFrom the document above, extract the following fields and "
        "output ONE valid JSON object with exactly these keys and nothing "
        "else: " + ask + ". Output only the JSON object.\nJSON:"
    )
    prompt_tokens = tokenizer.encode(tokenizer.decode(full_doc) + instruction)
    caches = make_caches(model)
    logits = prefill(model, caches, prompt_tokens)
    quantize_caches(caches, arm, seed, codec_cache)
    out = greedy_generate(model, caches, logits, JSON_GEN_TOKENS, eos_ids)
    text = tokenizer.decode(out).strip()
    # Models often repeat the JSON object; grade the FIRST balanced
    # object in the output (v1 used the last '}' and failed on
    # truncated repeats).
    start = text.find("{")
    obj = None
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start: i + 1])
                    except json.JSONDecodeError:
                        obj = None
                    break
    parse_ok = obj is not None
    score = 0.0
    detail = {"raw": text, "keys": keys}
    if parse_ok:
        good = 0
        for k, t in zip(keys, targets):
            v = str(obj.get(k, ""))
            if t.lower() in v.lower():
                good += 1
        score = good / len(keys)
        detail["matched"] = good
    else:
        detail["matched"] = 0
    detail["parse_ok"] = parse_ok
    detail["score"] = score
    caches = None
    mx.clear_cache()
    gc.collect()
    logger.info("json %d: parse_ok=%s score=%.2f", prompt_idx, parse_ok, score)
    return detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True,
                        choices=["bf16", "radial", "turboquant"])
    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo-root", default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tasks", default="needle,ppl,code,json")
    args = parser.parse_args()

    global _TurboQuantKVCache
    if args.arm == "turboquant":
        apply_dense_codec_guard()
        from mlx_vlm.turboquant import TurboQuantKVCache as _TurboQuantKVCache

    tasks = args.tasks.split(",")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(args.model)
    eos_ids = eos_id_set(tokenizer)
    cursor = CorpusCursor()
    codec_cache = {}
    results = {
        "arm": args.arm,
        "ctx": args.ctx,
        "seed": args.seed,
        "model_path": args.model,
        "model_revision": model_revision(args.model),
        "radial_codec": {
            "r_bits": 3, "a_bits": 5, "radius_mode": "uniform",
            "bytes_per_vector": RadialPolarCodec(128).bytes_per_vector(),
            "sketch_bytes_per_vector": sign_sketch_bytes(128),
        },
        "tasks": {},
    }

    if "needle" in tasks:
        results["tasks"]["needle"] = run_needle_set(
            model, tokenizer, cursor, eos_ids, args.arm, args.seed,
            codec_cache, args.ctx)
    if "ppl" in tasks:
        results["tasks"]["ppl"] = run_perplexity(
            model, tokenizer, cursor, args.arm, args.seed, codec_cache,
            args.ctx)
    if "code" in tasks:
        results["tasks"]["code"] = [
            run_code_task(model, tokenizer, cursor, eos_ids, args.arm,
                          args.seed, codec_cache, args.ctx, args.repo_root, i)
            for i in range(len(CODE_FILES))
        ]
    if "json" in tasks:
        results["tasks"]["json"] = [
            run_json_task(model, tokenizer, cursor, eos_ids, args.arm,
                          args.seed, codec_cache, args.ctx, i)
            for i in range(2)
        ]

    total_bytes = sum(e["bytes"] for e in BYTE_EVENTS)
    total_tokens = sum(e["tokens"] for e in BYTE_EVENTS)
    results["byte_accounting"] = {
        "events": BYTE_EVENTS,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "bytes_per_token": round(total_bytes / max(total_tokens, 1), 1),
        "bf16_bytes_per_token": 36 * 8 * 2 * 128 * 2,
    }
    results["wall_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("done in %.1fs -> %s", results["wall_seconds"], args.out)


if __name__ == "__main__":
    main()
