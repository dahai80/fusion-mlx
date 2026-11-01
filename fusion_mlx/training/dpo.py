# SPDX-License-Identifier: Apache-2.0
"""DPO / ORPO preference-alignment training (#399).

Importers/callers:
  - fusion_mlx.training.dpo_service imports DPOConfig + DPOTrainer
  - fusion_mlx.admin.fine_tune_route imports DPOService + DPOJob for routes

Affected API: new /admin/api/fine-tune/dpo/jobs + /orpo/jobs endpoints

Data schemas:
  DPOConfig      — beta, lr, iters, lora params, method (dpo|orpo)
  DPOStepResult  — per-step metrics (loss, reward_margin, acc_chosen)
  DPOTrainer     — preference-pair training loop (DPO ref-model loss / ORPO
                   odds-ratio loss), reuses the GRPO load-and-evict ref pattern

User verbatim instruction: "启动3个功能issue的修复落地"
"""

from __future__ import annotations

import functools
import gc
import logging
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    method: str = "dpo"  # dpo | orpo
    iters: int = 50
    batch_size: int = 2
    learning_rate: float = 1e-5
    lora_layers: int = 16
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    beta: float = 0.1  # DPO temperature; ORPO ignores (uses lambda_odds)
    lambda_odds: float = 1.0  # ORPO odds-ratio penalty weight
    max_seq_length: int = 1024
    seed: int = 0
    init_adapter_path: str = ""  # SFT adapter to chain DPO on (#1149)
    # #1460: DPO runs policy + reference forwards per pair — double the SFT
    # forward traffic — yet had no way to enable gradient checkpointing or an
    # admission budget. Same semantics as FineTuneConfig.
    gradient_checkpointing: bool = False
    memory_limit_gb: float = 0.0  # admission-gate budget; 0 = auto
    # #1472: keep ONE reference model resident for the whole job instead of
    # the per-pair load-and-evict (2 full loads per pair: chosen + rejected).
    # On a large base the repeated same-size allocations churn the Metal
    # allocator and phys_footprint climbs until the memory_enforcer fatal-exits
    # the engine (27B-8bit DPO observed at 165.9GB over a 120GB ceiling ~90s
    # after job start). Resident = policy + ref weights held steady (~2x model
    # bytes) — the right trade on a dedicated training box with free memory.
    # Default False preserves the load-and-evict behavior for
    # shared/serving deployments.
    keep_ref_resident: bool = False

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class DPOStepResult:
    loss: float
    reward_margin: float  # mean(policy_w - policy_l) in reward space
    acc_chosen: float  # fraction of pairs where chosen preferred


def _model_forward_logits(model, input_ids):
    out = model(input_ids[None, :])
    return out[0] if isinstance(out, tuple) else out


def _seq_logprob_differentiable(model, prompt_ids, completion_ids):
    # Differentiable per-token log p(completion | prompt). Returns (per_token, sum).
    full = mx.concatenate([prompt_ids, completion_ids])
    logits = _model_forward_logits(model, full)
    n_prompt = int(prompt_ids.shape[0])
    n_comp = int(completion_ids.shape[0])
    start = n_prompt - 1
    predictor = logits[0, start : start + n_comp, :].astype(mx.float32)
    log_probs = nn.log_softmax(predictor)
    targets = completion_ids[..., None]
    per_token = mx.take_along_axis(log_probs, targets, axis=-1)[..., 0]
    return per_token, mx.sum(per_token)


class DPOTrainer:
    # Preference-pair alignment with LoRA. DPO needs a frozen reference model
    # (load-on-demand base, evict after — same pattern as GRPO._ref_logprob).
    # ORPO folds the reference into an odds-ratio penalty (no ref model).

    def __init__(self, model, tokenizer, model_path, config):
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.config = config
        self.optimizer = optim.AdamW(learning_rate=config.learning_rate)
        self._ref_cache = {}  # (prompt, chosen, rejected) -> (ref_w, ref_l)
        self._ref_model = None  # #1472: resident ref when keep_ref_resident
        # #1469: compiled train step. MLX only streams (allocates+freed
        # per-op) inside mx.compile — a bare nn.value_and_grad keeps every
        # layer's activations alive through the whole backward eval, so a
        # 27B-8bit DPO step at seq 1024 peaked at 178GB wired and the
        # memory_enforcer fatal-exited the engine. Same pattern as the
        # mlx_lm SFT trainer: optimizer.update INSIDE the compiled fn,
        # grads never escape it. Measured on 4B/seq2048: eager 37GB peak,
        # compiled+checkpointing 6.7GB.
        self._compiled_step = None
        self._compile_failed = False

    def close(self):
        # #1472: free a resident reference model. Load-and-evict mode has
        # nothing to free here (per-pair evict already ran).
        if self._ref_model is not None:
            del self._ref_model
            self._ref_model = None
            gc.collect()
            mx.clear_cache()

    def _ref_logprob(self, prompt_ids, completion_ids, ref_model=None):
        # Load base model (no adapter), score, evict. Standalone load-and-evict.
        # #1472: ref_model given = resident mode — no load, no evict.
        import gc

        import mlx_lm.utils as mlx_utils

        model = ref_model
        owned = False
        if model is None:
            model, _ = mlx_utils.load(self.model_path, adapter_path=None)
            owned = True
        try:
            from fusion_mlx.training.logprob import compute_logprob

            total, _, _ = compute_logprob(model, prompt_ids, completion_ids)
            return total
        finally:
            if owned:
                del model
                gc.collect()
                mx.clear_cache()

    def _ref_logprobs_pair(self, prompt_ids, chosen_ids, rejected_ids):
        # Cache ref logprobs per pair to avoid reloading the base twice per step.
        # prompt_ids/chosen_ids/rejected_ids 来自 tokenizer.encode,是 Python list;
        # mx.sum 只接受 mx.array,需先转换(#420)。
        key = (
            int(mx.sum(mx.array(prompt_ids))),
            int(mx.sum(mx.array(chosen_ids))),
            int(mx.sum(mx.array(rejected_ids))),
        )
        if key in self._ref_cache:
            return self._ref_cache[key]
        if self.config.keep_ref_resident:
            # #1472: one resident ref load for the whole job — per-pair
            # load-and-evict churns same-size Metal allocations on large
            # bases until the memory_enforcer fatal-exits.
            if self._ref_model is None:
                import mlx_lm.utils as mlx_utils

                self._ref_model, _ = mlx_utils.load(self.model_path, adapter_path=None)
                logger.info(
                    "%s: resident reference model loaded (#1472)",
                    self.config.method.upper(),
                )
            ref_w = self._ref_logprob(prompt_ids, chosen_ids, ref_model=self._ref_model)
            ref_l = self._ref_logprob(
                prompt_ids, rejected_ids, ref_model=self._ref_model
            )
        else:
            ref_w = self._ref_logprob(prompt_ids, chosen_ids)
            ref_l = self._ref_logprob(prompt_ids, rejected_ids)
        self._ref_cache[key] = (ref_w, ref_l)
        return ref_w, ref_l

    @staticmethod
    def _pad_to(ids, n, pad_id=0):
        # Normalize to python ints: tokenizers return list[int], but stubs
        # (and some callers) hand back mx arrays whose list() yields 0-d
        # arrays — mx.array of those misbehaves.
        return [int(x) for x in ids] + [pad_id] * (n - len(ids))

    def _batch_tensors(self, samples):
        # Pad the batch to one (P, C) shape so every mx.compile call traces
        # the same graph — variable per-sample lengths would recompile per
        # step and grow the compile cache (#1469). Pads sit after each real
        # sequence; causal attention keeps them from touching real rows,
        # and completion masks zero their logprob contribution.
        P = max(len(s["prompt_ids"]) for s in samples)
        C = max(max(len(s["chosen_ids"]), len(s["rejected_ids"])) for s in samples)
        C = max(C, 1)
        prompt = mx.array([self._pad_to(s["prompt_ids"], P) for s in samples])
        starts = mx.array([len(s["prompt_ids"]) - 1 for s in samples])
        chosen = mx.array([self._pad_to(s["chosen_ids"], C) for s in samples])
        rejected = mx.array([self._pad_to(s["rejected_ids"], C) for s in samples])
        cmask = mx.array(
            [
                [1.0] * len(s["chosen_ids"]) + [0.0] * (C - len(s["chosen_ids"]))
                for s in samples
            ]
        )
        rmask = mx.array(
            [
                [1.0] * len(s["rejected_ids"]) + [0.0] * (C - len(s["rejected_ids"]))
                for s in samples
            ]
        )
        ref_w = mx.array([float(s.get("ref_w", 0.0)) for s in samples])
        ref_l = mx.array([float(s.get("ref_l", 0.0)) for s in samples])
        return {
            "prompt": prompt,
            "starts": starts,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_mask": cmask,
            "rejected_mask": rmask,
            "ref_w": ref_w,
            "ref_l": ref_l,
        }

    @staticmethod
    def _padded_logprob(model, prompt, starts, comp, comp_mask):
        # prompt (B,P) int; comp (B,C) int; starts (B,) = per-sample predictor
        # row for comp[0] (prompt_len-1); comp_mask (B,C) zeroes padded rows.
        # Returns (totals (B,), per_token (B,C)) differentiable logprobs.
        full = mx.concatenate([prompt, comp], axis=1)
        out = model(full)
        logits = out[0] if isinstance(out, tuple) else out
        C = int(comp.shape[1])
        idx = starts[:, None, None] + mx.arange(C)[None, :, None]
        pred = mx.take_along_axis(logits, idx, axis=1).astype(mx.float32)
        log_probs = nn.log_softmax(pred)
        per_token = mx.take_along_axis(log_probs, comp[..., None], axis=-1)[..., 0]
        totals = mx.sum(per_token * comp_mask, axis=1)
        return totals, per_token

    def _dpo_graph_loss(self, model, t):
        # L = -log sigma(beta*((pi_w-ref_w) - (pi_l-ref_l))), batched tensors.
        p_w, _ = self._padded_logprob(
            model, t["prompt"], t["starts"], t["chosen"], t["chosen_mask"]
        )
        p_l, _ = self._padded_logprob(
            model, t["prompt"], t["starts"], t["rejected"], t["rejected_mask"]
        )
        logits = self.config.beta * ((p_w - t["ref_w"]) - (p_l - t["ref_l"]))
        loss = mx.mean(-nn.log_sigmoid(logits))
        return loss, logits

    def _orpo_graph_loss(self, model, t):
        # ORPO: SFT loss on chosen + lambda*log sigma(log(p_w/p_l)) odds-ratio
        # penalty. No reference model (the odds ratio replaces the ref subtraction).
        p_w, per_w = self._padded_logprob(
            model, t["prompt"], t["starts"], t["chosen"], t["chosen_mask"]
        )
        p_l, _ = self._padded_logprob(
            model, t["prompt"], t["starts"], t["rejected"], t["rejected_mask"]
        )
        n_tok = mx.maximum(mx.sum(t["chosen_mask"]), 1e-8)
        sft = -mx.sum(per_w * t["chosen_mask"]) / n_tok
        log_odds = p_w - p_l
        loss = sft + self.config.lambda_odds * mx.mean(-nn.log_sigmoid(log_odds))
        return loss, log_odds

    def _dpo_loss(self, model, batch):
        # Legacy sample-list entry point (unit tests). Runs the same graph
        # loss as the compiled path, evaluated eagerly.
        t = self._batch_tensors(batch)
        loss, logits = self._dpo_graph_loss(model, t)
        mx.eval(loss, logits)
        margins = logits.tolist()
        accs = [float(m > 0) for m in margins]
        return loss, margins, accs

    def _orpo_loss(self, model, batch):
        t = self._batch_tensors(batch)
        loss, logits = self._orpo_graph_loss(model, t)
        mx.eval(loss, logits)
        margins = logits.tolist()
        accs = [float(m > 0) for m in margins]
        return loss, margins, accs

    def _ensure_compiled_step(self):
        if self._compiled_step is not None or self._compile_failed:
            return
        cfg = self.config
        loss_fn = (
            self._orpo_graph_loss if cfg.method == "orpo" else self._dpo_graph_loss
        )
        value_and_grad = nn.value_and_grad(self.model, loss_fn)
        state = [self.model.state, self.optimizer.state, mx.random.state]

        @functools.partial(mx.compile, inputs=state, outputs=state)
        def step(prompt, starts, chosen, rejected, cmask, rmask, ref_w, ref_l):
            t = {
                "prompt": prompt,
                "starts": starts,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_mask": cmask,
                "rejected_mask": rmask,
                "ref_w": ref_w,
                "ref_l": ref_l,
            }
            (loss, per_sample), grad = value_and_grad(self.model, t)
            self.optimizer.update(self.model, grad)
            return loss, per_sample

        self._compiled_step = step
        logger.info(
            "%s: compiled train step enabled (mx.compile streams forward+backward "
            "intermediates; bare value_and_grad retains them, #1469)",
            cfg.method.upper(),
        )

    def train_step(self, pairs_batch):
        # One step over a batch of preference pairs. For DPO, precompute ref
        # logprobs (load-and-evict or resident); for ORPO no ref needed.
        cfg = self.config
        is_orpo = cfg.method == "orpo"

        samples = []
        for pair in pairs_batch:
            prompt_ids = self.tokenizer.encode(pair["prompt"])
            chosen_ids = self.tokenizer.encode(pair["chosen"])
            rejected_ids = self.tokenizer.encode(pair["rejected"])
            # Truncate to max_seq_length budget (prompt + completion)
            budget = cfg.max_seq_length
            if len(chosen_ids) + len(prompt_ids) > budget:
                chosen_ids = chosen_ids[: budget - len(prompt_ids)]
            if len(rejected_ids) + len(prompt_ids) > budget:
                rejected_ids = rejected_ids[: budget - len(prompt_ids)]
            sample = {
                "prompt_ids": prompt_ids,
                "chosen_ids": chosen_ids,
                "rejected_ids": rejected_ids,
            }
            if not is_orpo:
                ref_w, ref_l = self._ref_logprobs_pair(
                    prompt_ids, chosen_ids, rejected_ids
                )
                sample["ref_w"] = ref_w
                sample["ref_l"] = ref_l
            samples.append(sample)

        t = self._batch_tensors(samples)
        loss = per_sample = None
        if not self._compile_failed:
            try:
                self._ensure_compiled_step()
                loss, per_sample = self._compiled_step(
                    t["prompt"],
                    t["starts"],
                    t["chosen"],
                    t["rejected"],
                    t["chosen_mask"],
                    t["rejected_mask"],
                    t["ref_w"],
                    t["ref_l"],
                )
                mx.eval(loss, per_sample)
            except Exception as exc:
                self._compiled_step = None
                self._compile_failed = True
                logger.warning(
                    "%s: mx.compile train step failed (%s) — falling back to "
                    "eager value_and_grad; backward intermediates will NOT "
                    "stream and large models may hit the memory enforcer (#1469)",
                    cfg.method.upper(),
                    exc,
                )
        if self._compile_failed:
            loss_fn = self._orpo_graph_loss if is_orpo else self._dpo_graph_loss
            value_and_grad = nn.value_and_grad(self.model, loss_fn)
            (loss, per_sample), grad = value_and_grad(self.model, t)
            self.optimizer.update(self.model, grad)
            mx.eval(loss, per_sample)
            mx.eval(self.model.parameters(), self.optimizer.state)

        margins = [float(m) for m in per_sample.tolist()]
        accs = [float(m > 0) for m in margins]
        n = max(len(margins), 1)
        result = DPOStepResult(
            loss=float(loss),
            reward_margin=sum(margins) / n,
            acc_chosen=sum(accs) / n,
        )
        logger.info(
            "%s step: loss=%.6f reward_margin=%.4f acc_chosen=%.4f n_pairs=%d",
            cfg.method.upper(),
            result.loss,
            result.reward_margin,
            result.acc_chosen,
            len(samples),
        )
        return result

    def save_adapter(self, adapter_path):
        from mlx.utils import tree_flatten

        weights = dict(tree_flatten(self.model.trainable_parameters()))
        mx.save_safetensors(adapter_path, weights)
        logger.info("%s: saved adapter to %s", self.config.method.upper(), adapter_path)
