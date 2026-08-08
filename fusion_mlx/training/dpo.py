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

    def _ref_logprob(self, prompt_ids, completion_ids):
        # Load base model (no adapter), score, evict. Standalone load-and-evict.
        import gc

        import mlx_lm.utils as mlx_utils

        model, _ = mlx_utils.load(self.model_path, adapter_path=None)
        try:
            from fusion_mlx.training.logprob import compute_logprob

            total, _, _ = compute_logprob(model, prompt_ids, completion_ids)
            return total
        finally:
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
        ref_w = self._ref_logprob(prompt_ids, chosen_ids)
        ref_l = self._ref_logprob(prompt_ids, rejected_ids)
        self._ref_cache[key] = (ref_w, ref_l)
        return ref_w, ref_l

    def _dpo_loss(self, model, batch):
        # batch: list of {prompt_ids, chosen_ids, rejected_ids, ref_w, ref_l}.
        # L = -log sigma(beta*((pi_w-ref_w) - (pi_l-ref_l)))
        cfg = self.config
        losses = []
        margins = []
        accs = []
        for s in batch:
            p = mx.array(s["prompt_ids"])
            cw = mx.array(s["chosen_ids"])
            cl = mx.array(s["rejected_ids"])
            _, policy_w = _seq_logprob_differentiable(model, p, cw)
            _, policy_l = _seq_logprob_differentiable(model, p, cl)
            ref_w = mx.array(s["ref_w"])
            ref_l = mx.array(s["ref_l"])
            logits = cfg.beta * ((policy_w - ref_w) - (policy_l - ref_l))
            losses.append(-nn.log_sigmoid(logits))
            margins.append(float(logits))
            accs.append(float(logits > 0))
        loss = mx.mean(mx.stack(losses))
        return loss, margins, accs

    def _orpo_loss(self, model, batch):
        # ORPO: SFT loss on chosen + lambda*log sigma(log(p_w/p_l)) odds-ratio
        # penalty. No reference model (the odds ratio replaces the ref subtraction).
        cfg = self.config
        sft_losses = []
        odds_losses = []
        margins = []
        accs = []
        for s in batch:
            p = mx.array(s["prompt_ids"])
            cw = mx.array(s["chosen_ids"])
            cl = mx.array(s["rejected_ids"])
            per_w, policy_w = _seq_logprob_differentiable(model, p, cw)
            _, policy_l = _seq_logprob_differentiable(model, p, cl)
            # SFT: negative log-likelihood of chosen (per-token mean)
            sft_losses.append(-mx.mean(per_w))
            # Odds-ratio: log sigma(log p_w - log p_l)
            log_odds = policy_w - policy_l
            odds_losses.append(-nn.log_sigmoid(log_odds))
            margins.append(float(log_odds))
            accs.append(float(log_odds > 0))
        loss = mx.mean(mx.stack(sft_losses)) + cfg.lambda_odds * mx.mean(
            mx.stack(odds_losses)
        )
        return loss, margins, accs

    def train_step(self, pairs_batch):
        # One step over a batch of preference pairs. For DPO, precompute ref
        # logprobs (load-and-evict); for ORPO no ref needed.
        cfg = self.config
        is_orpo = cfg.method == "orpo"
        loss_fn = self._orpo_loss if is_orpo else self._dpo_loss
        loss_value_and_grad = nn.value_and_grad(self.model, loss_fn)

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

        (loss, margins, accs), grad = loss_value_and_grad(self.model, samples)
        self.optimizer.update(self.model, grad)
        mx.eval(self.model.parameters(), self.optimizer.state)

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
