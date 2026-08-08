# SPDX-License-Identifier: Apache-2.0
"""Reward model training (#424).

Importers/callers:
  - fusion_mlx.training.reward_service imports RewardConfig + RewardTrainer
  - fusion_mlx.admin.fine_tune_route imports RewardService + RewardJob for routes

Affected API: new /admin/api/fine-tune/reward/jobs endpoints
  (POST create, GET list/{id}, POST cancel, DELETE)

Data schemas:
  RewardConfig    — iters, lr, batch_size, lora params, margin
  RewardStepResult — per-step metrics (loss, reward_margin, acc_chosen)
  RewardTrainer   — preference-pair reward-model training (Bradley-Terry),
                    LoRA backbone + scalar value head; reuses the DPO
                    load-and-evict / value_and_grad pattern from #399

User verbatim instruction: "对上游依赖给上游提issue和pr"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    iters: int = 50
    batch_size: int = 2
    learning_rate: float = 1e-5
    lora_layers: int = 16
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    max_seq_length: int = 1024
    seed: int = 0

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class RewardStepResult:
    loss: float
    reward_margin: float  # mean(score_w - score_l)
    acc_chosen: float  # fraction of pairs where chosen scored higher


def _model_forward_logits(model, input_ids):
    out = model(input_ids[None, :])
    return out[0] if isinstance(out, tuple) else out


class _ValueHead(nn.Module):
    # Scalar reward head over the last-token hidden state. Attached on top of
    # the (LoRA-wrapped) backbone; only this + LoRA params are trainable.

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def __call__(self, hidden):
        # hidden: (seq, hidden) -> last-token scalar.
        return self.proj(hidden[-1])


class RewardTrainer:
    # Preference-pair reward-model training with LoRA. Trains a scalar value
    # head so score(chosen) > score(rejected): Bradley-Terry loss
    # L = -log sigma(score_w - score_l). No reference model (unlike DPO).

    def __init__(self, model, tokenizer, model_path, config):
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.config = config
        self.optimizer = optim.AdamW(learning_rate=config.learning_rate)
        self._head = None
        self._warned_proxy = False

    def _hidden_size(self, prompt_ids):
        # Infer backbone hidden dim. Prefer model.hidden_size / args.hidden_size;
        # fall back to embedding output dim.
        hidden = getattr(self.model, "hidden_size", None)
        if hidden is None:
            args = getattr(self.model, "args", None)
            hidden = getattr(args, "hidden_size", None) if args else None
        if hidden is None:
            emb = getattr(self.model, "embed", None) or getattr(self.model, "wte", None)
            hidden = emb.weight.shape[1] if emb is not None else None
        if hidden is None:
            logits = _model_forward_logits(self.model, mx.array(prompt_ids))
            if isinstance(logits, tuple):
                logits = logits[0]
            hidden = int(logits.shape[-1])
            if not self._warned_proxy:
                logger.warning(
                    "RewardTrainer: hidden_size unknown, using logits dim %d",
                    hidden,
                )
                self._warned_proxy = True
        return int(hidden)

    def _init_head(self, prompt_ids):
        # Attach the value head as a registered submodule of the model so its
        # params flow through model.trainable_parameters() and a single
        # nn.value_and_grad(self.model, ...) call (MLX value_and_grad takes one
        # module, not a list — verified).
        if getattr(self.model, "value_head", None) is not None:
            self._head = self.model.value_head
            return
        hidden = self._hidden_size(prompt_ids)
        self.model.value_head = _ValueHead(hidden)
        self._head = self.model.value_head
        logger.info("RewardTrainer: value head hidden_size=%d", hidden)

    def _score(self, model, prompt_ids, completion_ids):
        # Differentiable scalar score for (prompt, completion): forward the
        # concatenated sequence, take the last-token hidden, project to scalar.
        full = mx.concatenate([prompt_ids, completion_ids])
        trunk = getattr(model, "transformer", None) or getattr(model, "model", None)
        if trunk is not None:
            hidden = trunk(full[None, :])
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            hidden = hidden[0]  # (seq, hidden)
        else:
            if not self._warned_proxy:
                logger.warning(
                    "RewardTrainer: backbone hidden unavailable, "
                    "scoring via logits-mean proxy"
                )
                self._warned_proxy = True
            logits = _model_forward_logits(model, full)
            if isinstance(logits, tuple):
                logits = logits[0]
            n_comp = int(completion_ids.shape[0])
            # (n_comp, vocab) mean-pooled to (vocab,) — head input dim must
            # match; _hidden_size fallback returns vocab dim in this branch.
            hidden = mx.mean(logits[0, -n_comp:, :].astype(mx.float32), axis=0)
            hidden = mx.expand_dims(hidden, 0)  # (1, vocab) so head[-1] works
        return model.value_head(hidden)

    def _reward_loss(self, model, batch):
        # batch: list of {prompt_ids, chosen_ids, rejected_ids}.
        # L = -log sigma(score_w - score_l)  (Bradley-Terry).
        losses = []
        margins = []
        accs = []
        for s in batch:
            p = mx.array(s["prompt_ids"])
            cw = mx.array(s["chosen_ids"])
            cl = mx.array(s["rejected_ids"])
            self._init_head(p)
            score_w = self._score(model, p, cw)
            score_l = self._score(model, p, cl)
            diff = score_w - score_l
            losses.append(-nn.log_sigmoid(diff))
            margins.append(float(diff))
            accs.append(float(diff > 0))
        loss = mx.mean(mx.stack(losses))
        return loss, margins, accs

    def train_step(self, pairs_batch):
        cfg = self.config
        samples = []
        for pair in pairs_batch:
            prompt_ids = self.tokenizer.encode(pair["prompt"])
            chosen_ids = self.tokenizer.encode(pair["chosen"])
            rejected_ids = self.tokenizer.encode(pair["rejected"])
            budget = cfg.max_seq_length
            if len(chosen_ids) + len(prompt_ids) > budget:
                chosen_ids = chosen_ids[: budget - len(prompt_ids)]
            if len(rejected_ids) + len(prompt_ids) > budget:
                rejected_ids = rejected_ids[: budget - len(prompt_ids)]
            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "chosen_ids": chosen_ids,
                    "rejected_ids": rejected_ids,
                }
            )

        # Build head (registered on model) before value_and_grad so its params
        # are in model.trainable_parameters().
        self._init_head(mx.array(samples[0]["prompt_ids"]))

        loss_value_and_grad = nn.value_and_grad(self.model, self._reward_loss)
        (loss, margins, accs), grads = loss_value_and_grad(self.model, samples)
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)

        n = max(len(margins), 1)
        result = RewardStepResult(
            loss=float(loss),
            reward_margin=sum(margins) / n,
            acc_chosen=sum(accs) / n,
        )
        logger.info(
            "REWARD step: loss=%.6f reward_margin=%.4f acc_chosen=%.4f n_pairs=%d",
            result.loss,
            result.reward_margin,
            result.acc_chosen,
            len(samples),
        )
        return result

    def save_adapter(self, adapter_path):
        from mlx.utils import tree_flatten

        # Head is a registered submodule (value_head.*), so trainable_parameters
        # already includes both LoRA params and the value head.
        weights = dict(tree_flatten(self.model.trainable_parameters()))
        mx.save_safetensors(adapter_path, weights)
        logger.info("REWARD: saved reward adapter to %s", adapter_path)
