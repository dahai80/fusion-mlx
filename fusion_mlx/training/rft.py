# SPDX-License-Identifier: Apache-2.0
"""RFT (Rejection-sampling Fine-Tuning) trainer.

Algorithm: for each prompt, sample N completions, score each via a reward
callback, keep the top-K by reward, then run plain SFT cross-entropy loss on
the winners (NOT PPO/RL — no reference model, no advantage, no clipping).
This is the simplest variant of rejection-sampling fine-tuning (ReST, RFT).

Mirrors fusion_mlx.training.grpo structure: same sampling, same reward
callback contract, same LoRA-only param set. Differs in the loss — pure
teacher-forced cross-entropy over the winning completions.
"""

import logging
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

logger = logging.getLogger(__name__)


@dataclass
class RFTConfig:
    num_samples: int = 8
    top_k: int = 2
    iters: int = 50
    batch_size: int = 2
    learning_rate: float = 1e-5
    lora_layers: int = 16
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    max_completion_len: int = 64
    reward_endpoint: str = ""
    temperature: float = 1.0
    seed: int = 0
    reward_timeout: float = 30.0

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class RFTStepResult:
    loss: float
    mean_reward: float
    accept_rate: float
    n_samples: int
    n_winners: int


def _model_forward_logits(model, input_ids):
    out = model(input_ids[None, :])
    return out[0] if isinstance(out, tuple) else out


def _sft_loss_differentiable(model, prompt_ids, completion_ids):
    # Teacher-forced cross-entropy over completion tokens. Returns
    # (per_token_logprobs [n_comp], mean_ce_loss scalar). Gradients flow
    # through the model only.
    full = mx.concatenate([prompt_ids, completion_ids])
    logits = _model_forward_logits(model, full)
    n_prompt = int(prompt_ids.shape[0])
    n_comp = int(completion_ids.shape[0])
    start = n_prompt - 1
    predictor = logits[0, start : start + n_comp, :].astype(mx.float32)
    log_probs = nn.log_softmax(predictor)
    targets = completion_ids[..., None]
    per_token = mx.take_along_axis(log_probs, targets, axis=-1)[..., 0]
    # mean negative log-likelihood over completion tokens (SFT loss)
    ce_loss = -mx.mean(per_token)
    return per_token, ce_loss


class RFTTrainer:
    # Rejection-sampling Fine-Tuning with LoRA. Samples N completions per
    # prompt, scores rewards via an HTTP callback, keeps top_k by reward, and
    # runs SFT cross-entropy updates on the winners. No reference model, no
    # PPO clipping — plain supervised loss on rejection-filtered samples.

    def __init__(self, model, tokenizer, model_path, config):
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.config = config
        self.optimizer = optim.AdamW(learning_rate=config.learning_rate)
        self._reward_history = []

    def _sample_completions(self, prompt_ids, num_samples):
        # Temperature sampling via mlx_lm.generate_step. temperature<=0 = greedy.
        from mlx_lm.generate import generate_step

        completions = []
        prompt_arr = (
            mx.array(prompt_ids) if not isinstance(prompt_ids, mx.array) else prompt_ids
        )
        sampler = None
        if self.config.temperature > 0:
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(self.config.temperature, 0.0)
        for g in range(num_samples):
            tokens = []
            for token_id, _ in generate_step(
                prompt_arr,
                self.model,
                max_tokens=self.config.max_completion_len,
                sampler=sampler,
            ):
                tid = int(token_id)
                tokens.append(tid)
                if tid == self.tokenizer.eos_token_id:
                    break
            completions.append(tokens)
        return completions

    def _compute_rewards(self, prompt_text, completion_texts):
        # HTTP POST to reward_endpoint: {prompt, completions} -> {rewards}.
        # Falls back to length-based reward when no endpoint configured.
        if not self.config.reward_endpoint:
            logger.warning("RFT: no reward_endpoint, using length-based sandbox reward")
            return [float(len(c)) for c in completion_texts]

        import json
        import urllib.request

        payload = json.dumps(
            {"prompt": prompt_text, "completions": completion_texts}
        ).encode()
        req = urllib.request.Request(
            self.config.reward_endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.reward_timeout) as resp:
            data = json.loads(resp.read().decode())
        rewards = data.get("rewards", [])
        if len(rewards) != len(completion_texts):
            raise ValueError(
                f"reward_endpoint returned {len(rewards)} rewards for "
                f"{len(completion_texts)} completions"
            )
        return [float(r) for r in rewards]

    @staticmethod
    def _select_top_k(completions, rewards, top_k):
        # Keep the top_k completions by reward. Returns list of
        # (completion_ids, reward) pairs. top_k clamped to available count.
        n = len(completions)
        if n == 0:
            return []
        k = max(1, min(top_k, n))
        ranked = sorted(zip(completions, rewards), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    def _sft_loss(self, model, winners):
        # winners: list of {prompt_ids, completion_ids}. Returns mean SFT
        # cross-entropy loss over all winning completion tokens.
        losses = []
        for sample in winners:
            prompt_ids = mx.array(sample["prompt_ids"])
            completion_ids = mx.array(sample["completion_ids"])
            _, ce_loss = _sft_loss_differentiable(model, prompt_ids, completion_ids)
            losses.append(ce_loss)
        return mx.mean(mx.stack(losses))

    def train_step(self, prompts_batch):
        # One RFT step over a batch of prompts. Samples N completions, queries
        # rewards, keeps top_k winners per prompt, then takes one optimizer
        # step on the SFT cross-entropy loss over winners.
        cfg = self.config
        loss_value_and_grad = nn.value_and_grad(self.model, self._sft_loss)

        all_winners = []
        all_rewards = []
        total_sampled = 0
        for prompt_text in prompts_batch:
            prompt_ids = self.tokenizer.encode(prompt_text)
            completions = self._sample_completions(prompt_ids, cfg.num_samples)
            completion_texts = [self.tokenizer.decode(c) for c in completions]
            rewards = self._compute_rewards(prompt_text, completion_texts)
            all_rewards.extend(rewards)
            total_sampled += len(completions)

            picked = self._select_top_k(completions, rewards, cfg.top_k)
            for comp_ids, _r in picked:
                all_winners.append(
                    {
                        "prompt_ids": prompt_ids,
                        "completion_ids": comp_ids,
                    }
                )

        n_winners = len(all_winners)
        if n_winners == 0:
            raise ValueError("RFT step produced no winners (empty batch or sampling)")

        loss, grad = loss_value_and_grad(self.model, all_winners)
        self.optimizer.update(self.model, grad)
        mx.eval(self.model.parameters(), self.optimizer.state)

        mean_reward = sum(all_rewards) / max(len(all_rewards), 1)
        accept_rate = n_winners / max(total_sampled, 1)
        self._reward_history.append(mean_reward)
        result = RFTStepResult(
            loss=float(loss),
            mean_reward=float(mean_reward),
            accept_rate=float(accept_rate),
            n_samples=total_sampled,
            n_winners=n_winners,
        )
        logger.info(
            "RFT step: loss=%.6f mean_reward=%.4f accept_rate=%.4f "
            "n_samples=%d n_winners=%d",
            result.loss,
            result.mean_reward,
            result.accept_rate,
            result.n_samples,
            result.n_winners,
        )
        return result

    def save_adapter(self, adapter_path):
        from mlx.utils import tree_flatten

        weights = dict(tree_flatten(self.model.trainable_parameters()))
        mx.save_safetensors(adapter_path, weights)
        logger.info("RFT: saved adapter to %s", adapter_path)
