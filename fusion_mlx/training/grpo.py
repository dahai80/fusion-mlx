import logging
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    group_size: int = 4
    iters: int = 50
    batch_size: int = 2
    learning_rate: float = 1e-5
    lora_layers: int = 16
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    max_completion_len: int = 64
    clip_ratio: float = 0.2
    reward_endpoint: str = ""
    temperature: float = 1.0
    seed: int = 0
    reward_timeout: float = 30.0

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class GRPOStepResult:
    loss: float
    mean_reward: float
    mean_ratio: float
    n_samples: int


def _model_forward_logits(model, input_ids):
    # Forward returning (1, L, V) logits, handling both array and
    # (logits, cache) returns from mlx_lm models.
    out = model(input_ids[None, :])
    return out[0] if isinstance(out, tuple) else out


def _seq_logprob_differentiable(model, prompt_ids, completion_ids):
    # Differentiable per-token log p(completion | prompt) under the model.
    # Returns (per_token_logprobs array [n_comp], sum scalar).
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


class GRPOTrainer:
    # Group Relative Policy Optimization with LoRA. Samples G completions per
    # prompt, scores rewards via an HTTP callback, computes group-normalized
    # advantages, and runs PPO-clipped policy-gradient updates against LoRA
    # params. Reference logprobs use a load-on-demand base model (evict after).

    def __init__(self, model, tokenizer, model_path, config):
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.config = config
        self.optimizer = optim.AdamW(learning_rate=config.learning_rate)
        self._reward_history = []

    def _sample_completions(self, prompt_ids, group_size):
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
        for g in range(group_size):
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
            logger.warning(
                "GRPO: no reward_endpoint, using length-based sandbox reward"
            )
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
    def _advantages(rewards):
        n = len(rewards)
        if n == 0:
            return []
        mean = sum(rewards) / n
        var = sum((r - mean) ** 2 for r in rewards) / n
        std = var**0.5
        eps = 1e-8
        if std < eps:
            return [0.0 for _ in rewards]
        return [(r - mean) / (std + eps) for r in rewards]

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

    def _grpo_loss(self, model, batch):
        # batch: list of dicts {prompt_ids, completion_ids, ref_logprob, advantage}.
        # Returns (loss, mean_ratio). Gradients flow through policy logprob only;
        # ref_logprob and advantage are constants.
        losses = []
        ratios = []
        for sample in batch:
            prompt_ids = mx.array(sample["prompt_ids"])
            completion_ids = mx.array(sample["completion_ids"])
            _, policy_lp_sum = _seq_logprob_differentiable(
                model, prompt_ids, completion_ids
            )
            ref_lp = mx.array(sample["ref_logprob"])
            advantage = mx.array(sample["advantage"])
            ratio = mx.exp(policy_lp_sum - ref_lp)
            clipped = mx.clip(
                ratio,
                1 - self.config.clip_ratio,
                1 + self.config.clip_ratio,
            )
            obj = mx.minimum(ratio * advantage, clipped * advantage)
            losses.append(-obj)
            ratios.append(ratio)
        loss = mx.mean(mx.stack(losses))
        mean_ratio = mx.mean(mx.stack(ratios))
        return loss, mean_ratio

    def train_step(self, prompts_batch):
        # One GRPO step over a batch of prompts. Samples completions, queries
        # rewards, computes advantages + reference logprobs, then takes one
        # optimizer step on the PPO-clipped policy loss.
        cfg = self.config
        loss_value_and_grad = nn.value_and_grad(self.model, self._grpo_loss)

        all_samples = []
        all_rewards = []
        for prompt_text in prompts_batch:
            prompt_ids = self.tokenizer.encode(prompt_text)
            completions = self._sample_completions(prompt_ids, cfg.group_size)
            completion_texts = [self.tokenizer.decode(c) for c in completions]
            rewards = self._compute_rewards(prompt_text, completion_texts)
            advantages = self._advantages(rewards)
            all_rewards.extend(rewards)

            for comp_ids, adv in zip(completions, advantages):
                ref_lp = self._ref_logprob(prompt_ids, comp_ids)
                all_samples.append(
                    {
                        "prompt_ids": prompt_ids,
                        "completion_ids": comp_ids,
                        "ref_logprob": ref_lp,
                        "advantage": adv,
                    }
                )

        (loss, mean_ratio), grad = loss_value_and_grad(self.model, all_samples)
        self.optimizer.update(self.model, grad)
        mx.eval(self.model.parameters(), self.optimizer.state)

        mean_reward = sum(all_rewards) / max(len(all_rewards), 1)
        self._reward_history.append(mean_reward)
        result = GRPOStepResult(
            loss=float(loss),
            mean_reward=float(mean_reward),
            mean_ratio=float(mean_ratio),
            n_samples=len(all_samples),
        )
        logger.info(
            "GRPO step: loss=%.6f mean_ratio=%.4f mean_reward=%.4f n_samples=%d",
            result.loss,
            result.mean_ratio,
            result.mean_reward,
            result.n_samples,
        )
        return result

    def save_adapter(self, adapter_path):
        from mlx.utils import tree_flatten

        weights = dict(tree_flatten(self.model.trainable_parameters()))
        mx.save_safetensors(adapter_path, weights)
        logger.info("GRPO: saved adapter to %s", adapter_path)
