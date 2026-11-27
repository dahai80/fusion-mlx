import gc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class RewardScoreResult:
    rewards: list = field(default_factory=list)
    model_id: str = ""
    adapter_name: str = ""

    def to_dict(self):
        return {
            "rewards": self.rewards,
            "model_id": self.model_id,
            "adapter_name": self.adapter_name,
        }


def _attach_value_head(model, prompt_ids):
    # Mirror RewardTrainer._init_head: attach the scalar value head as a
    # registered submodule if the loaded adapter config marks reward_model.
    if getattr(model, "value_head", None) is not None:
        return
    hidden = getattr(model, "hidden_size", None)
    if hidden is None:
        args = getattr(model, "args", None)
        hidden = getattr(args, "hidden_size", None) if args else None
    if hidden is None:
        emb = getattr(model, "embed", None) or getattr(model, "wte", None)
        hidden = emb.weight.shape[1] if emb is not None else None
    if hidden is None:
        out = model(mx.array(prompt_ids)[None, :])
        logits = out[0] if isinstance(out, tuple) else out
        hidden = int(logits.shape[-1])
        logger.warning("reward_score: hidden_size unknown, using logits dim %d", hidden)
    from fusion_mlx.training.reward import _ValueHead

    model.value_head = _ValueHead(int(hidden))
    logger.info("reward_score: attached value head hidden_size=%d", hidden)


def _score_completion(model, prompt_ids, completion_ids):
    # Non-differentiable mirror of RewardTrainer._score: forward the
    # concatenated sequence, take the last-token hidden, project to scalar.
    full = mx.concatenate([mx.array(prompt_ids), mx.array(completion_ids)])
    trunk = getattr(model, "transformer", None) or getattr(model, "model", None)
    if trunk is not None:
        hidden = trunk(full[None, :])
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        hidden = hidden[0]
    else:
        logger.warning(
            "reward_score: backbone hidden unavailable, scoring via logits proxy"
        )
        out = model(full)
        logits = out[0] if isinstance(out, tuple) else out
        n_comp = int(completion_ids.shape[0])
        hidden = mx.mean(logits[0, -n_comp:, :].astype(mx.float32), axis=0)
        hidden = mx.expand_dims(hidden, 0)
    return float(model.value_head(hidden))


def _load_reward_model(model_path, adapter_path):
    # Reward adapters carry value_head.* tensors alongside the LoRA params
    # (#1742). mlx_lm's load_adapters rejects any tensor outside the base
    # model's adapter parameter set, so split the file here: apply the LoRA
    # layers ourselves, validate them loudly, then load the value head onto
    # the attached submodule.
    import mlx_lm.utils as mlx_utils
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, tokenizer = mlx_utils.load(model_path)
    with open(Path(adapter_path) / "adapter_config.json") as f:
        cfg = json.load(f)
    fine_tune_type = cfg.get("fine_tune_type", "lora")
    weights = mx.load(str(Path(adapter_path) / "adapters.safetensors"))
    lora_weights = {k: v for k, v in weights.items() if not k.startswith("value_head.")}
    head_weights = {k: v for k, v in weights.items() if k.startswith("value_head.")}

    if fine_tune_type != "full":
        linear_to_lora_layers(
            model,
            cfg["num_layers"],
            cfg["lora_parameters"],
            use_dora=(fine_tune_type == "dora"),
        )
    param_names = {name for name, _ in tree_flatten(model.parameters())}
    unknown = [k for k in lora_weights if k not in param_names]
    if unknown:
        raise ValueError(
            f"Adapter at {adapter_path} contains incompatible tensors:\n"
            + "\n".join(
                f"  {k}: not an adapter parameter in the model" for k in unknown
            )
        )
    model.load_weights(list(lora_weights.items()), strict=False)

    if head_weights:
        prompt_probe = tokenizer.encode("score")
        _attach_value_head(model, prompt_probe)
        head_items = [(k[len("value_head.") :], v) for k, v in head_weights.items()]
        try:
            model.value_head.load_weights(head_items, strict=True)
        except ValueError as e:
            raise ValueError(
                f"Adapter at {adapter_path} has incompatible value head tensors: {e}"
            )
        mx.eval(model.value_head.parameters())
        logger.info(
            "reward_score: loaded value head (%d tensors) from adapter", len(head_items)
        )
    return model, tokenizer


def score_completions(model_path, adapter_path, prompt, completions):
    # Load model + reward adapter (LoRA + value head), score each completion
    # under the RM value head, evict. Standalone load-and-evict path mirroring
    # logprob.score_text; not routed through the inference pool.

    logger.info(
        "score_completions: model=%s adapter=%s prompt_len=%d n_completions=%d",
        model_path,
        adapter_path,
        len(prompt),
        len(completions),
    )

    config_path = Path(adapter_path) / "adapter_config.json" if adapter_path else None
    is_reward = False
    if config_path and config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        is_reward = bool(cfg.get("reward_model", False))
    if not is_reward:
        raise ValueError(
            f"adapter {adapter_path} is not a reward model "
            "(adapter_config.json missing reward_model=true)"
        )

    model, tokenizer = _load_reward_model(model_path, adapter_path)
    try:
        prompt_ids = tokenizer.encode(prompt)
        _attach_value_head(model, prompt_ids)
        rewards = []
        for comp in completions:
            comp_ids = tokenizer.encode(comp)
            r = _score_completion(model, prompt_ids, comp_ids)
            rewards.append(r)
        logger.info("score_completions: rewards=%s", rewards)
        return rewards
    finally:
        del model
        del tokenizer
        gc.collect()
        mx.clear_cache()
        logger.info("score_completions: model evicted")
