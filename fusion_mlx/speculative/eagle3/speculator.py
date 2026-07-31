# SPDX-License-Identifier: Apache-2.0
import logging
import os
import time
from dataclasses import dataclass

import mlx.core as mx

logger = logging.getLogger(__name__)

EAGLE3_DRAFT_MODELS = {
    "llama3.1-8b": {
        "hf_path": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "target_family": "llama3",
        "draft_vocab_size": 32000,
    },
    "qwen3-8b": {
        "hf_path": "RedHatAI/Qwen3-8B-speculator.eagle3",
        "target_family": "qwen3",
        "draft_vocab_size": 151936,
    },
}

ENV_DRAFT_MODEL = "FUSION_EAGLE3_DRAFT_MODEL"
ENV_DRAFT_TOKENS = "FUSION_EAGLE3_DRAFT_TOKENS"
ENV_DRAFT_TEMP = "FUSION_EAGLE3_DRAFT_TEMP"


@dataclass
class Eagle3DraftConfig:
    draft_model_key: str = "llama3.1-8b"
    num_draft: int = 5
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "Eagle3DraftConfig":
        key = os.environ.get(ENV_DRAFT_MODEL, "llama3.1-8b")
        num_draft = int(os.environ.get(ENV_DRAFT_TOKENS, "5"))
        temp = float(os.environ.get(ENV_DRAFT_TEMP, "0.0"))
        if key not in EAGLE3_DRAFT_MODELS:
            logger.warning(
                "eagle3: unknown draft_model_key=%s, fallback llama3.1-8b", key
            )
            key = "llama3.1-8b"
        return cls(draft_model_key=key, num_draft=num_draft, temperature=temp)


class Eagle3Speculator:
    def __init__(self, config: Eagle3DraftConfig | None = None):
        self.config = config or Eagle3DraftConfig.from_env()
        self.model = None
        self.tokenizer = None
        self._draft_cache = None
        self._prev_token = None
        self._loaded = False
        self._total_drafts = 0
        self._total_accepted = 0

    @property
    def model_path(self) -> str:
        info = EAGLE3_DRAFT_MODELS.get(self.config.draft_model_key)
        if info is None:
            return ""
        return info["hf_path"]

    @property
    def target_family(self) -> str:
        info = EAGLE3_DRAFT_MODELS.get(self.config.draft_model_key)
        if info is None:
            return ""
        return info["target_family"]

    def load(self) -> bool:
        if self._loaded:
            return True
        try:
            t0 = time.perf_counter()
            import mlx_lm

            hf_path = self.model_path
            if not hf_path:
                logger.warning("eagle3: no draft model path configured")
                return False

            local_path = os.path.expanduser(
                os.path.join("~/.fusion-mlx/models", hf_path)
            )
            if os.path.isdir(local_path):
                load_path = local_path
            else:
                load_path = hf_path
                logger.info(
                    "eagle3: local path %s not found, using HF path %s",
                    local_path,
                    hf_path,
                )

            self.model, self.tokenizer = mlx_lm.load(load_path)
            dt = time.perf_counter() - t0
            self._loaded = True
            logger.info(
                "eagle3: loaded %s in %.1fs, num_draft=%d, target=%s",
                load_path,
                dt,
                self.config.num_draft,
                self.target_family,
            )
            return True
        except Exception as e:
            logger.warning("eagle3: failed to load %s: %s", self.model_path, e)
            return False

    def reset(self):
        self._draft_cache = None
        self._prev_token = None

    def on_new_request(self, request_id: str, prompt_tokens: list[int]):
        self.reset()
        logger.info(
            "eagle3: on_new_request req=%s prompt_tokens=%d",
            request_id[:8],
            len(prompt_tokens) if prompt_tokens else 0,
        )
        if self.model is not None and prompt_tokens:
            try:
                from mlx_lm.models.cache import KVCache

                with mx.stream(mx.default_stream(mx.gpu)):
                    input_ids = mx.array(prompt_tokens, mx.uint32)
                    self._draft_cache = [KVCache() for _ in self.model.layers]
                    self.model(input_ids[None], cache=self._draft_cache)
                    mx.eval(self._draft_cache)
                    logger.info(
                        "eagle3: prefill success, layers=%d",
                        len(self._draft_cache),
                    )
            except Exception as e:
                logger.warning("eagle3: prefill failed: %s", e)
                self._draft_cache = None

    def generate_draft_tokens(self, current_token: int) -> list[int]:
        if self.model is None or not self._loaded:
            return []
        if self._draft_cache is None:
            return []

        drafts = []
        token = current_token
        try:
            with mx.stream(mx.default_stream(mx.gpu)):
                for _ in range(self.config.num_draft):
                    input_ids = mx.array([token], mx.uint32)
                    logits = self.model(input_ids[None], cache=self._draft_cache)
                    logits = logits.squeeze(0).squeeze(0)

                    if self.config.temperature > 0:
                        from mlx_lm.sample_utils import make_sampler

                        sampler = make_sampler(temp=self.config.temperature)
                        next_token = sampler(logits)
                    else:
                        next_token = mx.argmax(logits)

                    mx.eval(next_token)
                    token = int(next_token)
                    drafts.append(token)

            self._total_drafts += len(drafts)
            return drafts
        except Exception as e:
            logger.warning("eagle3: generate failed: %s", e)
            self.reset()
            return []

    def record_accepted(self, n_accepted: int):
        self._total_accepted += n_accepted

    def get_stats(self) -> dict:
        rate = (
            self._total_accepted / self._total_drafts
            if self._total_drafts > 0
            else 0.0
        )
        return {
            "method": "eagle3",
            "draft_model_key": self.config.draft_model_key,
            "model_path": self.model_path,
            "target_family": self.target_family,
            "num_draft": self.config.num_draft,
            "total_drafts": self._total_drafts,
            "total_accepted": self._total_accepted,
            "acceptance_rate": rate,
            "loaded": self._loaded,
        }
