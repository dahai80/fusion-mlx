# SPDX-License-Identifier: Apache-2.0
# Rewritten Eagle3Speculator to use custom MLX-native model (model.py)
# instead of mlx_lm.load() which fails for EAGLE3's non-standard weight keys.
# Caller: fusion_mlx/engine_core._init_draft() when FUSION_SPEC_METHOD=eagle3
# Interface: DraftModelDecoder (load/on_new_request/generate_draft_tokens/record_accepted/get_stats)
# User instruction: "遇到问题就解决问题，没有模型就下载模型" — fix everything, produce real results
import logging
import os
import time
from dataclasses import dataclass

import mlx.core as mx

from .model import bind_target_embedding, create_eagle3_model

logger = logging.getLogger(__name__)

EAGLE3_DRAFT_MODELS = {
    "llama3.1-8b": {
        "hf_path": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "target_family": "llama3",
        "draft_vocab_size": 32000,
        "target_model": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
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
    # Phase-2 item 4: a small temperature on the draft sampler helps the
    # draft distribution match the target's, improving acceptance vs
    # greedy argmax (0.0). Override via FUSION_EAGLE3_DRAFT_TEMP.
    temperature: float = 0.1

    @classmethod
    def from_env(cls) -> "Eagle3DraftConfig":
        key = os.environ.get(ENV_DRAFT_MODEL, "llama3.1-8b")
        num_draft = int(os.environ.get(ENV_DRAFT_TOKENS, "5"))
        temp = float(os.environ.get(ENV_DRAFT_TEMP, "0.1"))
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

    # Map a target_family to the substrings that must appear in a
    # compatible target model name. Eagle3 draft weights are trained
    # against a specific family; running llama3-trained Eagle3 with a
    # Qwen target (or vice versa) produces garbage drafts.
    _FAMILY_MATCHERS = {
        "llama3": ("llama", "llama3", "llama-3"),
        "qwen3": ("qwen", "qwen3", "qwen-3"),
    }

    def is_compatible(self, target_model_name: str) -> bool:
        """Return True if the loaded target model matches the draft's family.

        Guards against silent garbage when an Eagle3 draft trained for
        one family is paired with a target from another (e.g. EAGLE3-LLaMA3
        against a Qwen model). Matching is case-insensitive substring on
        the family matchers above; the local path basename is checked too
        so a ``~/.fusion-mlx/models/.../Qwen3-...`` directory still matches.
        """
        family = self.target_family
        matchers = self._FAMILY_MATCHERS.get(family)
        if not matchers:
            # Unknown family — no matcher, allow (best-effort) but warn.
            logger.warning(
                "eagle3: no family matcher for target_family=%s, "
                "skipping compatibility guard",
                family,
            )
            return True
        name = (target_model_name or "").lower()
        # Also consider the basename of a local model dir.
        base = os.path.basename(name.rstrip("/"))
        hay = f"{name} {base}"
        compatible = any(m in hay for m in matchers)
        if not compatible:
            logger.warning(
                "eagle3: target model %r does not match family %r (expected "
                "one of %s) — disabling spec decode to avoid garbage drafts",
                target_model_name,
                family,
                matchers,
            )
        else:
            logger.info(
                "eagle3: target model %r compatible with family %r",
                target_model_name,
                family,
            )
        return compatible

    def load(self) -> bool:
        if self._loaded:
            return True
        try:
            t0 = time.perf_counter()
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

            self.model = create_eagle3_model(load_path)

            target_info = EAGLE3_DRAFT_MODELS.get(self.config.draft_model_key, {})
            target_model_hf = target_info.get("target_model", "")
            if target_model_hf:
                self._try_bind_target_embed(target_model_hf)

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
            import traceback

            traceback.print_exc()
            return False

    @property
    def capture_layers(self):
        return [8, 16, 31]

    def set_hidden_capture(self, hidden_capture):
        self._hidden_capture = hidden_capture

    def _build_hidden_from_capture(self, seq_len: int):
        if not hasattr(self, "_hidden_capture") or self._hidden_capture is None:
            return None
        captured = self._hidden_capture.get_prefill_captured()
        if not captured:
            return None
        sorted_ids = sorted(captured.keys())
        states = [captured[lid] for lid in sorted_ids]
        try:
            min_sl = min(s.shape[1] for s in states)
            states = [s[:, -min_sl:, :] for s in states]
            cat = mx.concatenate(states, axis=-1)
            projected = self.model.fc(cat)
            mx.eval(projected)
            self._hidden_capture.clear_prefill_captured()
            return projected
        except Exception as e:
            logger.warning("eagle3: _build_hidden_from_capture failed: %s", e)
            return None

    def _get_decode_hidden(self):
        if not hasattr(self, "_hidden_capture") or self._hidden_capture is None:
            return self._prefill_hidden
        captured = self._hidden_capture.get_captured()
        if not captured:
            return self._prefill_hidden
        sorted_ids = sorted(captured.keys())
        states = [captured[lid][:, -1:, :] for lid in sorted_ids]
        try:
            cat = mx.concatenate(states, axis=-1)
            projected = self.model.fc(cat)
            mx.eval(projected)
            return projected
        except Exception as e:
            logger.warning("eagle3: _get_decode_hidden failed: %s", e)
            return self._prefill_hidden

    def bind_target_embed_from_model(self, target_embed):
        if self.model is None:
            logger.warning("eagle3: no eagle3 model loaded, cannot bind embed")
            return
        try:
            if (
                hasattr(target_embed, "weight")
                and target_embed.weight.dtype != mx.uint32
            ):
                embed_w = target_embed.weight
                if embed_w.shape[-1] != self.model.hidden_size:
                    logger.warning(
                        "eagle3: target embed shape %s != hidden_size %d",
                        embed_w.shape,
                        self.model.hidden_size,
                    )
                    return
                bind_target_embedding(self.model, embed_w)
                return
            vocab = self.model.target_vocab_size
            chunk = 4096
            rows = []
            for start in range(0, vocab, chunk):
                end = min(start + chunk, vocab)
                ids = mx.arange(start, end, dtype=mx.uint32)
                emb = target_embed(ids)
                rows.append(emb)
            embed_w = mx.concatenate(rows, axis=0)
            bind_target_embedding(self.model, embed_w)
        except Exception as e:
            logger.warning("eagle3: bind_target_embed_from_model failed: %s", e)

    def _try_bind_target_embed(self, target_model_hf: str):
        target_local = os.path.expanduser(
            os.path.join("~/.fusion-mlx/models", target_model_hf)
        )
        if not os.path.isdir(target_local):
            logger.info(
                "eagle3: target model %s not found locally, skipping embed bind",
                target_local,
            )
            return
        try:
            import glob

            from safetensors import safe_open

            safetensor_files = sorted(
                glob.glob(os.path.join(target_local, "*.safetensors"))
            )
            if not safetensor_files:
                logger.info(
                    "eagle3: no safetensors in %s, skipping embed bind", target_local
                )
                return
            for sf in safetensor_files:
                with safe_open(sf, framework="numpy") as f:
                    for k in f:
                        if k == "model.embed_tokens.weight":
                            arr = f.get_tensor(k)
                            if arr.shape[-1] != self.model.hidden_size:
                                logger.info(
                                    "eagle3: target embed_tokens shape %s != hidden_size %d, skipping (quantized?)",
                                    arr.shape,
                                    self.model.hidden_size,
                                )
                                return
                            embed_w = mx.array(arr)
                            bind_target_embedding(self.model, embed_w)
                            return
            logger.info(
                "eagle3: embed_tokens not found in target model weights, using lm_head-derived init"
            )
        except Exception as e:
            logger.warning("eagle3: failed to bind target embed: %s", e)

    def reset(self):
        self._draft_cache = None
        self._prev_token = None
        self._prefill_hidden = None

    def on_new_request(self, request_id: str, prompt_tokens: list[int]):
        self.reset()
        logger.info(
            "eagle3: on_new_request req=%s prompt_tokens=%d",
            request_id[:8] if request_id else "none",
            len(prompt_tokens) if prompt_tokens else 0,
        )
        if self.model is not None and prompt_tokens:
            try:
                from mlx_lm.models.cache import KVCache

                with mx.stream(mx.default_stream(mx.gpu)):
                    hidden_state = self._build_hidden_from_capture(len(prompt_tokens))
                    if hidden_state is not None:
                        hs_sl = hidden_state.shape[1]
                        ids = prompt_tokens[-hs_sl:]
                    else:
                        ids = prompt_tokens
                    input_ids = mx.array(ids, mx.uint32)
                    self._draft_cache = [KVCache() for _ in self.model.layers]
                    self.model.forward_standalone(
                        input_ids[None],
                        cache=self._draft_cache,
                        hidden_state=hidden_state,
                    )
                    mx.eval(self._draft_cache)
                    if hidden_state is not None:
                        self._prefill_hidden = hidden_state[:, -1:, :]
                        mx.eval(self._prefill_hidden)
                    logger.info(
                        "eagle3: prefill success, layers=%d has_hidden=%s",
                        len(self._draft_cache),
                        hidden_state is not None,
                    )
            except Exception as e:
                logger.warning("eagle3: prefill failed: %s", e)
                import traceback

                traceback.print_exc()
                self._draft_cache = None

    def generate_draft_tokens(self, current_token: int) -> list[int]:
        if self.model is None or not self._loaded:
            return []
        if self._draft_cache is None:
            return []

        hidden_state = self._get_decode_hidden()
        drafts = []
        token = current_token
        try:
            with mx.stream(mx.default_stream(mx.gpu)):
                for _ in range(self.config.num_draft):
                    input_ids = mx.array([token], mx.uint32)
                    logits = self.model.forward_standalone(
                        input_ids[None],
                        cache=self._draft_cache,
                        hidden_state=hidden_state,
                    )
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
            import traceback

            traceback.print_exc()
            self.reset()
            return []

    def record_accepted(self, n_accepted: int):
        self._total_accepted += n_accepted

    def get_stats(self) -> dict:
        rate = (
            self._total_accepted / self._total_drafts if self._total_drafts > 0 else 0.0
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
