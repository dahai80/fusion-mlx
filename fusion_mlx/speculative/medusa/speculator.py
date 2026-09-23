# SPDX-License-Identifier: Apache-2.0
# Medusa v9 speculator — backbone-free parallel draft heads on post-norm hidden.
# Interface: DraftModelDecoder (load/on_new_request/generate_draft_tokens/
# record_accepted/get_stats/reset).
# Drafts K tokens in ONE forward (parallel heads, no autoregression).
# Hidden source: HiddenStateCapture on the LAST target layer (pre-norm output),
# then apply model.norm to get h_L36 (post-norm) — the representation lm_head
# expects. Adapter: h_j = h_L36 + down_j(relu(up_j(h_L36))), logits = h_j @ W.T.
import logging
import os
from collections import deque
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

ENV_DRAFT_TOKENS = "FUSION_MEDUSA_DRAFT_TOKENS"
ENV_WEIGHTS_PATH = "FUSION_MEDUSA_WEIGHTS"
ENV_RANK = "FUSION_MEDUSA_RANK"
ENV_ADAPTIVE_WINDOW = "FUSION_MEDUSA_ADAPTIVE_WINDOW"
ENV_ADAPTIVE_BREAK_EVEN = "FUSION_MEDUSA_ADAPTIVE_BREAK_EVEN"
ENV_ADAPTIVE_PROBE = "FUSION_MEDUSA_ADAPTIVE_PROBE"

DEFAULT_WEIGHTS = os.path.expanduser("~/.fusion-mlx/calibration/medusa_v9_L36.npz")
DEFAULT_K = 4
DEFAULT_RANK = 256
ADAPTIVE_WINDOW = int(os.environ.get(ENV_ADAPTIVE_WINDOW, "6"))
ADAPTIVE_BREAK_EVEN = float(os.environ.get(ENV_ADAPTIVE_BREAK_EVEN, "0.05"))
ADAPTIVE_PROBE = int(os.environ.get(ENV_ADAPTIVE_PROBE, "8"))


@dataclass
class MedusaConfig:
    num_draft: int = DEFAULT_K
    rank: int = DEFAULT_RANK
    weights_path: str = DEFAULT_WEIGHTS

    @classmethod
    def from_env(cls):
        k = int(os.environ.get(ENV_DRAFT_TOKENS, str(DEFAULT_K)))
        rank = int(os.environ.get(ENV_RANK, str(DEFAULT_RANK)))
        path = os.environ.get(ENV_WEIGHTS_PATH, DEFAULT_WEIGHTS)
        return cls(num_draft=k, rank=rank, weights_path=path)


class MedusaSpeculator:
    """Parallel-head draft model on post-norm target hidden state.

    No autoregressive draft loop — all K heads run from the SAME h_L36 in
    one batched matmul, so draft cost is O(1) GPU forward regardless of K
    (up/down are rank-256, vocab matmul is the dominant cost and runs once
    for all heads stacked).
    """

    def __init__(self, config: MedusaConfig | None = None):
        self.config = config or MedusaConfig.from_env()
        self.k_heads = self.config.num_draft
        self.rank = self.config.rank
        self.hidden_size = 0
        self._frozen_w = None
        self._norm = None
        self._ups = None
        self._downs = None
        self._hidden_capture = None
        self._loaded = False
        self._total_drafts = 0
        self._total_accepted = 0
        self._recent_accept: deque[tuple[int, int]] = deque(maxlen=ADAPTIVE_WINDOW)
        self._adaptive_paused = False
        self._skip_count = 0

    @property
    def model_path(self) -> str:
        return self.config.weights_path

    @property
    def capture_layers(self):
        return [self._target_num_layers - 1] if self._target_num_layers else []

    _target_num_layers = 0

    def set_target_num_layers(self, n: int) -> None:
        self._target_num_layers = n

    def set_hidden_capture(self, hidden_capture) -> None:
        self._hidden_capture = hidden_capture

    def load(self) -> bool:
        if self._loaded:
            return True
        path = self.config.weights_path
        if not os.path.exists(path):
            logger.warning("medusa: weights not found at %s, disabled", path)
            return False
        try:
            import numpy as np

            d = np.load(path)
            stored_k = int(d["k_heads"]) if "k_heads" in d else self.k_heads
            stored_rank = int(d["rank"]) if "rank" in d else self.rank
            if stored_k != self.k_heads:
                logger.info(
                    "medusa: adjusting k_heads %d->%d from weights",
                    self.k_heads,
                    stored_k,
                )
                self.k_heads = stored_k
            if stored_rank != self.rank:
                self.rank = stored_rank
            self._ups = []
            self._downs = []
            for j in range(self.k_heads):
                up_w = mx.array(d[f"head{j}_up.weight"]).astype(mx.float32)
                down_w = mx.array(d[f"head{j}_down.weight"]).astype(mx.float32)
                self._ups.append(mx.stop_gradient(up_w))
                self._downs.append(mx.stop_gradient(down_w))
            if "val_d1" in d:
                logger.info(
                    "medusa: loaded k=%d rank=%d val_d1=%.1f%% d2=%.1f%% d3=%.1f%% d4=%.1f%%",
                    self.k_heads,
                    self.rank,
                    float(d["val_d1"]) * 100,
                    float(d["val_d2"]) * 100,
                    float(d["val_d3"]) * 100,
                    float(d["val_d4"]) * 100,
                )
            mx.eval(self._ups, self._downs)
            self._loaded = True
            logger.info(
                "medusa: loaded %d heads rank=%d from %s",
                self.k_heads,
                self.rank,
                path,
            )
            return True
        except Exception as e:
            logger.warning("medusa: load failed: %s", e, exc_info=True)
            return False

    def bind_target(self, model) -> None:
        inner = model
        if hasattr(inner, "model"):
            inner = inner.model
        self._norm = getattr(inner, "norm", None)
        lm_head = getattr(model, "lm_head", None)
        if lm_head is None:
            logger.warning("medusa: no lm_head on target model")
            return
        try:
            w = lm_head.weight
            if hasattr(lm_head, "scales"):
                bits = getattr(lm_head, "bits", 4)
                group_size = getattr(lm_head, "group_size", 64)
                w = mx.dequantize(
                    w,
                    lm_head.scales,
                    lm_head.biases,
                    group_size=group_size,
                    bits=bits,
                )
                logger.info(
                    "medusa: dequantized lm_head bits=%d group=%d -> %s",
                    bits,
                    group_size,
                    w.shape,
                )
            self._frozen_w = mx.stop_gradient(w).astype(mx.float32)
            self.hidden_size = w.shape[1]
            mx.eval(self._frozen_w)
        except Exception as e:
            logger.warning("medusa: bind_target failed: %s", e, exc_info=True)

    def reset(self):
        self._recent_accept.clear()
        self._adaptive_paused = False
        self._skip_count = 0

    def on_new_request(self, request_id: str, prompt_tokens: list[int]):
        self.reset()
        if self._hidden_capture is not None:
            self._hidden_capture.on_new_request()
        logger.info(
            "medusa: on_new_request req=%s prompt_tokens=%d",
            request_id[:8] if request_id else "none",
            len(prompt_tokens) if prompt_tokens else 0,
        )

    def _should_skip_draft(self) -> bool:
        if len(self._recent_accept) < ADAPTIVE_WINDOW:
            return False
        total_a = sum(a for a, _ in self._recent_accept)
        total_t = sum(t for _, t in self._recent_accept)
        rate = total_a / total_t if total_t > 0 else 0.0
        if rate < ADAPTIVE_BREAK_EVEN:
            if not self._adaptive_paused:
                self._adaptive_paused = True
                self._skip_count = 0
                logger.info(
                    "medusa: adaptive pause — acceptance %.1f%% < %.1f%%",
                    rate * 100,
                    ADAPTIVE_BREAK_EVEN * 100,
                )
            return True
        if self._adaptive_paused:
            logger.info(
                "medusa: adaptive resume — acceptance %.1f%% recovered",
                rate * 100,
            )
            self._adaptive_paused = False
            self._skip_count = 0
        return False

    def _get_post_norm_hidden(self) -> mx.array | None:
        if self._hidden_capture is None or self._norm is None:
            return None
        captured = self._hidden_capture.get_captured()
        if not captured:
            return None
        last_idx = max(captured.keys())
        h = captured[last_idx]
        h_last = h[:, -1:, :]
        h_post = self._norm(h_last)
        return h_post

    def generate_draft_tokens(self, current_token: int) -> list[int]:
        if not self._loaded or self._frozen_w is None:
            return []
        if self._should_skip_draft():
            self._skip_count += 1
            if self._skip_count % ADAPTIVE_PROBE != 0:
                return []
            logger.debug("medusa: adaptive re-probe at skip=%d", self._skip_count)
        h_post = self._get_post_norm_hidden()
        if h_post is None:
            return []
        try:
            with mx.stream(mx.default_stream(mx.gpu)):
                h = h_post.reshape(-1)
                hs = []
                for j in range(self.k_heads):
                    up = mx.matmul(h, self._ups[j].T)
                    up = nn.relu(up)
                    delta = mx.matmul(up, self._downs[j].T)
                    hs.append(h + delta)
                h_flat = mx.stack(hs, axis=0)
                logits = mx.matmul(h_flat, self._frozen_w.T)
                draft_toks = mx.argmax(logits, axis=-1)
                mx.eval(draft_toks)
            drafts = draft_toks.tolist()
            self._total_drafts += len(drafts)
            return drafts
        except Exception as e:
            logger.warning("medusa: generate failed: %s", e, exc_info=True)
            self.reset()
            return []

    def record_accepted(self, n_accepted: int):
        self._total_accepted += n_accepted
        self._recent_accept.append((n_accepted, self.k_heads))
        if not self._adaptive_paused and len(self._recent_accept) >= ADAPTIVE_WINDOW:
            total_a = sum(a for a, _ in self._recent_accept)
            total_t = sum(t for _, t in self._recent_accept)
            rate = total_a / total_t if total_t > 0 else 0.0
            if rate < ADAPTIVE_BREAK_EVEN:
                self._adaptive_paused = True
                self._skip_count = 0
                logger.info(
                    "medusa: adaptive pause (eager) — acceptance %.1f%% < %.1f%%",
                    rate * 100,
                    ADAPTIVE_BREAK_EVEN * 100,
                )

    def get_stats(self) -> dict:
        rate = (
            self._total_accepted / self._total_drafts if self._total_drafts > 0 else 0.0
        )
        return {
            "method": "medusa",
            "weights_path": self.config.weights_path,
            "k_heads": self.k_heads,
            "rank": self.rank,
            "total_drafts": self._total_drafts,
            "total_accepted": self._total_accepted,
            "acceptance_rate": rate,
            "loaded": self._loaded,
            "paused": self._adaptive_paused,
        }
