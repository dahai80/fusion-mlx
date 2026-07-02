# SPDX-License-Identifier: Apache-2.0
"""N-gram speculative decoding predictor.

CPU-side token prediction based on n-gram frequency analysis.
Zero GPU overhead — predictions are verified against the target model
in a single forward pass (same as draft-model spec decode).

For repetitive content (code, structured text, JSON), n-gram matching
achieves 40-60% acceptance rates, yielding 1.5-2x throughput gains.
"""

import logging
import os
from collections import defaultdict

logger = logging.getLogger(__name__)

NGRAM_ORDER = int(os.environ.get("FUSION_NGRAM_ORDER", "5"))
NGRAM_NUM_DRAFT = int(os.environ.get("FUSION_NGRAM_NUM_DRAFT", "3"))
NGRAM_MAX_ENTRIES = int(os.environ.get("FUSION_NGRAM_MAX_ENTRIES", "65536"))
NGRAM_MIN_HITS = int(os.environ.get("FUSION_NGRAM_MIN_HITS", "2"))


class NGramPredictor:
    """N-gram based token predictor for speculative decoding.

    Maintains a frequency table of n-gram → next-token mappings.
    For a context [t1, t2, ..., tn], looks up the most frequent
    continuation token. Chains predictions for K draft tokens.

    Thread safety: only called from the engine/step thread.
    """

    def __init__(
        self,
        order: int = NGRAM_ORDER,
        num_draft: int = NGRAM_NUM_DRAFT,
        max_entries: int = NGRAM_MAX_ENTRIES,
        min_hits: int = NGRAM_MIN_HITS,
    ):
        self.order = order
        self.num_draft = num_draft
        self.max_entries = max_entries
        self.min_hits = min_hits

        # n-gram tables: order → {tuple(tokens) → {next_token → count}}
        self._tables: dict[int, dict[tuple, dict[int, int]]] = {}
        for n in range(2, order + 1):
            self._tables[n] = defaultdict(lambda: defaultdict(int))

        # Recent token buffer for building n-grams
        self._history: list[int] = []
        self._total_predictions = 0
        self._total_accepted = 0
        self._total_entries = 0

    def reset(self):
        """Reset history for a new request."""
        self._history.clear()

    def add_token(self, token: int):
        """Record a token and update n-gram tables."""
        self._history.append(token)

        # Update n-gram tables for all orders
        for n, table in self._tables.items():
            if len(self._history) > n:
                key = tuple(self._history[-(n + 1):-1])
                next_tok = self._history[-1]
                entry = table[key]
                entry[next_tok] += 1
                if len(entry) == 1:
                    self._total_entries += 1

        # Evict if over capacity
        if self._total_entries > self.max_entries:
            self._evict()

    def add_tokens(self, tokens: list[int]):
        """Record multiple tokens."""
        for t in tokens:
            self.add_token(t)

    def predict(self, context: list[int] | None = None) -> list[int]:
        """Predict next K tokens using n-gram matching.

        Args:
            context: Optional context tokens. If None, uses internal history.

        Returns:
            List of predicted token IDs (may be shorter than num_draft).
        """
        if context is None:
            context = self._history

        if len(context) < 2:
            return []

        drafts = []
        extended = list(context)

        for _ in range(self.num_draft):
            best_token = None
            best_count = 0

            # Try from highest order down to 2
            for n in range(min(self.order, len(extended)), 1, -1):
                key = tuple(extended[-n:])
                table = self._tables.get(n)
                if table is None:
                    continue
                entry = table.get(key)
                if entry is None:
                    continue

                # Find most frequent continuation
                for tok, count in entry.items():
                    if count > best_count:
                        best_count = count
                        best_token = tok

                if best_count >= self.min_hits:
                    break

            if best_token is None or best_count < self.min_hits:
                break

            drafts.append(best_token)
            extended.append(best_token)

        self._total_predictions += len(drafts)
        return drafts

    def record_accepted(self, n_accepted: int):
        """Record how many draft tokens were accepted."""
        self._total_accepted += n_accepted

    def _evict(self):
        """Evict low-frequency entries to stay under capacity."""
        target = self.max_entries // 2
        removed = 0
        for n, table in self._tables.items():
            keys_to_remove = []
            for key, entry in table.items():
                total = sum(entry.values())
                if total <= 1:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del table[key]
                removed += 1
                if removed >= target:
                    break
            if removed >= target:
                break
        self._total_entries -= removed
        logger.debug("ngram: evicted %d entries, total=%d", removed, self._total_entries)

    def get_stats(self) -> dict:
        rate = (
            self._total_accepted / self._total_predictions
            if self._total_predictions > 0
            else 0.0
        )
        return {
            "order": self.order,
            "num_draft": self.num_draft,
            "total_entries": self._total_entries,
            "total_predictions": self._total_predictions,
            "total_accepted": self._total_accepted,
            "acceptance_rate": rate,
            "history_len": len(self._history),
        }
