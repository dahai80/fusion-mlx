# SPDX-License-Identifier: Apache-2.0
import json
import logging
from dataclasses import dataclass, field

from .registry import get_spec_decoder, iter_spec_decoders

logger = logging.getLogger(__name__)


class SpeculativeConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SpeculativeConfig:
    method: str
    model: str | None = None
    num_speculative_tokens: int = 5
    tree_budget: int = 0
    raw: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = {"method": self.method}
        if self.model is not None:
            d["model"] = self.model
        if self.num_speculative_tokens != 5:
            d["num_speculative_tokens"] = self.num_speculative_tokens
        if self.tree_budget > 0:
            d["tree_budget"] = self.tree_budget
        d.update(self.raw)
        return json.dumps(d)


_ALLOWED_KEYS: dict[str, set[str]] = {
    "ddtree": {"method", "model", "num_speculative_tokens", "tree_budget"},
    "dflash": {"method", "model", "num_speculative_tokens", "block_size"},
    "mtp": {"method", "model", "num_speculative_tokens", "chain_or_tree"},
    "suffix": {
        "method",
        "num_speculative_tokens",
        "max_draft_tokens",
        "max_suffix_len",
        "min_confidence",
    },
    "dspark": {"method", "model", "num_speculative_tokens"},
    "vlm-mtp": {"method", "model", "num_speculative_tokens", "draft_block_size"},
}


def parse_speculative_config(value: str) -> SpeculativeConfig:
    if not value.strip():
        raise SpeculativeConfigError("empty speculative config")
    try:
        d = json.loads(value)
    except json.JSONDecodeError as e:
        raise SpeculativeConfigError(f"invalid JSON: {e}") from e
    if not isinstance(d, dict):
        raise SpeculativeConfigError("config must be a JSON object")
    method = d.get("method")
    if not method:
        raise SpeculativeConfigError("'method' key is required")
    plugin = get_spec_decoder(method)
    if plugin is None:
        available = sorted(p.method for p in iter_spec_decoders())
        raise SpeculativeConfigError(
            f"unknown method {method!r}; available: {available}"
        )
    resolved = plugin.method
    allowed = _ALLOWED_KEYS.get(resolved, set())
    if allowed:
        unknown = set(d.keys()) - allowed
        if unknown:
            logger.warning(
                "speculative config: unknown keys for %s: %s",
                resolved,
                sorted(unknown),
            )
    num_spec = d.get("num_speculative_tokens", 5)
    if not isinstance(num_spec, int) or num_spec < 1:
        raise SpeculativeConfigError(
            f"num_speculative_tokens must be a positive int, got {num_spec!r}"
        )
    tree_budget = d.get("tree_budget", 0)
    model = d.get("model")
    raw = {
        k: v
        for k, v in d.items()
        if k not in {"method", "model", "num_speculative_tokens", "tree_budget"}
    }
    return SpeculativeConfig(
        method=resolved,
        model=model,
        num_speculative_tokens=num_spec,
        tree_budget=tree_budget,
        raw=raw,
    )


def legacy_ddtree_config(
    *,
    num_speculative_tokens: int = 5,
    tree_budget: int = 0,
    model: str | None = None,
) -> SpeculativeConfig:
    return SpeculativeConfig(
        method="ddtree",
        model=model,
        num_speculative_tokens=num_speculative_tokens,
        tree_budget=tree_budget,
    )


def legacy_dflash_config(
    *,
    num_speculative_tokens: int = 5,
    model: str | None = None,
) -> SpeculativeConfig:
    return SpeculativeConfig(
        method="dflash",
        model=model,
        num_speculative_tokens=num_speculative_tokens,
    )


def legacy_mtp_config(
    *,
    num_speculative_tokens: int = 5,
    model: str | None = None,
) -> SpeculativeConfig:
    return SpeculativeConfig(
        method="mtp",
        model=model,
        num_speculative_tokens=num_speculative_tokens,
    )


def require_migrated_speculative_config(value: str) -> SpeculativeConfig:
    cfg = parse_speculative_config(value)
    if cfg.method in ("ddtree", "dflash"):
        logger.info(
            "speculative config: %s -> dflash (unified block-diffusion path)",
            cfg.method,
        )
    return cfg
