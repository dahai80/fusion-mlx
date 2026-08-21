# SPDX-License-Identifier: Apache-2.0
import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpecDecoderPlugin:
    method: str
    description: str
    config_enabled: bool = True
    legacy_hint: str | None = None
    aliases: tuple[str, ...] = ()


_PLUGINS: dict[str, SpecDecoderPlugin] = {}
_PLUGINS_LOCK = threading.Lock()


def register_spec_decoder(plugin: SpecDecoderPlugin) -> None:
    with _PLUGINS_LOCK:
        if plugin.method in _PLUGINS:
            existing = _PLUGINS[plugin.method]
            if existing == plugin:
                return
            logger.warning(
                "spec-decode registry: overwriting %s (%s -> %s)",
                plugin.method,
                existing.description,
                plugin.description,
            )
        _PLUGINS[plugin.method] = plugin
        for alias in plugin.aliases:
            if alias in _PLUGINS and _PLUGINS[alias] != plugin:
                logger.warning(
                    "spec-decode registry: alias %s already mapped to %s",
                    alias,
                    _PLUGINS[alias].method,
                )
            _PLUGINS[alias] = plugin


def get_spec_decoder(name: str) -> SpecDecoderPlugin | None:
    with _PLUGINS_LOCK:
        return _PLUGINS.get(name)


def iter_spec_decoders() -> list[SpecDecoderPlugin]:
    with _PLUGINS_LOCK:
        seen = set()
        result = []
        for plugin in _PLUGINS.values():
            if plugin.method not in seen:
                seen.add(plugin.method)
                result.append(plugin)
        return result


register_spec_decoder(
    SpecDecoderPlugin(
        method="ddtree",
        description="Draft-then-verify tree drafter (z-lab DFlash)",
        config_enabled=True,
        aliases=("dflash", "block-diffusion"),
    )
)

register_spec_decoder(
    SpecDecoderPlugin(
        method="dflash2",
        description="DFlash2 block-diffusion speculative decoding (z-lab dflash pkg, Qwen3.8)",
        config_enabled=True,
        aliases=("block-diffusion-2",),
    )
)

register_spec_decoder(
    SpecDecoderPlugin(
        method="mtp",
        description="Multi-Token Prediction (chain/tree) via model MTP heads",
        config_enabled=True,
        aliases=("multi-token", "mtp-chain", "mtp-tree"),
    )
)

register_spec_decoder(
    SpecDecoderPlugin(
        method="suffix",
        description="Suffix-decoding adaptive n-gram drafter",
        config_enabled=True,
        aliases=("ngram", "prompt-lookup", "pld"),
    )
)

register_spec_decoder(
    SpecDecoderPlugin(
        method="dspark",
        description="DSpark speculative decoding (DistilDraft)",
        config_enabled=True,
        aliases=("distil-draft",),
    )
)

register_spec_decoder(
    SpecDecoderPlugin(
        method="vlm-mtp",
        description="VLM MTP drafter (gemma4 assistant)",
        config_enabled=True,
        aliases=("gemma4-mtp",),
    )
)

register_spec_decoder(
    SpecDecoderPlugin(
        method="eagle3",
        description="EAGLE3 speculative decoding (1-layer drafter, llama3/qwen3)",
        config_enabled=True,
        aliases=("eagle", "eagle3-llama", "eagle3-qwen"),
    )
)

register_spec_decoder(
    SpecDecoderPlugin(
        method="dfly",
        description="DFly block-parallel drafter (DFlash + hidden correction, Hy3-native)",
        config_enabled=True,
        aliases=("dfly-hy3", "hy3-dfly"),
    )
)
