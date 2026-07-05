# SPDX-License-Identifier: Apache-2.0
import logging
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RAPID_EXTRA_EOS_ATTR = "_rapid_extra_eos_token_ids"


def is_gemma4_model(model_path: str) -> bool:
    return "gemma" in model_path.lower() and "4" in model_path.lower()


def is_harmony_model(model_path: str) -> bool:
    return "harmony" in model_path.lower()


def is_qwen3_model(model_name: str) -> bool:
    model_lower = model_name.lower()
    return "qwen3" in model_lower or "Qwen3" in model_name


def apply_qwen3_fix(
    tokenizer_config: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    if is_qwen3_model(model_name):
        tokenizer_config["eos_token"] = "<|im_end|>"
        logger.debug("Qwen3 detected: setting eos_token to <|im_end|>")
    return tokenizer_config


def _read_json_file(path: Path) -> dict[str, Any] | None:
    import json

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Failed to read %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _find_tokenizer_json(
    tokenizer: Any,
    model_path: str | Path | None = None,
) -> Path | None:
    candidates: list[str | Path] = []
    if model_path:
        candidates.append(model_path)
    tokenizer_path = getattr(tokenizer, "name_or_path", None)
    if tokenizer_path:
        candidates.append(tokenizer_path)
    for candidate in candidates:
        candidate_path = Path(candidate).expanduser()
        tokenizer_file = candidate_path / "tokenizer.json"
        if tokenizer_file.exists():
            return tokenizer_file
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(str(candidate), "tokenizer.json")
        except Exception:
            cached = None
        if cached and isinstance(cached, str):
            cached_path = Path(cached)
            if cached_path.exists():
                return cached_path
    return None


@lru_cache(maxsize=128)
def _detokenizer_factory_from_tokenizer_json(
    tokenizer_file: str,
) -> Callable[[Any], Any] | None:
    tokenizer_content = _read_json_file(Path(tokenizer_file))
    if not tokenizer_content or "decoder" not in tokenizer_content:
        return None
    try:
        from mlx_lm.tokenizer_utils import (
            BPEStreamingDetokenizer,
            SPMStreamingDetokenizer,
            _is_bpe_decoder,
            _is_spm_decoder,
            _is_spm_decoder_no_space,
        )
    except ImportError:
        return None
    decoder = tokenizer_content["decoder"]
    if _is_spm_decoder(decoder):
        return SPMStreamingDetokenizer
    if _is_spm_decoder_no_space(decoder):
        from functools import partial

        return partial(SPMStreamingDetokenizer, trim_space=False)
    if _is_bpe_decoder(decoder):
        return BPEStreamingDetokenizer
    return None


class _CompatNaiveStreamingDetokenizer:
    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer
        self._tokenizer.decode([0])

    def add(self, token: int) -> None:
        pass

    def finalize(self) -> str:
        return ""

    def __repr__(self) -> str:
        return "<CompatNaiveStreamingDetokenizer>"


def create_streaming_detokenizer(
    tokenizer: Any,
    model_path: str | Path | None = None,
) -> Any | None:
    has_existing_attr = True
    try:
        detokenizer = tokenizer.detokenizer
    except AttributeError:
        has_existing_attr = False
        detokenizer = None
    except Exception as exc:
        has_existing_attr = False
        detokenizer = None
        logger.debug("Failed to read tokenizer.detokenizer: %s", exc)

    if detokenizer is not None:
        return detokenizer

    tokenizer_file = _find_tokenizer_json(tokenizer, model_path)
    if tokenizer_file is not None:
        factory = _detokenizer_factory_from_tokenizer_json(str(tokenizer_file))
        if factory is not None:
            try:
                return factory(tokenizer)
            except Exception as exc:
                logger.debug(
                    "Failed to create decoder-aware detokenizer from %s: %s",
                    tokenizer_file,
                    exc,
                )

    if has_existing_attr:
        return None

    try:
        from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer
    except ImportError:
        return None

    try:
        return NaiveStreamingDetokenizer(tokenizer)
    except Exception as exc:
        logger.debug("Failed to create naive streaming detokenizer: %s", exc)

    try:
        return _CompatNaiveStreamingDetokenizer(tokenizer)
    except Exception as compat_exc:
        logger.debug(
            "Failed to create compatibility naive streaming detokenizer: %s",
            compat_exc,
        )
        return None


def get_tokenizer_config(
    model_name: str,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    config: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if is_qwen3_model(model_name):
        config["eos_token"] = "<|im_end|>"
        logger.debug("Qwen3 detected: setting eos_token to <|im_end|>")
    return config
