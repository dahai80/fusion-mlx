# SPDX-License-Identifier: Apache-2.0
import json
import logging
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from .chat_templates import DEFAULT_CHATML_TEMPLATE, NEMOTRON_CHAT_TEMPLATE

logger = logging.getLogger(__name__)

RAPID_EXTRA_EOS_ATTR = "_rapid_extra_eos_token_ids"

FALLBACK_MODELS = [
    "nemotron",
    "NVIDIA-Nemotron",
]


def _needs_tokenizer_fallback(model_name: str) -> bool:
    model_lower = model_name.lower()
    return any(pattern.lower() in model_lower for pattern in FALLBACK_MODELS)


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


_BYTE_LEVEL_MOJIBAKE_MARKERS: tuple[str, ...] = (
    "Ġ",
    "Ċ",
    "ĉ",
)

_METASPACE_MARKER = "▁"


def _decoder_has_metaspace_replace(decoder) -> bool:
    try:
        state_raw = decoder.__getstate__()
    except Exception:
        return False
    try:
        state = json.loads(state_raw)
    except Exception:
        return False

    def _walk(node) -> bool:
        if not isinstance(node, dict):
            return False
        ntype = node.get("type")
        if ntype == "Replace":
            pattern = node.get("pattern") or {}
            content = node.get("content", "")
            pattern_str = pattern.get("String") or pattern.get("Regex") or ""
            if pattern_str == _METASPACE_MARKER and content == " ":
                return True
        if ntype == "Sequence":
            for child in node.get("decoders", []) or []:
                if _walk(child):
                    return True
        return False

    return _walk(state)


def repair_byte_level_decoder(tokenizer) -> bool:
    if tokenizer is None:
        return False
    candidates = [tokenizer]
    if hasattr(tokenizer, "_tokenizer"):
        candidates.append(tokenizer._tokenizer)
    inner = next(
        (c for c in candidates if hasattr(c, "backend_tokenizer")),
        None,
    )
    if inner is None:
        return False
    backend = inner.backend_tokenizer
    if _decoder_has_metaspace_replace(backend.decoder):
        try:
            spaced_ids = inner.encode("a b c", add_special_tokens=False)
            spaced_tokens = inner.convert_ids_to_tokens(spaced_ids)
        except Exception:
            spaced_tokens = []
        if any(isinstance(t, str) and _METASPACE_MARKER in t for t in spaced_tokens):
            logger.debug(
                "repair_byte_level_decoder: skipping %s — decoder has "
                "load-bearing Replace('%s', ' ') step (hybrid "
                "SentencePiece-metaspace tokenizer)",
                type(inner).__name__,
                _METASPACE_MARKER,
            )
            return False
    probe_id: int | None = None
    probe_pretty: str | None = None
    try:
        vocab = inner.get_vocab()
    except Exception:
        return False
    for pretty, tid in sorted(vocab.items(), key=lambda kv: kv[1]):
        if not isinstance(pretty, str):
            continue
        if any(pretty.startswith(m) for m in _BYTE_LEVEL_MOJIBAKE_MARKERS):
            probe_id = tid
            probe_pretty = pretty
            break
    if probe_id is None:
        return False
    try:
        decoded = inner.decode([probe_id], skip_special_tokens=False)
    except Exception:
        return False
    if not any(m in decoded for m in _BYTE_LEVEL_MOJIBAKE_MARKERS):
        return False
    original_decoder = backend.decoder
    try:
        from tokenizers import decoders as _decoders

        backend.decoder = _decoders.ByteLevel()
    except Exception as exc:
        logger.warning(
            "repair_byte_level_decoder: failed to swap decoder on %s: %s",
            type(inner).__name__,
            exc,
        )
        return False
    try:
        verify = inner.decode([probe_id], skip_special_tokens=False)
    except Exception:
        verify = decoded
    if any(m in verify for m in _BYTE_LEVEL_MOJIBAKE_MARKERS):
        try:
            backend.decoder = original_decoder
        except Exception as exc:
            logger.warning(
                "repair_byte_level_decoder: could not restore original "
                "decoder on %s after failed verification: %s",
                type(inner).__name__,
                exc,
            )
        logger.warning(
            "repair_byte_level_decoder: swap did not clear mojibake on %s "
            "(probe id=%d pretty=%r decoded=%r); restored original decoder",
            type(inner).__name__,
            probe_id,
            probe_pretty,
            verify,
        )
        return False
    try:
        spaced_ids = inner.encode("a b c", add_special_tokens=False)
        spaced_decoded = inner.decode(spaced_ids, skip_special_tokens=False)
    except Exception:
        spaced_decoded = ""
    if _METASPACE_MARKER in spaced_decoded:
        try:
            backend.decoder = original_decoder
        except Exception as exc:
            logger.warning(
                "repair_byte_level_decoder: could not restore original "
                "decoder on %s after spaced-sample verification failed: %s",
                type(inner).__name__,
                exc,
            )
        logger.warning(
            "repair_byte_level_decoder: post-swap spaced-sample decode "
            "leaked metaspace marker on %s (encode('a b c') -> %r); "
            "restored original decoder",
            type(inner).__name__,
            spaced_decoded,
        )
        return False
    logger.info(
        "repair_byte_level_decoder: swapped %s.backend_tokenizer.decoder to "
        "ByteLevel (probe id=%d pretty=%r -> decoded=%r)",
        type(inner).__name__,
        probe_id,
        probe_pretty,
        verify,
    )
    return True


def augment_eos_token_ids_from_generation_config(
    tokenizer, model_path_or_name: str
) -> None:
    from .generation_config import load_generation_config_eos_ids

    extras = load_generation_config_eos_ids(model_path_or_name)
    if not extras:
        return
    wrapper_set = getattr(tokenizer, "_eos_token_ids", None)
    if isinstance(wrapper_set, set):
        before = set(wrapper_set)
        wrapper_set.update(extras)
        added = sorted(set(wrapper_set) - before)
        if added:
            logger.info(
                "augment_eos: added %s to TokenizerWrapper stop set for %s",
                added,
                model_path_or_name,
            )
        return
    try:
        existing = getattr(tokenizer, RAPID_EXTRA_EOS_ATTR, None) or ()
        merged_set = set(int(x) for x in existing) | set(extras)
        merged = tuple(sorted(merged_set))
        setattr(tokenizer, RAPID_EXTRA_EOS_ATTR, merged)
        logger.info(
            "augment_eos: set %s=%s on %s for %s",
            RAPID_EXTRA_EOS_ATTR,
            list(merged),
            type(tokenizer).__name__,
            model_path_or_name,
        )
    except Exception as exc:
        logger.debug(
            "augment_eos: could not stash extras on %s (%s)",
            type(tokenizer).__name__,
            exc,
        )


def _apply_chat_template_sidecar(model_path: Path, tokenizer) -> bool:
    if getattr(tokenizer, "chat_template", None):
        return False
    jinja_path = model_path / "chat_template.jinja"
    if jinja_path.exists():
        tokenizer.chat_template = jinja_path.read_text(encoding="utf-8-sig")
        logger.info("Chat template loaded from chat_template.jinja sidecar")
        return True
    json_path = model_path / "chat_template.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                "Found chat_template.json at %s but failed to parse: %s",
                json_path,
                e,
            )
            return False
        template = data.get("chat_template")
        if isinstance(template, str) and template:
            tokenizer.chat_template = template
            logger.info("Chat template loaded from chat_template.json sidecar")
            return True
        logger.warning(
            "chat_template.json at %s has no 'chat_template' string key; "
            "got keys=%s",
            json_path,
            list(data.keys()),
        )
    return False


def _resolve_model_path(model_name: str) -> Path | None:
    local = Path(model_name)
    if local.is_dir():
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_name))
    except Exception as e:
        logger.debug("_resolve_model_path(%s) failed: %s", model_name, e)
        return None


_VENDORED_MODEL_TYPES = {"deepseek_v4"}


def _register_vendored_archs() -> None:
    import sys

    if "mlx_lm.models.deepseek_v4" not in sys.modules:
        try:
            from ..models import deepseek_v4 as _ds_v4

            sys.modules.setdefault("mlx_lm.models.deepseek_v4", _ds_v4)
        except Exception as e:
            logger.debug("deepseek_v4 vendored module unavailable: %s", e)


def _is_vendored_arch_model(model_name: str) -> bool:
    try:
        local = Path(model_name)
        if local.is_dir():
            config_path = local / "config.json"
        else:
            from huggingface_hub import hf_hub_download

            config_path = Path(
                hf_hub_download(repo_id=model_name, filename="config.json")
            )
        if not config_path.exists():
            return False
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("model_type") in _VENDORED_MODEL_TYPES
    except Exception as e:
        logger.debug("_is_vendored_arch_model(%s) failed: %s", model_name, e)
        return False


def _post_load_ubc_evict(model_name: str) -> None:
    import sys as _sys

    if _sys.platform != "darwin":
        return
    try:
        from ..runtime.ubc_evict import ubc_evict_paths

        model_path = _resolve_model_path(model_name)
        if model_path is None:
            return
        shards = sorted(str(p) for p in model_path.glob("*.safetensors"))
        if not shards:
            return
        ubc_evict_paths(shards)
    except Exception as e:
        logger.debug("Defect 4 post-load UBC evict skipped (non-fatal): %s", e)


def _read_json_file(path: Path) -> dict[str, Any] | None:
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


def _read_num_mtp_layers(config: dict) -> int:
    n = config.get("num_nextn_predict_layers", 0)
    if n == 0:
        n = config.get("text_config", {}).get("num_nextn_predict_layers", 0)
    return n


def _try_inject_mtp(model, model_path, config):
    num = _read_num_mtp_layers(config)
    if num > 0:
        try:
            from ..speculative.mtp.qwen3_5_inject import inject_mtp_support
        except ImportError:
            logger.debug("MTP inject module not available")
            return
        if config.get("num_nextn_predict_layers", 0) == 0:
            config = {**config, "num_nextn_predict_layers": num}
        inject_mtp_support(model, model_path, config)


def _try_inject_mtp_post_load(model, model_name):
    from mlx_lm.utils import _download

    model_path = _download(model_name)
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return
    with open(config_path) as f:
        config = json.load(f)
    num_mtp = _read_num_mtp_layers(config)
    if num_mtp > 0 and getattr(model, "mtp", None) is None:
        mtp_file = Path(model_path) / "model-mtp.safetensors"
        if mtp_file.exists():
            logger.info(
                "[MTP] Found MTP config (layers=%d) and weights, injecting...",
                num_mtp,
            )
            _try_inject_mtp(model, model_path, config)
        else:
            logger.info(
                "[MTP] Config has num_nextn_predict_layers=%d "
                "but model-mtp.safetensors not found, skipping MTP.",
                num_mtp,
            )


def load_model_with_fallback(model_name: str, tokenizer_config: dict = None):
    result = _load_model_with_fallback_impl(model_name, tokenizer_config)
    _post_load_ubc_evict(model_name)
    return result


def _load_model_with_fallback_impl(model_name: str, tokenizer_config: dict = None):
    from mlx_lm import load

    _register_vendored_archs()
    tokenizer_config = tokenizer_config or {}

    if _needs_tokenizer_fallback(model_name):
        logger.info(
            "Model %s requires tokenizer fallback, loading directly...",
            model_name,
        )
        return _load_with_tokenizer_fallback(model_name)

    if _is_vendored_arch_model(model_name):
        logger.info(
            "Model %s uses a vendored architecture, "
            "skipping AutoConfig path and loading directly...",
            model_name,
        )
        return _load_with_tokenizer_fallback(model_name)

    try:
        from ..models.gemma4_text import is_gemma4_model as _is_g4
        has_gemma4 = True
    except ImportError:
        has_gemma4 = False
        _is_g4 = is_gemma4_model

    if has_gemma4 and _is_g4(model_name):
        try:
            model, tokenizer = load(model_name, tokenizer_config=tokenizer_config)
            logger.info("Gemma 4 loaded natively via mlx-lm")
            if not getattr(tokenizer, "chat_template", None):
                mp = _resolve_model_path(model_name)
                if mp is not None:
                    _apply_chat_template_sidecar(mp, tokenizer)
            augment_eos_token_ids_from_generation_config(tokenizer, model_name)
            repair_byte_level_decoder(tokenizer)
            return model, tokenizer
        except Exception as e:
            try:
                from ..models.gemma4_text import load_gemma4_text

                logger.info(
                    "Gemma 4 native load failed (%s), "
                    "falling back to text-only wrapper (legacy mlx-lm)",
                    e,
                )
                return load_gemma4_text(model_name, tokenizer_config)
            except ImportError:
                raise

    try:
        model, tokenizer = load(model_name, tokenizer_config=tokenizer_config)
        _try_inject_mtp_post_load(model, model_name)
        if not getattr(tokenizer, "chat_template", None):
            mp = _resolve_model_path(model_name)
            if mp is not None:
                _apply_chat_template_sidecar(mp, tokenizer)
        augment_eos_token_ids_from_generation_config(tokenizer, model_name)
        repair_byte_level_decoder(tokenizer)
        return model, tokenizer
    except ValueError as e:
        if (
            "TokenizersBackend" in str(e)
            or "Tokenizer class" in str(e)
            or "does not recognize this architecture" in str(e)
        ):
            logger.warning("Standard tokenizer loading failed, using fallback: %s", e)
            return _load_with_tokenizer_fallback(model_name)
        elif "parameters not in model" in str(e) or (
            "Missing" in str(e) and "parameters" in str(e)
        ):
            logger.warning(
                "Model has extra/missing parameters (likely VLM / MTP weights), "
                "retrying with strict=False: %s",
                e,
            )
            return _load_strict_false(model_name, tokenizer_config)
        else:
            raise


def _load_strict_false(model_name: str, tokenizer_config: dict = None):
    from mlx_lm.utils import load_model, load_tokenizer

    local_path = Path(model_name)
    if local_path.is_dir():
        model_path = local_path
    else:
        from huggingface_hub import snapshot_download

        model_path = Path(snapshot_download(model_name))

    model, config = load_model(model_path, strict=False)
    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config or {},
        eos_token_ids=config.get("eos_token_id", None),
    )
    _try_inject_mtp(model, model_path, config)
    _apply_chat_template_sidecar(model_path, tokenizer)
    augment_eos_token_ids_from_generation_config(tokenizer, str(model_path))
    repair_byte_level_decoder(tokenizer)
    return model, tokenizer


def _load_with_tokenizer_fallback(model_name: str):
    from mlx_lm.utils import load_model

    logger.info("Loading with tokenizer fallback...")

    local_path = Path(model_name)
    if local_path.is_dir():
        model_path = local_path
    else:
        from huggingface_hub import snapshot_download

        model_path = Path(snapshot_download(model_name))

    model, _ = load_model(model_path)

    tokenizer_json = model_path / "tokenizer.json"
    if tokenizer_json.exists():
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        logger.info("Loading tokenizer from tokenizer.json")
        base_tokenizer = Tokenizer.from_file(str(tokenizer_json))

        tokenizer_config_path = model_path / "tokenizer_config.json"
        bos_token = "<s>"
        eos_token = "</s>"
        unk_token = "<unk>"
        chat_template = None

        if tokenizer_config_path.exists():
            with open(tokenizer_config_path) as f:
                config = json.load(f)
                bos_token = config.get("bos_token", bos_token)
                eos_token = config.get("eos_token", eos_token)
                unk_token = config.get("unk_token", unk_token)
                chat_template = config.get("chat_template")

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=base_tokenizer,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token="<pad>",
        )

        if chat_template:
            tokenizer.chat_template = chat_template
            logger.info("Chat template loaded from tokenizer_config.json")
        elif _apply_chat_template_sidecar(model_path, tokenizer):
            pass
        elif _needs_tokenizer_fallback(model_name):
            tokenizer.chat_template = NEMOTRON_CHAT_TEMPLATE
            logger.info("Using official Nemotron chat template with thinking support")
        else:
            tokenizer.chat_template = DEFAULT_CHATML_TEMPLATE
            logger.info("Using default ChatML chat template")

        repair_byte_level_decoder(tokenizer)
        logger.info("Tokenizer loaded via fallback successfully")
        return model, tokenizer
    else:
        raise ValueError(f"No tokenizer.json found in {model_path}")
