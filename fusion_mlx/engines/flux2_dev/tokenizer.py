import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_TOKENIZER_REPO = "diffusers/FLUX.2-dev-bnb-4bit"
_TOKENIZER_SUBDIR = "tokenizer"
_DEFAULT_MAX_LENGTH = 512


def _ensure_mirror_endpoint():
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logger.info(
            "flux2_dev tokenizer: set HF_ENDPOINT=%s", os.environ["HF_ENDPOINT"]
        )


def load_mistral_tokenizer(max_length=_DEFAULT_MAX_LENGTH):
    _ensure_mirror_endpoint()
    from huggingface_hub import snapshot_download
    from mflux.models.common.tokenizer.tokenizer_loader import LanguageTokenizer
    from transformers import AutoTokenizer

    local_root = os.path.expanduser("~/.fusion-mlx/models")
    tok_dir = snapshot_download(
        repo_id=_TOKENIZER_REPO,
        allow_patterns=[f"{_TOKENIZER_SUBDIR}/*"],
        cache_dir=local_root,
    )
    tok_path = Path(tok_dir) / _TOKENIZER_SUBDIR
    logger.info("flux2_dev tokenizer: loading from %s", tok_path)
    hf_tokenizer = AutoTokenizer.from_pretrained(str(tok_path), trust_remote_code=True)
    # Mistral chat template defaults to left-padding, but the text encoder
    # applies causal attention: left-pad rows are fully-masked -> NaN that
    # leaks into real positions via causal keys. Force right-padding so real
    # tokens occupy positions 0..N and RoPE positions align. Matches the
    # Qwen3 text-encoder assumption mflux Flux2Klein relies on.
    hf_tokenizer.padding_side = "right"
    logger.info(
        "flux2_dev tokenizer: forced padding_side=right (causal-mask NaN guard)"
    )
    if getattr(hf_tokenizer, "chat_template", None) is None:
        chat_file = tok_path / "chat_template.jinja"
        if chat_file.exists():
            hf_tokenizer.chat_template = chat_file.read_text()
            logger.info("flux2_dev tokenizer: loaded chat_template from %s", chat_file)
    wrapper = LanguageTokenizer(
        tokenizer=hf_tokenizer,
        max_length=max_length,
        padding="max_length",
        return_attention_mask=True,
        use_chat_template=True,
        add_special_tokens=True,
    )
    logger.info(
        "flux2_dev tokenizer: ready (vocab=%s max_length=%d chat_template=%s)",
        getattr(hf_tokenizer, "vocab_size", "?"),
        max_length,
        bool(getattr(hf_tokenizer, "chat_template", None)),
    )
    return wrapper
