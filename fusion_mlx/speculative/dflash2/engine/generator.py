# SPDX-License-Identifier: Apache-2.0
import logging
from collections.abc import Generator

import mlx.core as mx

logger = logging.getLogger(__name__)


def _local_draft_path(draft_repo: str):
    # Resolve a draft repo id to a local directory if it is one. Returns
    # the Path if draft_repo points to an existing dir (absolute or under
    # the standard model-dir), else None (let dflash download from HF).
    import os
    from pathlib import Path

    if not draft_repo:
        return None
    p = Path(draft_repo)
    candidates = [p] if p.is_absolute() else []
    model_dir = os.environ.get("FUSION_MODEL_DIR") or os.path.expanduser(
        "~/.fusion-mlx/models"
    )
    candidates.append(Path(model_dir) / draft_repo)
    for c in candidates:
        if c.is_dir() and (c / "config.json").exists():
            return c
    return None


class DFlash2Generator:
    def __init__(
        self,
        target_repo: str,
        draft_repo: str,
        block_size: int = 5,
        prefill_step_size: int = 2048,
    ) -> None:
        if not target_repo:
            raise ValueError("target_repo must be a non-empty string")
        if not draft_repo:
            raise ValueError("draft_repo must be a non-empty string")
        if block_size <= 0 or block_size > 5:
            raise ValueError(
                f"block_size must be in [1, 5] for MLX quantized targets; got {block_size}"
            )
        from dflash import model_mlx as _dflash

        logger.info("[dflash2] loading target=%s via mlx-lm", target_repo)
        self.target, self.tokenizer = _dflash.load(target_repo)
        logger.info("[dflash2] loading draft=%s (DFlash2DraftModel)", draft_repo)
        # dflash.load_draft calls huggingface_hub.snapshot_download, which
        # rejects local directory paths (HFValidationError). Short-circuit
        # it when draft_repo is an existing local dir so the draft loads
        # from disk (no re-download, honors CLAUDE.md hf-mirror workflow).
        draft_path = _local_draft_path(draft_repo)
        if draft_path is not None:
            _orig_download = _dflash.snapshot_download
            _dflash.snapshot_download = lambda _id, **_kw: str(draft_path)
            try:
                self.draft = _dflash.load_draft(str(draft_path))
            finally:
                _dflash.snapshot_download = _orig_download
        else:
            self.draft = _dflash.load_draft(draft_repo)
        self.draft.bind(self.target)
        self.target_repo = target_repo
        self.draft_repo = draft_repo
        self.block_size = block_size
        self.prefill_step_size = prefill_step_size
        logger.info(
            "[dflash2] ready target=%s draft=%s block_size=%d",
            target_repo,
            draft_repo,
            block_size,
        )

    def _encode(self, prompt_tokens) -> mx.array:
        if isinstance(prompt_tokens, mx.array):
            return prompt_tokens
        if isinstance(prompt_tokens, str):
            enc = getattr(self.tokenizer, "encode", None)
            if enc is None:
                raise TypeError("tokenizer has no encode(); pass token ids")
            ids = enc(prompt_tokens)
            return mx.array(ids)
        return mx.array(list(prompt_tokens))

    def stream_from_tokens(
        self,
        prompt_tokens,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> Generator[int, None, None]:
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1; got {max_new_tokens}")
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0; got {temperature}")
        from dflash import model_mlx as _dflash

        prompt = self._encode(prompt_tokens)
        emitted = 0
        upstream = _dflash.stream_generate(
            self.target,
            self.draft,
            self.tokenizer,
            prompt,
            block_size=self.block_size,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            prefill_step_size=self.prefill_step_size,
        )
        try:
            for resp in upstream:
                for tok in resp.tokens:
                    yield int(tok)
                    emitted += 1
                    if emitted >= max_new_tokens:
                        return
        finally:
            close = getattr(upstream, "close", None)
            if callable(close):
                close()
