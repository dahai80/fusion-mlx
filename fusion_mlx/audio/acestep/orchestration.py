# SPDX-License-Identifier: Apache-2.0
# ACE-Step MLX orchestration (issue #988): wires Qwen3-Embedding text encoder
# + silence_latent + DiT generate + VAE decode for text2music. Reconstructed
# from ACE-Step/Ace-Step-v1.5 Space demo (MIT) — own implementation, no copy.
from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx

from .config import AceStepConfig
from .pipeline import AceStepModel
from .vae import AutoencoderOobleckMLX

logger = logging.getLogger(__name__)

DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"

SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}
"""

LATENT_HZ = 25
VAE_HOP = 1920
SAMPLE_RATE = 48000


def _format_lyrics(lyrics: str, language: str) -> str:
    return f"# Languages\n{language}\n\n# Lyric\n{lyrics}\n"


def _build_meta_str(bpm="N/A", timesignature="N/A", keyscale="N/A", duration=30) -> str:
    dur_str = (
        f"{int(duration)} seconds"
        if isinstance(duration, (int, float))
        else str(duration)
    )
    return f"- bpm: {bpm}\n- timesignature: {timesignature}\n- keyscale: {keyscale}\n- duration: {dur_str}\n"


class AceStepOrchestrator:
    # Text2music orchestration: Qwen3-Embedding (text+lyric) + silence_latent +
    # DiT flow-match + VAE decode. No LM planner (thinking=False base path).
    def __init__(
        self,
        config: AceStepConfig,
        model: AceStepModel,
        vae: AutoencoderOobleckMLX,
        silence_latent: mx.array,
    ):
        self.config = config
        self.model = model
        self.vae = vae
        # silence_latent: (1, 15000, 64) — tiled to duration
        self.silence_latent = silence_latent
        self._text_model = None
        self._text_tokenizer = None
        logger.info(
            "AceStepOrchestrator ready: silence_latent %s, DiT layers=%d, VAE hop=%d",
            silence_latent.shape,
            config.num_hidden_layers,
            vae.hop_length,
        )

    def load_text_encoder(self, repo_or_path: str):
        # Qwen3-Embedding-0.6B ships as base Qwen3Model weights (no model. prefix,
        # no lm_head). mlx-lm load() fails on it (expects Qwen3ForCausalLM). Build
        # Qwen3Model directly + load base weights + tokenizer from utils.
        import json

        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model
        from mlx_lm.utils import load_tokenizer

        logger.info("loading text encoder: %s", repo_or_path)
        local_path = (
            snapshot_download(repo_or_path)
            if not Path(repo_or_path).exists()
            else repo_or_path
        )
        cfg = json.loads(Path(local_path, "config.json").read_text())
        fields = ModelArgs.__dataclass_fields__
        kwargs = {k: v for k, v in cfg.items() if k in fields}
        args = ModelArgs(**kwargs)
        model = Qwen3Model(args)
        st_path = Path(local_path, "model.safetensors")
        weights = mx.load(str(st_path))
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())
        self._text_model = model
        self._text_tokenizer = load_tokenizer(Path(local_path))
        logger.info(
            "text encoder loaded: hidden=%d layers=%d vocab=%d",
            args.hidden_size,
            args.num_hidden_layers,
            self._text_tokenizer.vocab_size,
        )

    def _encode(self, text: str, max_length: int = 256) -> tuple[mx.array, mx.array]:
        if self._text_model is None or self._text_tokenizer is None:
            raise RuntimeError(
                "text encoder not loaded; call load_text_encoder() first"
            )
        tokens = self._text_tokenizer.encode(text)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        ids = mx.array([tokens])
        mask = mx.ones_like(ids)
        hidden = self._text_model(ids)
        return hidden, mask

    def generate_music(
        self,
        caption: str,
        lyrics: str = "",
        duration: float = 30.0,
        language: str = "en",
        bpm: str = "N/A",
        timesignature: str = "N/A",
        keyscale: str = "N/A",
        seed: int | None = 42,
        shift: float = 1.0,
        infer_method: str = "ode",
    ) -> mx.array:
        # Returns wav array (1, 2, samples) at 48kHz.
        T = int(duration * LATENT_HZ)
        T = min(T, self.silence_latent.shape[1])
        logger.info(
            "generate_music: caption=%r dur=%.1fs T=%d lyrics_len=%d",
            caption[:60],
            duration,
            T,
            len(lyrics),
        )

        # text + lyric hidden states
        meta_str = _build_meta_str(bpm, timesignature, keyscale, duration)
        text_prompt = SFT_GEN_PROMPT.format(DEFAULT_DIT_INSTRUCTION, caption, meta_str)
        text_hidden, text_mask = self._encode(text_prompt, max_length=256)

        lyrics_text = (
            _format_lyrics(lyrics, language) if lyrics else _format_lyrics("", language)
        )
        lyric_hidden, lyric_mask = self._encode(lyrics_text, max_length=2048)

        # src_latents = silence tiled to T
        src_latents = self.silence_latent[:, :T, :].astype(mx.float32)
        # chunk_masks = all ones (full generation)
        chunk_masks = mx.ones(
            (1, T, self.config.audio_acoustic_hidden_dim), dtype=mx.float32
        )

        generated = self.model.generate(
            text_hidden_states=text_hidden,
            text_attention_mask=text_mask,
            lyric_hidden_states=lyric_hidden,
            lyric_attention_mask=lyric_mask,
            src_latents=src_latents,
            chunk_masks=chunk_masks,
            refer_audio_packed=None,
            refer_audio_order_mask=None,
            attention_mask=mx.ones((1, T), dtype=mx.float32),
            seed=seed,
            shift=shift,
            infer_method=infer_method,
        )
        mx.eval(generated)
        logger.info("DiT generate done: %s", generated.shape)

        # VAE decode: (1, T, 64) -> (1, 64, T) -> (1, 2, T*1920)
        z = mx.swapaxes(generated, 1, 2)
        wav = self.vae.decode(z)
        mx.eval(wav)
        logger.info("VAE decode done: %s", wav.shape)
        return wav


__all__ = ["AceStepOrchestrator"]
