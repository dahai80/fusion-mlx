import logging
import os

import mlx.core as mx
import numpy as np
from PIL import Image

from fusion_mlx.image.sd2.config import SD2Config, SD2ModelPaths
from fusion_mlx.image.sd2.scheduler import SD2DDIMScheduler
from fusion_mlx.image.sd2.text_encoder import SD2TextEncoder
from fusion_mlx.image.sd2.unet import SD2UNet
from fusion_mlx.image.sd2.vae import SD2VAE
from fusion_mlx.image.sd2.weights import load_clip, load_unet, load_vae

logger = logging.getLogger(__name__)


class GenResult:
    def __init__(self, image: Image.Image):
        self.image = image


def _local_path(subfolder: str, filename: str) -> str | None:
    base = os.environ.get("SD2_LOCAL_DIR")
    if not base:
        return None
    cand = (
        os.path.join(base, subfolder, filename)
        if subfolder
        else os.path.join(base, filename)
    )
    if os.path.exists(cand):
        logger.info("SD2 local resolve %s", cand)
        return cand
    return None


def _resolve(repo: str, subfolder: str, filename: str) -> str:
    local = _local_path(subfolder, filename)
    if local:
        return local
    from huggingface_hub import hf_hub_download

    path_in_repo = f"{subfolder}/{filename}" if subfolder else filename
    return hf_hub_download(repo, path_in_repo)


def _resolve_dir(repo: str, subfolder: str) -> str:
    base = os.environ.get("SD2_LOCAL_DIR")
    if base and os.path.isdir(os.path.join(base, subfolder)):
        logger.info("SD2 local resolve_dir %s/%s", base, subfolder)
        return os.path.join(base, subfolder)
    from huggingface_hub import snapshot_download

    # snapshot_download returns the repo snapshot root (contains all allowed
    # subfolders), NOT the subfolder itself. Append subfolder so callers get
    # the actual directory (e.g. <root>/tokenizer with vocab.json +
    # merges.txt). Without this, CLIPTokenizer.from_pretrained(root) finds no
    # tokenizer_config and falls back to a degenerate tokenizer (cf. #482).
    root = snapshot_download(
        repo, allow_patterns=[f"{subfolder}/*"], endpoint=os.environ.get("HF_ENDPOINT")
    )
    return os.path.join(root, subfolder)


def _load_safetensors(path: str) -> dict:
    from safetensors import safe_open

    raw = {}
    with safe_open(path, "pt") as f:
        for k in f.keys():  # noqa: SIM118
            raw[k] = mx.array(f.get_tensor(k).numpy())
    return raw


def _to_pil(image: mx.array) -> Image.Image:
    arr = np.array(image[0].astype(mx.float32))
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr / 2 + 0.5).clip(0, 1)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _pil_to_nchw(image: Image.Image, height: int, width: int) -> mx.array:
    if image.size != (width, height):
        image = image.resize((width, height), Image.LANCZOS)
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    arr = (arr * 2.0) - 1.0
    arr = np.transpose(arr, (2, 0, 1))
    return mx.array(arr)[None]


class SD2Pipeline:
    def __init__(self, model_config=None, model_path: str | None = None, quantize=None):
        self.config = SD2Config().unet
        self.text_cfg = SD2Config().text
        self.vae_cfg = SD2Config().vae
        self.paths = SD2ModelPaths()
        self.quantize = quantize
        self.unet = None
        self.vae = None
        self.clip_l = None
        self.tokenizer_l = None
        self._loaded = False

    def _load_all(self) -> None:
        if self._loaded:
            return
        logger.info("SD2 pipeline loading components")
        self._load_text_encoder()
        self._load_unet_and_vae()
        self._loaded = True
        logger.info("SD2 pipeline ready")

    def _load_text_encoder(self) -> None:
        from transformers import CLIPTokenizer

        tc = self.text_cfg
        # SD2.1 text encoder = ViT-H/14 in CLIPTextModel layout: hidden 1024,
        # 23 layers, 16 heads, intermediate 4096, gelu, no text_projection
        # (projection_dim=None). Reuse SDXLCLIPTextModel verbatim.
        self.clip_l = SD2TextEncoder(
            dims=tc.hidden_size,
            num_layers=tc.num_hidden_layers,
            num_heads=tc.num_attention_heads,
            intermediate=tc.intermediate_size,
            act=tc.hidden_act,
            vocab=tc.vocab,
            max_pos=tc.max_pos,
            projection_dim=None,
        )
        clip_l_path = _resolve(
            self.paths.repo, self.paths.text_subfolder, self.paths.text_file
        )
        load_clip(self.clip_l, _load_safetensors(clip_l_path))
        self.tokenizer_l = CLIPTokenizer.from_pretrained(
            _resolve_dir(self.paths.repo, self.paths.tokenizer_subfolder)
        )
        logger.info("SD2 text encoder loaded (ViT-H 1024/%d/16)", tc.num_hidden_layers)

    def _load_unet_and_vae(self) -> None:
        unet_path = _resolve(
            self.paths.repo, self.paths.unet_subfolder, self.paths.unet_file
        )
        self.unet = SD2UNet(self.config)
        load_unet(self.unet, _load_safetensors(unet_path))
        vae_path = _resolve(
            self.paths.repo, self.paths.vae_subfolder, self.paths.vae_file
        )
        self.vae = SD2VAE()
        load_vae(self.vae, _load_safetensors(vae_path))
        if self.quantize:
            try:
                import mlx.nn as nn

                nn.quantize(self.unet, group_size=64, bits=8)
                logger.info("SD2 unet quantized to 8-bit")
            except Exception:
                logger.debug("SD2 unet quantize skipped", exc_info=True)
        mx.eval(self.unet.parameters(), self.vae.parameters())
        logger.info("SD2 unet + vae loaded")

    def _encode_prompt(self, prompt: str) -> mx.array:
        # SD2.1 uses a single ViT-H text encoder; return last hidden state
        # (full sequence). Pooled output unused (no add_embedding).
        tok_l = self.tokenizer_l(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )["input_ids"].astype(np.int32)
        hidden_l, _ = self.clip_l(mx.array(tok_l))
        hidden_l = hidden_l.astype(mx.float32)
        mx.eval(hidden_l)
        return hidden_l

    def _make_scheduler(self, num_inference_steps: int) -> SD2DDIMScheduler:
        # SD2.1 ships DDIMScheduler (v_prediction). set_alpha_to_one=False,
        # clip_sample=False, steps_offset=1 — matches scheduler_config.json.
        scheduler = SD2DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="v_prediction",
            timestep_spacing="leading",
            steps_offset=1,
            set_alpha_to_one=False,
            clip_sample=False,
        )
        scheduler.set_timesteps(num_inference_steps)
        return scheduler

    def generate_image(
        self,
        seed: int = 0,
        prompt: str = "",
        num_inference_steps: int = 30,
        height: int = 512,
        width: int = 512,
        guidance: float = 7.5,
        negative_prompt: str = "",
        image_path: str | None = None,
        image_strength: float | None = None,
        **kwargs,
    ) -> GenResult:
        self._load_all()
        cfg = self.config
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("SD2 height/width must be multiples of 8")
        h_lat = height // 8
        w_lat = width // 8

        is_img2img = image_path is not None
        strength = float(image_strength) if image_strength is not None else 1.0
        if is_img2img:
            if strength <= 0.0 or strength > 1.0:
                logger.warning(
                    "SD2 image_strength=%.3f out of (0,1], clamping", strength
                )
                strength = max(min(strength, 1.0), 1e-3)
            eff_steps = max(1, int(round(num_inference_steps * strength)))
            logger.info(
                "SD2 img2img prompt=%r steps=%d eff_steps=%d strength=%.3f %dx%d",
                prompt[:60],
                num_inference_steps,
                eff_steps,
                strength,
                width,
                height,
            )
        else:
            eff_steps = num_inference_steps
            logger.info(
                "SD2 txt2img prompt=%r steps=%d %dx%d guidance=%.2f seed=%d",
                prompt[:60],
                num_inference_steps,
                width,
                height,
                guidance,
                seed,
            )

        mx.random.seed(seed)
        context = self._encode_prompt(prompt)
        if guidance > 1:
            context_un = self._encode_prompt(negative_prompt or "")
        else:
            context_un = None

        scheduler = self._make_scheduler(num_inference_steps)
        timesteps = scheduler.timesteps
        if is_img2img and eff_steps < len(timesteps):
            timesteps = timesteps[len(timesteps) - eff_steps :]
            start_idx = len(scheduler.sigmas) - 1 - eff_steps
            scheduler.sigmas = scheduler.sigmas[start_idx:]
            scheduler._step_index = 0

        if is_img2img:
            init_img = Image.open(image_path).convert("RGB")
            init_lat = self.vae.encode(_pil_to_nchw(init_img, height, width))
            init_lat = init_lat.astype(mx.float32)
            sigma_start = float(scheduler.sigmas[0])
            noise = mx.random.normal(init_lat.shape, dtype=mx.float32)
            latent = init_lat + sigma_start * noise
            mx.eval(latent)
            logger.info(
                "SD2 img2img init latent encoded, sigma_start=%.4f", sigma_start
            )
        else:
            latent = mx.random.normal(
                (1, cfg.in_channels, h_lat, w_lat), dtype=mx.float32
            )
            latent = latent * scheduler.init_noise_sigma

        for i, t in enumerate(timesteps):
            t_arr = mx.array([float(t)])
            if guidance > 1:
                latent_in = mx.concatenate([latent, latent], axis=0)
                context_in = mx.concatenate([context, context_un], axis=0)
                t_in = mx.concatenate([t_arr, t_arr], axis=0)
                noise = self.unet(
                    scheduler.scale_model_input(latent_in), t_in, context_in
                )
                noise_cond, noise_un = mx.split(noise, 2, axis=0)
                noise = noise_un + guidance * (noise_cond - noise_un)
            else:
                noise = self.unet(scheduler.scale_model_input(latent), t_arr, context)
            latent = scheduler.step(noise, latent)
            mx.eval(latent)
            logger.info("SD2 step %d/%d done", i + 1, len(timesteps))
        image = self.vae.decode(latent)
        mx.eval(image)
        return GenResult(image=_to_pil(image))
