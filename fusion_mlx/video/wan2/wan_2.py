import math
import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .attention import WanLayerNorm, _linear_dtype
from .config import WanModelConfig
from .rope import rope_params, rope_precompute_cos_sin
from .transformer import WanAttentionBlock
from .vace import VACEBlock

logger = logging.getLogger(__name__)


def sinusoidal_embedding_1d(dim: int, position: mx.array) -> mx.array:
    assert dim % 2 == 0
    half = dim // 2
    pos = position.astype(mx.float32)
    inv_freq = mx.power(10000.0, -mx.arange(half).astype(mx.float32) / half)
    sinusoid = pos[..., None] * inv_freq  # [..., half]
    return mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=-1)


class Head(nn.Module):

    def __init__(self, dim: int, out_dim: int, patch_size: tuple, eps: float = 1e-6):
        super().__init__()
        self.out_dim = out_dim
        self.patch_size = patch_size
        proj_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, proj_dim)
        self.modulation = (mx.random.normal((1, 2, dim)) * (dim**-0.5)).astype(
            mx.float32
        )

    def __call__(self, x: mx.array, e: mx.array) -> mx.array:
        if e.ndim == 2:
            e = e[:, None, :]  # [B, 1, dim]
        # Compute modulation in float32 (matching reference's autocast(float32))
        mod = self.modulation[:, None, :, :] + e[:, :, None, :]  # float32
        e0 = mod[:, :, 0, :]  # [B, L_e, dim] shift
        e1 = mod[:, :, 1, :]  # [B, L_e, dim] scale
        x_norm = self.norm(x)
        x_mod = x_norm * (1 + e1) + e0
        return self.head(x_mod)


class WanModel(nn.Module):

    def __init__(self, config: WanModelConfig):
        super().__init__()
        self.config = config
        dim = config.dim
        self.dim = dim
        self.num_heads = config.num_heads
        self.out_dim = config.out_dim
        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.freq_dim = config.freq_dim

        # Patch embedding: Conv3d implemented as a reshaped linear
        # For kernel (1,2,2) and stride (1,2,2): reshape input then linear
        patch_dim = config.in_dim * math.prod(config.patch_size)
        self.patch_embedding_proj = nn.Linear(patch_dim, dim)
        self._patch_size = config.patch_size

        # Text embedding MLP
        self.text_embedding_0 = nn.Linear(config.text_dim, dim)
        self.text_embedding_act = nn.GELU(approx="tanh")
        self.text_embedding_1 = nn.Linear(dim, dim)

        # Time embedding MLP
        self.time_embedding_0 = nn.Linear(config.freq_dim, dim)
        self.time_embedding_act = nn.SiLU()
        self.time_embedding_1 = nn.Linear(dim, dim)

        # Time projection for modulation (6x dim)
        self.time_projection_act = nn.SiLU()
        self.time_projection = nn.Linear(dim, dim * 6)

        # Transformer blocks
        self.blocks = [
            WanAttentionBlock(
                dim=dim,
                ffn_dim=config.ffn_dim,
                num_heads=config.num_heads,
                window_size=config.window_size,
                qk_norm=config.qk_norm,
                cross_attn_norm=config.cross_attn_norm,
                eps=config.eps,
            )
            for _ in range(config.num_layers)
        ]

        # Output head
        self.head = Head(dim, config.out_dim, config.patch_size, config.eps)

        # Precompute RoPE frequencies — three separate tables concatenated.
        # Reference computes three rope_params with different dim normalizations
        # so each axis (temporal/height/width) gets its own full frequency range.
        d = dim // config.num_heads
        self.freqs = mx.concatenate(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            axis=1,
        )

        # Precompute sinusoidal inv_freq for time embedding.
        half = config.freq_dim // 2
        self._inv_freq = mx.array(
            np.power(10000.0, -np.arange(half, dtype=np.float64) / half).astype(
                np.float32
            )
        )

        # VACE: Video-Conditioned Auxiliary Control Encoding
        # Only instantiated when vace_layers is non-empty (VACE-14B model).
        self._vace_layers = tuple(sorted(config.vace_layers)) if config.vace_layers else ()
        if config.vace_layers:
            vace_patch_dim = config.vace_in_dim * math.prod(config.patch_size)
            self.vace_patch_embedding_proj = nn.Linear(vace_patch_dim, dim)
            self.vace_blocks = [
                VACEBlock(
                    dim=dim,
                    ffn_dim=config.ffn_dim,
                    num_heads=config.num_heads,
                    qk_norm=config.qk_norm,
                    cross_attn_norm=config.cross_attn_norm,
                    eps=config.eps,
                    has_before_proj=(i == 0),
                )
                for i in range(len(config.vace_layers))
            ]
            # Map from block index -> vace_block index for O(1) lookup
            self._vace_block_map = {
                layer_idx: vace_idx
                for vace_idx, layer_idx in enumerate(config.vace_layers)
            }

    def _patchify(self, x: mx.array) -> tuple:
        c, f, h, w = x.shape
        pt, ph, pw = self._patch_size

        f_out = f // pt
        h_out = h // ph
        w_out = w // pw

        # Reshape: [C, F, H, W] -> [F', H', W', C, pt, ph, pw] -> [F'*H'*W', C*pt*ph*pw]
        # Order must be [C, pt, ph, pw] (C slowest) to match Conv3d weight layout
        x = x.reshape(c, f_out, pt, h_out, ph, w_out, pw)
        x = x.transpose(1, 3, 5, 0, 2, 4, 6)  # [F', H', W', C, pt, ph, pw]
        x = x.reshape(f_out * h_out * w_out, -1)  # [L, C*pt*ph*pw]

        # Project and cast to model dtype to prevent float32 cascade from input latents
        patches = self.patch_embedding_proj(x)  # [L, dim]
        patches = patches.astype(_linear_dtype(self.patch_embedding_proj))
        patches = patches[None, :, :]  # [1, L, dim]

        return patches, (f_out, h_out, w_out)

    def _patchify_vace(self, x: mx.array) -> tuple:
        # Patchify control hidden states using vace_patch_embedding_proj.
        # x shape: [C_vace, F, H, W] where C_vace = vace_in_dim
        c, f, h, w = x.shape
        pt, ph, pw = self._patch_size

        f_out = f // pt
        h_out = h // ph
        w_out = w // pw

        x = x.reshape(c, f_out, pt, h_out, ph, w_out, pw)
        x = x.transpose(1, 3, 5, 0, 2, 4, 6)
        x = x.reshape(f_out * h_out * w_out, -1)

        patches = self.vace_patch_embedding_proj(x)
        patches = patches.astype(_linear_dtype(self.vace_patch_embedding_proj))
        patches = patches[None, :, :]

        return patches, (f_out, h_out, w_out)

    def unpatchify(self, x: mx.array, grid_sizes: list) -> list:
        c = self.out_dim
        pt, ph, pw = self.patch_size
        out = []
        for i, (f, h, w) in enumerate(grid_sizes):
            seq_len = f * h * w
            u = x[i, :seq_len]  # [L, out_dim * pt * ph * pw]
            u = u.reshape(f, h, w, pt, ph, pw, c)
            # Rearrange: [F', H', W', pt, ph, pw, C] -> [C, F'*pt, H'*ph, W'*pw]
            u = u.transpose(6, 0, 3, 1, 4, 2, 5)  # [C, F', pt, H', ph, W', pw]
            u = u.reshape(c, f * pt, h * ph, w * pw)
            out.append(u)
        return out

    def embed_text(self, context: list) -> mx.array:
        model_dtype = _linear_dtype(self.patch_embedding_proj)
        context_padded = []
        for ctx in context:
            pad_len = self.text_len - ctx.shape[0]
            if pad_len > 0:
                ctx = mx.concatenate(
                    [ctx, mx.zeros((pad_len, ctx.shape[1]), dtype=ctx.dtype)],
                    axis=0,
                )
            context_padded.append(ctx)
        context_batch = mx.stack(context_padded)  # [B, text_len, text_dim]
        context_batch = self.text_embedding_1(
            self.text_embedding_act(self.text_embedding_0(context_batch))
        )
        return context_batch.astype(model_dtype)

    def prepare_cross_kv(self, context: mx.array) -> list:
        kv_caches = []
        for block in self.blocks:
            kv_caches.append(block.cross_attn.prepare_kv(context))
        return kv_caches

    def prepare_rope(self, grid_sizes: list) -> tuple:
        w_dtype = _linear_dtype(self.patch_embedding_proj)
        return rope_precompute_cos_sin(grid_sizes, self.freqs, dtype=w_dtype)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        # Remap PyTorch Conv3d weights to MLX Linear for patch embeddings.
        # Conv3d weight shape: [out_ch, in_ch, kt, kh, kw]
        # Linear weight shape: [out_dim, in_dim] where in_dim = in_ch*kt*kh*kw
        remapped = {}
        for k, v in weights.items():
            nk = k
            # Remap vace_patch_embedding (Conv3d) -> vace_patch_embedding_proj (Linear)
            if k.startswith("vace_patch_embedding."):
                suffix = k[len("vace_patch_embedding."):]
                nk = f"vace_patch_embedding_proj.{suffix}"
                if suffix == "weight" and v.ndim == 5:
                    out_ch, in_ch, kt, kh, kw = v.shape
                    v = v.reshape(out_ch, in_ch * kt * kh * kw)
                    logger.info(
                        "VACE: reshaped vace_patch_embedding.weight %s -> %s",
                        (out_ch, in_ch, kt, kh, kw),
                        v.shape,
                    )
                elif suffix == "bias" and v.ndim > 1:
                    v = v.reshape(-1)
            # Remap patch_embedding (Conv3d) -> patch_embedding_proj (Linear)
            elif k.startswith("patch_embedding."):
                suffix = k[len("patch_embedding."):]
                nk = f"patch_embedding_proj.{suffix}"
                if suffix == "weight" and v.ndim == 5:
                    out_ch, in_ch, kt, kh, kw = v.shape
                    v = v.reshape(out_ch, in_ch * kt * kh * kw)
                elif suffix == "bias" and v.ndim > 1:
                    v = v.reshape(-1)
            remapped[nk] = v
        return remapped

    def __call__(
        self,
        x_list: list,
        t: mx.array,
        context: list | mx.array,
        seq_len: int,
        cross_kv_caches: list | None = None,
        y: list | None = None,
        rope_cos_sin: tuple | None = None,
        control_hidden_states: list | None = None,
        control_scales: list[float] | None = None,
    ) -> list:
        # Detect identical inputs (CFG B=2) to avoid duplicate patchify work.
        # Check BEFORE I2V concat since concat creates new array objects.
        batch_size = len(x_list)
        all_same = batch_size > 1 and all(
            x_list[i] is x_list[0] for i in range(1, batch_size)
        )
        if all_same and y is not None:
            all_same = all(y[i] is y[0] for i in range(1, len(y)))

        # I2V: channel-concatenate conditioning y with noise x
        if y is not None:
            x_list = [mx.concatenate([u, v], axis=0) for u, v in zip(x_list, y)]

        if all_same:
            # Patchify once and broadcast — saves a Linear projection per step
            p, gs = self._patchify(x_list[0])  # [1, L, dim]
            grid_sizes = [gs] * batch_size
            seq_lens_list = [p.shape[1]] * batch_size
            # Pad and broadcast
            if p.shape[1] < seq_len:
                p = mx.concatenate(
                    [p, mx.zeros((1, seq_len - p.shape[1], self.dim), dtype=p.dtype)],
                    axis=1,
                )
            x = mx.broadcast_to(p, (batch_size,) + p.shape[1:])
        else:
            patches = []
            grid_sizes = []
            seq_lens_list = []
            for vid in x_list:
                p, gs = self._patchify(vid)  # [1, L, dim]
                patches.append(p)
                grid_sizes.append(gs)
                seq_lens_list.append(p.shape[1])
            x = mx.concatenate(
                [
                    (
                        mx.concatenate(
                            [
                                p,
                                mx.zeros(
                                    (1, seq_len - p.shape[1], self.dim), dtype=p.dtype
                                ),
                            ],
                            axis=1,
                        )
                        if p.shape[1] < seq_len
                        else p
                    )
                    for p in patches
                ],
                axis=0,
            )  # [B, seq_len, dim]

        # Time embedding: sinusoidal from precomputed inv_freq.
        # inv_freq was computed in float64 for precision, stored as float32.
        # With integer timesteps (matching reference), float32 sin/cos is fine.
        if t.ndim == 0:
            t = t[None]

        sinusoid = t[..., None].astype(mx.float32) * self._inv_freq
        sin_emb = mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=-1)

        if t.ndim == 1:
            # Standard T2V: scalar timestep per batch element [B]
            e = self.time_embedding_1(
                self.time_embedding_act(self.time_embedding_0(sin_emb))
            )  # [B, dim]
            e0 = self.time_projection(self.time_projection_act(e))  # [B, dim*6]
            e0 = e0.reshape(batch_size, 1, 6, self.dim)
        else:
            # I2V: per-token timesteps [B, L]
            e = self.time_embedding_1(
                self.time_embedding_act(self.time_embedding_0(sin_emb))
            )  # [B, L, dim]
            e0 = self.time_projection(self.time_projection_act(e))  # [B, L, dim*6]
            e0 = e0.reshape(batch_size, -1, 6, self.dim)

        # Text embedding: skip MLP if context is already embedded (mx.array)
        if isinstance(context, mx.array):
            # Pre-embedded: expand to batch size if needed
            context_batch = context
            if context_batch.shape[0] == 1 and batch_size > 1:
                context_batch = mx.broadcast_to(
                    context_batch, (batch_size,) + context_batch.shape[1:]
                )
        else:
            context_batch = self.embed_text(context)

        # Pre-compute attention mask from seq_lens (constant across all blocks)
        attn_mask = None
        w_dtype = _linear_dtype(self.patch_embedding_proj)
        if any(sl < seq_len for sl in seq_lens_list):
            attn_mask = mx.zeros((batch_size, 1, 1, seq_len), dtype=w_dtype)
            for i, sl in enumerate(seq_lens_list):
                attn_mask[i, :, :, sl:] = -1e9

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens_list,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context_batch,
            context_lens=None,
            rope_cos_sin=rope_cos_sin,
            attn_mask=attn_mask,
        )

        # VACE: prepare control hints if control states are provided
        vace_hints = {}
        if self._vace_layers and control_hidden_states is not None:
            if control_scales is None:
                control_scales = [1.0] * len(self._vace_layers)

            # Patchify control hidden states
            ctrl_patches = []
            ctrl_grid_sizes = []
            ctrl_seq_lens = []
            for ctrl in control_hidden_states:
                cp, cgs = self._patchify_vace(ctrl)
                ctrl_patches.append(cp)
                ctrl_grid_sizes.append(cgs)
                ctrl_seq_lens.append(cp.shape[1])

            # Pad and concatenate control patches
            max_ctrl_len = max(ctrl_seq_lens)
            ctrl_x = mx.concatenate(
                [
                    (
                        mx.concatenate(
                            [p, mx.zeros((1, max_ctrl_len - p.shape[1], self.dim), dtype=p.dtype)],
                            axis=1,
                        )
                        if p.shape[1] < max_ctrl_len
                        else p
                    )
                    for p in ctrl_patches
                ],
                axis=0,
            )

            # Run VACE blocks to produce conditioning hints (reversed order for pop)
            ctrl_state = ctrl_x
            for vi, vace_block in enumerate(self.vace_blocks):
                conditioning, ctrl_state = vace_block(
                    x, ctrl_state, e=e0, seq_lens=ctrl_seq_lens,
                    grid_sizes=ctrl_grid_sizes, freqs=self.freqs,
                    context=context_batch, context_lens=None,
                    rope_cos_sin=rope_cos_sin, attn_mask=attn_mask,
                )
                # Map vace_block index to main block layer index
                layer_idx = self._vace_layers[vi]
                vace_hints[layer_idx] = (conditioning, control_scales[vi])

        # Run transformer blocks with optional VACE injection
        for i, block in enumerate(self.blocks):
            kv = cross_kv_caches[i] if cross_kv_caches is not None else None
            x = block(x, cross_kv_cache=kv, **kwargs)

            # VACE injection at specified layers
            if i in vace_hints:
                hint, scale = vace_hints[i]
                # Pad hint to match x seq_len if needed
                if hint.shape[1] < x.shape[1]:
                    hint = mx.concatenate(
                        [hint, mx.zeros((hint.shape[0], x.shape[1] - hint.shape[1], self.dim), dtype=hint.dtype)],
                        axis=1,
                    )
                x = x + hint * scale

        # Output head
        x = self.head(x, e)

        # Unpatchify
        outputs = self.unpatchify(x, grid_sizes)
        return [u.astype(mx.float32) for u in outputs]
