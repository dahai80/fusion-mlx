"""DSpark draft model in MLX.

Mirrors the PyTorch reference at
refs/deepspec/deepspec/modeling/dspark/qwen3/modeling.py. Replication-critical
conventions (verified against the reference, exercised by
tests/test_draft_parity.py):

1. Context features: the target-model tap for ``layer_id`` is the *output of*
   decoder layer ``layer_id`` (HF ``hidden_states[layer_id + 1]``); ``-1``
   means the embedding output. The concatenated taps go through ``fc`` then
   ``RMSNorm(hidden_norm)`` once per forward.
2. Projected ctx is prepended to every draft layer's K/V:
   ``k = [k_ctx; k_block]``, likewise for v. ``k_norm`` is applied AFTER
   concatenating ctx and block keys (reference modeling.py:107-113).
3. RoPE: the reference slices cos/sin to the last ``q_len`` positions for q
   but applies the full length (ctx + block) to k, i.e. q positions are
   ``offset + ctx_len .. offset + ctx_len + q_len`` while k positions start
   at ``offset``. ``offset`` is the draft KV cache length (committed prefix
   already consumed by previous rounds).
4. Attention within the draft block is bidirectional (full attention over
   [cached ctx; new ctx; block]) — no causal mask.
5. Draft input = embedding of ``[anchor_token, mask_token_id x (block-1)]``.
6. Draft KV cache: ctx+block K/V are appended during forward; the runtime
   crops the block back each round (reference ``past_key_values_draft.crop``),
   so the cache retains the rotated ctx K/V of the whole committed prefix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm.models import cache as cache_lib
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.utils import quantize_model

from .heads import ConfidenceHead, MarkovHead


def resolve_model_path(path_or_repo: str) -> Path:
    path = Path(path_or_repo)
    if path.exists():
        return path
    return Path(snapshot_download(path_or_repo))


@dataclass
class DraftArgs:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    head_dim: int
    tie_word_embeddings: bool
    attention_bias: bool = False
    attention_dropout: float = 0.0
    rope_scaling: dict | None = None
    block_size: int = 16
    dspark_config: dict | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> DraftArgs:
        config = dict(config)
        if "rope_theta" not in config:
            # transformers 5.x nests rope settings under rope_parameters.
            rope_parameters = config.get("rope_parameters") or {}
            if "rope_theta" in rope_parameters:
                config["rope_theta"] = rope_parameters["rope_theta"]
        if "dspark_config" not in config and "dflash_config" in config:
            # Legacy skeleton key; dspark_metal.convert emits "dspark_config".
            config["dspark_config"] = config["dflash_config"]
        keys = {
            "model_type",
            "hidden_size",
            "num_hidden_layers",
            "intermediate_size",
            "num_attention_heads",
            "rms_norm_eps",
            "vocab_size",
            "num_key_value_heads",
            "max_position_embeddings",
            "rope_theta",
            "head_dim",
            "tie_word_embeddings",
            "attention_bias",
            "attention_dropout",
            "rope_scaling",
            "block_size",
            "dspark_config",
        }
        return cls(**{key: config[key] for key in keys if key in config})


class DSparkAttention(nn.Module):
    def __init__(self, args: DraftArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            args.hidden_size,
            self.n_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        target_hidden: mx.array,
        cache: cache_lib.KVCache | None = None,
        offset: int = 0,
    ) -> mx.array:
        batch_size, query_len, _ = hidden_states.shape
        context_len = target_hidden.shape[1]

        queries = self.q_proj(hidden_states)
        queries = self.q_norm(
            queries.reshape(batch_size, query_len, self.n_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)

        kv_states = mx.concatenate([target_hidden, hidden_states], axis=1)
        keys = self.k_proj(kv_states)
        values = self.v_proj(kv_states)
        keys = self.k_norm(
            keys.reshape(
                batch_size,
                context_len + query_len,
                self.n_kv_heads,
                self.head_dim,
            )
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_size,
            context_len + query_len,
            self.n_kv_heads,
            self.head_dim,
        ).transpose(0, 2, 1, 3)

        # Reference RoPE convention: cos/sin cover positions
        # [offset, offset + ctx_len + q_len); k takes the full range, q the
        # last q_len positions. With a cache, offset is the cached length.
        if cache is not None:
            offset = cache.offset
        queries = self.rope(queries, offset=offset + context_len)
        keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        # Bidirectional attention over [cached ctx; new ctx; block]: no mask.
        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=None,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, query_len, -1)
        return self.o_proj(output)


class DSparkDecoderLayer(nn.Module):
    def __init__(self, args: DraftArgs):
        super().__init__()
        self.self_attn = DSparkAttention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size,
            eps=args.rms_norm_eps,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        target_hidden: mx.array,
        cache: cache_lib.KVCache | None = None,
        offset: int = 0,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            target_hidden,
            cache=cache,
            offset=offset,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class DSparkDraftModel(nn.Module):
    """DSpark draft: 5-layer Qwen3-style backbone with ctx-KV injection,
    embed/lm_head copies, vanilla Markov head, and confidence head.

    Module names match the converted checkpoint keys exactly
    (``embed_tokens``, ``lm_head``, ``layers.{i}``, ``fc``, ``hidden_norm``,
    ``norm``, ``markov_head.markov_w1/markov_w2``, ``confidence_head.proj``)
    so ``load_weights`` maps directly with no transposes.
    """

    def __init__(self, args: DraftArgs):
        super().__init__()
        self.args = args
        dspark_config = dict(args.dspark_config or {})
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DSparkDecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.target_layer_ids = list(dspark_config["target_layer_ids"])
        self.fc = nn.Linear(
            len(self.target_layer_ids) * args.hidden_size,
            args.hidden_size,
            bias=False,
        )
        self.hidden_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.block_size = args.block_size
        self.mask_token_id = int(dspark_config["mask_token_id"])

        self.markov_rank = int(dspark_config.get("markov_rank", 0))
        if self.markov_rank > 0:
            markov_head_type = str(dspark_config.get("markov_head_type", "vanilla"))
            if markov_head_type != "vanilla":
                raise NotImplementedError(
                    f"markov_head_type={markov_head_type!r} is not supported; "
                    "only 'vanilla' Markov heads are implemented."
                )
            self.markov_head = MarkovHead(args.vocab_size, self.markov_rank)
        else:
            self.markov_head = None

        self.enable_confidence_head = bool(
            dspark_config.get("enable_confidence_head", False)
        )
        self.confidence_head_with_markov = bool(
            dspark_config.get("confidence_head_with_markov", False)
        )
        if self.enable_confidence_head:
            input_dim = args.hidden_size
            if self.confidence_head_with_markov:
                assert self.markov_head is not None, (
                    "confidence_head_with_markov requires a Markov head "
                    "(markov_rank > 0)."
                )
                input_dim += self.markov_rank
            self.confidence_head = ConfidenceHead(input_dim)
        else:
            self.confidence_head = None

        # STS calibration temperatures (paper §3.2.1), written into the
        # converted model's config.json by scripts/calibrate.py. The runtime
        # divides confidence logits by these before sigmoid + threshold;
        # None (uncalibrated checkpoint) keeps raw confidences.
        sts_temperatures = dspark_config.get("sts_temperatures")
        if sts_temperatures is not None:
            sts_temperatures = [float(t) for t in sts_temperatures]
            if len(sts_temperatures) != self.block_size:
                raise ValueError(
                    f"dspark_config.sts_temperatures has {len(sts_temperatures)} "
                    f"entries but block_size is {self.block_size}"
                )
            if any(t <= 0.0 for t in sts_temperatures):
                raise ValueError(
                    "dspark_config.sts_temperatures must all be positive, got "
                    f"{sts_temperatures}"
                )
        self.sts_temperatures: list[float] | None = sts_temperatures

        # Set by maybe_quantize_draft_model (None = bf16 draft).
        self.draft_quantization: dict[str, Any] | None = None

    def make_cache(self) -> list[cache_lib.KVCache]:
        return [cache_lib.KVCache() for _ in self.layers]

    def block_inputs(self, anchor_tokens: mx.array) -> tuple[mx.array, mx.array]:
        """Draft block input ids and embeddings for the given anchor token(s).

        anchor_tokens: [B] or [B, 1] int — the last committed token.
        Returns ``(input_ids [B, block_size], embeddings [B, block_size, H])``
        where input_ids = [anchor, mask_token_id x (block_size - 1)].
        """
        if anchor_tokens.ndim == 1:
            anchor_tokens = anchor_tokens[:, None]
        batch_size = anchor_tokens.shape[0]
        masks = mx.full(
            (batch_size, self.block_size - 1),
            self.mask_token_id,
            dtype=anchor_tokens.dtype,
        )
        input_ids = mx.concatenate([anchor_tokens, masks], axis=1)
        return input_ids, self.embed_tokens(input_ids)

    def __call__(
        self,
        noise_embedding: mx.array,
        target_hidden: mx.array,
        cache: list[cache_lib.KVCache] | None = None,
        offset: int = 0,
    ) -> mx.array:
        """Run the draft backbone over one block.

        noise_embedding: [B, q_len, H] embeddings of [anchor, mask x (q_len-1)].
        target_hidden: [B, ctx_len, num_taps * H] raw concatenated target taps
            (this method applies fc + hidden_norm).
        cache: optional per-layer KVCache list. When given, ctx+block K/V are
            appended to it; the caller crops the block back to the committed
            length after the round. Positions derive from ``cache.offset``.
        offset: cacheless-position offset (ignored when ``cache`` is given) —
            the absolute position of the first ctx token.
        Returns post-``norm`` hidden states [B, q_len, H].
        """
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states = layer(
                hidden_states,
                target_hidden,
                cache=layer_cache,
                offset=offset,
            )
        return self.norm(hidden_states)

    def compute_logits(self, hidden_states: mx.array) -> mx.array:
        """Pre-Markov base logits U from post-norm backbone hidden states."""
        return self.lm_head(hidden_states)

    def confidence_logits(
        self,
        hidden_states: mx.array,
        prev_token_ids: mx.array | None = None,
    ) -> mx.array | None:
        """Confidence *logits* (sigmoid applied by the caller at thresholding).

        Mirrors ``predict_confidence_step``: features are the post-``norm``
        backbone hidden states, concatenated with ``markov_w1[x_prev]`` when
        ``confidence_head_with_markov``. Returns [B, L] float32 or None.
        """
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            assert prev_token_ids is not None
            prev_embeddings = self.markov_head.prev_embeddings(prev_token_ids)
            features = mx.concatenate(
                [hidden_states, prev_embeddings.astype(hidden_states.dtype)],
                axis=-1,
            )
            return self.confidence_head(features).astype(mx.float32)
        return self.confidence_head(hidden_states).astype(mx.float32)


def load_draft_model(
    path_or_repo: str,
    quantize_bits: int | None = None,
    quantize_group_size: int = 64,
    quantize_embeddings: bool = True,
) -> tuple[DSparkDraftModel, Path]:
    """Load a converted DSpark draft, optionally quantizing it at load time.

    Quantization is load-time rather than conversion-time: one converted bf16
    directory serves every bit-width (quantizing the ~1–2 GB draft takes ~1 s
    at load), and the applied setting is recorded on the in-memory config
    (``dspark_config.draft_quantization``) plus ``draft.draft_quantization``.
    ``quantize_embeddings=False`` keeps embed_tokens/lm_head in bf16 (used
    together with runtime target-embedding reuse; see ``api.DSparkGenerator``).
    """
    model_path = resolve_model_path(path_or_repo)
    config = json.loads((model_path / "config.json").read_text())
    dspark_config = config.get("dspark_config") or {}
    if dspark_config.get("reuse_target_embeddings"):
        raise NotImplementedError(
            f"{path_or_repo!r} was converted with --reuse-target-embeddings "
            "(embed_tokens/lm_head omitted from the draft weights); loading "
            "such checkpoints standalone is not supported yet. Re-convert "
            "without the flag to materialize the copies."
        )
    draft = DSparkDraftModel(DraftArgs.from_dict(config))

    weights: list[tuple[str, mx.array]] = []
    for weight_file in sorted(model_path.glob("model*.safetensors")):
        weights.extend(mx.load(str(weight_file)).items())
    if not weights:
        raise FileNotFoundError(f"No draft weights found in {model_path}")
    draft.load_weights(weights)
    mx.eval(draft.parameters())
    draft.draft_quantization = maybe_quantize_draft_model(
        draft,
        bits=quantize_bits,
        group_size=quantize_group_size,
        quantize_embeddings=quantize_embeddings,
    )
    return draft, model_path


def maybe_quantize_draft_model(
    draft: DSparkDraftModel,
    bits: int | None,
    group_size: int,
    quantize_embeddings: bool = True,
) -> dict[str, Any] | None:
    """Quantize a loaded draft in place (no-op when ``bits`` is None).

    Uses ``mlx_lm.utils.quantize_model``: every module with ``to_quantized``
    and a group-size-divisible last dim is quantized (nn.Linear →
    nn.QuantizedLinear, nn.Embedding → nn.QuantizedEmbedding — including
    embed_tokens/lm_head and the Markov head). With
    ``quantize_embeddings=False`` the vocab-sized embed_tokens/lm_head copies
    stay bf16 (so they can be rebound to the target's identical tensors).
    Records the applied setting under ``dspark_config.draft_quantization``.
    """
    if bits is None:
        return None
    quant_predicate = None
    if not quantize_embeddings:

        def quant_predicate(path: str, module: nn.Module) -> bool:
            return path not in ("embed_tokens", "lm_head")

    _, quantized_config = quantize_model(
        model=draft,
        config={},
        group_size=group_size,
        bits=bits,
        quant_predicate=quant_predicate,
    )
    mx.eval(draft.parameters())
    record = dict(quantized_config.get("quantization") or {})
    record["quantize_embeddings"] = quantize_embeddings
    if draft.args.dspark_config is not None:
        draft.args.dspark_config["draft_quantization"] = record
    return record
