# SPDX-License-Identifier: Apache-2.0
"""Grammar-constrained decoding via xgrammar or llguidance.

Provides a logits processor that enforces grammar constraints by masking
invalid tokens at sampling time.  Follows the same ``__call__(tokens, logits)``
interface used by :class:`ThinkingBudgetProcessor`.

Backend selection (``grammar_backend`` parameter):
  - ``"auto"`` (default): prefer llguidance > xgrammar > no constraint
  - ``"llguidance"``: use llguidance (LLMatcher) exclusively
  - ``"xgrammar"``: use xgrammar (GrammarMatcher) exclusively

Phase-awareness (thinking vs. output) is handled by the *grammar itself*
via structural tag APIs (xgrammar or llguidance StructTag), not by this
processor.  For thinking models the grammar is compiled as a sequence
of [tag(think_start, any_text, think_end), constrained_schema] so that
the bitmask is permissive during reasoning and constrained during output.

The processor supports two usage modes:

1. **Per-request** (original): call ``processor(tokens, logits)`` directly.
    Handles accept + bitmask fill + mask application in one call.

2. **Batched**: call ``processor.advance(tokens)`` to accept the previous
    token, then use ``BatchGrammarMatcher.batch_fill_next_token_bitmask``
    with the exposed ``matcher`` property to fill bitmasks in parallel
    across the batch, and apply the combined bitmask externally.
"""

import enum
import logging
from functools import lru_cache

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


class GrammarBackend(enum.Enum):
    AUTO = "auto"
    LLGUIDANCE = "llguidance"
    XGRAMMAR = "xgrammar"


def resolve_grammar_backend(requested: str | None = None) -> GrammarBackend:
    """Resolve the grammar backend from a request parameter.

    Priority chain for AUTO: llguidance > xgrammar > unavailable.
    """
    if requested is None or requested == "auto":
        if _is_llguidance_available():
            logger.debug("grammar backend auto-resolved to llguidance")
            return GrammarBackend.LLGUIDANCE
        if _is_xgrammar_available():
            logger.debug("grammar backend auto-resolved to xgrammar")
            return GrammarBackend.XGRAMMAR
        return GrammarBackend.AUTO
    try:
        return GrammarBackend(requested)
    except ValueError:
        logger.warning("unknown grammar_backend %r, falling back to auto", requested)
        return resolve_grammar_backend("auto")


def _is_llguidance_available() -> bool:
    try:
        import llguidance  # noqa: F401

        return True
    except ImportError:
        return False


def _is_xgrammar_available() -> bool:
    try:
        import xgrammar  # noqa: F401

        return True
    except ImportError:
        return False


# ─── Process-level LRU cache for xgrammar GrammarCompiler ───────────────────
# xgrammar's ``TokenizerInfo.from_huggingface`` + ``GrammarCompiler`` init is
# expensive (vocab indexing + bitmask table build). Each engine instance
# (batched / vlm) already memoizes one compiler on ``self._grammar_compiler``,
# but across instances the same compiler was rebuilt.
# ``_get_grammar_compiler_cached`` keys on a stable tokenizer identity plus
# ``vocab_size`` so all engines in this process that share a
# tokenizer/vocab reuse one ``GrammarCompiler``.


@lru_cache(maxsize=8)
def _get_grammar_compiler_cached(
    tokenizer_id: str,
    vocab_size_key: int | None,
) -> object | None:
    """Build (or fetch) a cached ``xgrammar.GrammarCompiler``."""
    from fusion_mlx._torch_stub import install as _install_torch_stub

    _install_torch_stub()
    import xgrammar as xgr

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer_id,
        vocab_size=vocab_size_key if vocab_size_key is not None else None,
    )
    return xgr.GrammarCompiler(tokenizer_info)


def _tokenizer_cache_key(tokenizer) -> str | None:
    """Derive a stable cache key from a tokenizer-like object."""
    for attr in ("name_or_path", "name", "model", "_model_name"):
        cand = getattr(tokenizer, attr, None)
        if cand:
            return str(cand)
    return None


# ─── xgrammar backend ───────────────────────────────────────────────────────


def create_grammar_compiler(tokenizer, model):
    """Create an xgrammar GrammarCompiler for the given tokenizer and model.

    Cached at the process level by ``(tokenizer identity, vocab_size)`` so
    engines sharing a tokenizer family reuse one ``GrammarCompiler`` instead
    of rebuilding the bitmask table per instance. Returns ``None`` if vocab
    size cannot be determined or xgrammar is unavailable.
    """
    from fusion_mlx._torch_stub import install as _install_torch_stub

    _install_torch_stub()
    import xgrammar as xgr  # noqa: F401  (probe import for early failure)

    from fusion_mlx.utils.tokenizer import resolve_vocab_size, unwrap_tokenizer

    hf_tokenizer = unwrap_tokenizer(tokenizer)
    vocab_size = resolve_vocab_size(model)

    tid = _tokenizer_cache_key(hf_tokenizer)
    if tid is None:
        kwargs = {}
        if vocab_size is not None:
            kwargs["vocab_size"] = vocab_size
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(hf_tokenizer, **kwargs)
        return xgr.GrammarCompiler(tokenizer_info)

    try:
        compiler = _get_grammar_compiler_cached(tid, vocab_size)
    except Exception as exc:
        logger.debug("grammar compiler cache miss-build for %s: %s", tid, exc)
        return None
    if compiler is None:
        return None
    return compiler


# ─── llguidance backend ─────────────────────────────────────────────────────


def create_llguidance_matcher(
    tokenizer,
    grammar_spec: dict | str,
    *,
    vocab_size: int | None = None,
) -> object | None:
    """Create an ``llguidance.LLMatcher`` for the given tokenizer and grammar.

    Args:
        tokenizer: HF / mlx-lm tokenizer object.
        grammar_spec: One of:
            - dict with key ``"json_schema"``: JSON schema for structured output
            - dict with key ``"regex"``: regex pattern
            - dict with key ``"choice"``: list of allowed strings
            - dict with key ``"grammar"``: Lark/GBNF grammar string
            - str: treated as a Lark grammar
        vocab_size: Optional vocab size override.

    Returns:
        An ``llguidance.LLMatcher`` instance, or ``None`` if llguidance is
        unavailable or the matcher cannot be built.
    """
    try:
        import llguidance as lg
    except ImportError:
        logger.debug("llguidance not available for grammar compilation")
        return None

    from fusion_mlx.utils.tokenizer import unwrap_tokenizer

    hf_tokenizer = unwrap_tokenizer(tokenizer)

    try:
        ll_tokenizer = _build_ll_tokenizer(hf_tokenizer, vocab_size=vocab_size)
    except Exception as exc:
        logger.warning("failed to build llguidance tokenizer: %s", exc)
        return None

    try:
        grammar_str = _resolve_llguidance_grammar(grammar_spec)
    except Exception as exc:
        logger.warning("failed to resolve llguidance grammar: %s", exc)
        return None

    try:
        matcher = lg.LLMatcher(ll_tokenizer, grammar_str)
        logger.info("llguidance LLMatcher created (vocab_size=%s)", vocab_size)
        return matcher
    except Exception as exc:
        logger.warning("failed to create llguidance LLMatcher: %s", exc)
        return None


class _HfTokenizerForLlguidance:
    """Adapter wrapping an HF tokenizer for llguidance's TokenizerWrapper.

    llguidance expects an object with:
      - ``.tokens``: Sequence[bytes] — raw byte representation of each token
      - ``.eos_token_id``: int
      - ``.bos_token_id``: int | None
      - ``.special_token_ids``: Sequence[int]
      - ``__call__(s)``: returns list[int] token ids
    Standard HF tokenizers lack ``.tokens``, so we provide it here.
    """

    def __init__(self, hf_tokenizer):
        self._hf = hf_tokenizer
        self.eos_token_id: int = hf_tokenizer.eos_token_id
        self.bos_token_id: int | None = getattr(hf_tokenizer, "bos_token_id", None)
        self._vocab = hf_tokenizer.get_vocab()

        special = set()
        if hasattr(hf_tokenizer, "all_special_ids"):
            special.update(hf_tokenizer.all_special_ids)
        if self.eos_token_id is not None:
            special.add(self.eos_token_id)
        if self.bos_token_id is not None:
            special.add(self.bos_token_id)
        self.special_token_ids: list[int] = sorted(special)

        self._byte_decoder = _tokenizer_byte_decoder(hf_tokenizer)
        self.tokens: list[bytes] = []
        for idx in range(len(self._vocab)):
            tok_str = hf_tokenizer.convert_ids_to_tokens(idx)
            self.tokens.append(self._token_bytes(tok_str))

    def _token_bytes(self, tok_str) -> bytes:
        if tok_str is None:
            return b""
        if not isinstance(tok_str, str):
            return bytes(tok_str)
        if self._byte_decoder is not None:
            try:
                return bytes(self._byte_decoder[c] for c in tok_str)
            except KeyError:
                pass
        # SentencePiece marks word starts with U+2581 instead of a space.
        return tok_str.replace("▁", " ").encode("utf-8")

    def __call__(self, s: str) -> list[int]:
        return self._hf.encode(s, add_special_tokens=False)


def _standard_byte_level_decoder() -> dict[str, int]:
    # GPT-2 bytes_to_unicode() inverted: the character → raw byte map used by
    # byte-level BPE vocabularies (Qwen, GPT-2, Llama-3, ...).
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def _tokenizer_byte_decoder(hf_tokenizer) -> dict[str, int] | None:
    # #1441: byte-level BPE tokenizers report vocab strings whose characters are
    # a permutation of the 256 byte values — a leading space is ``Ġ`` (Ġ),
    # not 0x20. Encoding those strings as UTF-8 feeds the WRONG bytes to the
    # matcher, so nearly every space-prefixed token became invalid and the mask
    # forced the model onto the few tokens that happened to match, which is how
    # json_schema guided decoding produced corrupted values ("invoice_no": ": ")
    # instead of the actual field contents. Detect the byte-level alphabet and
    # return the char → byte map so token bytes are exact.
    dec = getattr(hf_tokenizer, "byte_decoder", None)
    if isinstance(dec, dict) and dec:
        try:
            return {str(k): int(v) for k, v in dec.items()}
        except (TypeError, ValueError):
            pass
    try:
        vocab = hf_tokenizer.get_vocab()
    except Exception:
        return None
    if not any("Ġ" in t or "Ċ" in t for t in vocab):
        return None
    return _standard_byte_level_decoder()


_LL_TOKENIZER_CACHE: dict[tuple[int, int | None], tuple[object, object]] = {}
_LL_TOKENIZER_CACHE_MAX = 8


def _build_ll_tokenizer(hf_tokenizer, *, vocab_size: int | None = None):
    # #1441: prefer llguidance's own HF bridge (``llguidance.hf.from_tokenizer``);
    # it serialises the Rust backend tokenizer, so token bytes are exact for
    # byte-level BPE and SentencePiece alike. The hand-rolled wrapper below is
    # only a fallback for llguidance versions that predate that helper.
    import llguidance as lg

    key = (id(hf_tokenizer), vocab_size)
    cached = _LL_TOKENIZER_CACHE.get(key)
    if cached is not None and cached[0] is hf_tokenizer:
        return cached[1]

    ll_tokenizer = None
    hf_bridge = getattr(lg, "hf", None)
    if hf_bridge is not None and hasattr(hf_bridge, "from_tokenizer"):
        try:
            ll_tokenizer = hf_bridge.from_tokenizer(hf_tokenizer, n_vocab=vocab_size)
            logger.info(
                "llguidance tokenizer built via llguidance.hf (vocab_size=%s)",
                vocab_size,
            )
        except Exception as exc:
            logger.warning(
                "llguidance.hf.from_tokenizer failed (%s); falling back to the "
                "hand-rolled tokenizer wrapper",
                exc,
            )

    if ll_tokenizer is None:
        wrapped = _HfTokenizerForLlguidance(hf_tokenizer)
        ll_tokenizer = lg.LLTokenizer(lg.TokenizerWrapper(wrapped))
        logger.info(
            "llguidance tokenizer built via hand-rolled wrapper (vocab_size=%s)",
            vocab_size,
        )

    if len(_LL_TOKENIZER_CACHE) >= _LL_TOKENIZER_CACHE_MAX:
        _LL_TOKENIZER_CACHE.clear()
    _LL_TOKENIZER_CACHE[key] = (hf_tokenizer, ll_tokenizer)
    return ll_tokenizer


def _resolve_llguidance_grammar(grammar_spec: dict | str) -> str:
    """Convert a grammar specification to llguidance grammar string.

    llguidance's ``grammar_from()`` supports: json_schema, json, regex,
    choice, lark, gbnf, ebnf, cfg, grammar, llguidance formats.
    """
    import llguidance as lg

    if isinstance(grammar_spec, str):
        return lg.grammar_from("lark", grammar_spec)

    if isinstance(grammar_spec, dict):
        if "json_schema" in grammar_spec:
            schema = grammar_spec["json_schema"]
            if isinstance(schema, dict):
                import json

                schema = json.dumps(schema)
            return lg.grammar_from("json_schema", schema)

        if "regex" in grammar_spec:
            return lg.grammar_from("regex", grammar_spec["regex"])

        if "choice" in grammar_spec:
            import json

            return lg.grammar_from("choice", json.dumps(grammar_spec["choice"]))

        if "grammar" in grammar_spec:
            text = grammar_spec["grammar"]
            fmt = grammar_spec.get("format", "lark")
            return lg.grammar_from(fmt, text)

    raise ValueError(f"unsupported grammar_spec type: {type(grammar_spec)}")


# ─── Unified GrammarConstraintProcessor ─────────────────────────────────────


class GrammarConstraintProcessor:
    """Logits processor that enforces grammar constraints via bitmask.

    Supports two backends:
      - xgrammar: uses ``GrammarMatcher.fill_next_token_bitmask``
      - llguidance: uses ``LLMatcher.compute_bitmask``

    The external interface (``__call__``, ``accept_token``, ``advance``,
    ``is_terminated``, ``matcher``) is identical regardless of backend.

    Args:
        compiled_grammar: Backend-specific compiled grammar object.
            For xgrammar: an ``xgrammar.CompiledGrammar`` instance.
            For llguidance: an ``llguidance.LLMatcher`` instance.
        vocab_size: Model vocabulary size (from model config, not tokenizer).
        backend: Which backend to use. If ``None``, auto-detected from the
            type of ``compiled_grammar``.
    """

    def __init__(
        self,
        compiled_grammar,
        vocab_size: int,
        *,
        backend: GrammarBackend | None = None,
        eos_token_ids: list[int] | None = None,
    ):
        self._vocab_size = vocab_size
        self._terminated = False
        self._first_call = True
        # Once the grammar reaches an accepting state the only valid
        # continuation is end-of-sequence. Without this the sampler keeps
        # generating unconstrained text past a complete grammar (a JSON
        # string schema produced ``"alpha"`` followed by prose).
        self._eos_ids = [t for t in (eos_token_ids or []) if isinstance(t, int)]

        if backend is None:
            backend = self._detect_backend(compiled_grammar)

        self._backend = backend

        if backend == GrammarBackend.XGRAMMAR:
            self._init_xgrammar(compiled_grammar, vocab_size)
        elif backend == GrammarBackend.LLGUIDANCE:
            self._init_llguidance(compiled_grammar, vocab_size)
        else:
            raise ValueError(f"unsupported grammar backend: {backend}")

    @staticmethod
    def _detect_backend(compiled_grammar) -> GrammarBackend:
        """Auto-detect backend from the compiled grammar type."""
        type_name = type(compiled_grammar).__module__
        if "llguidance" in type_name:
            return GrammarBackend.LLGUIDANCE
        if "xgrammar" in type_name:
            return GrammarBackend.XGRAMMAR
        try:
            import llguidance as lg

            if isinstance(compiled_grammar, lg.LLMatcher):
                return GrammarBackend.LLGUIDANCE
        except ImportError:
            pass
        try:
            import xgrammar as xgr  # noqa: F401 - availability probe

            if hasattr(compiled_grammar, "fill_next_token_bitmask"):
                return GrammarBackend.XGRAMMAR
        except ImportError:
            pass
        return GrammarBackend.XGRAMMAR

    def _init_xgrammar(self, compiled_grammar, vocab_size: int):
        from fusion_mlx._torch_stub import install as _install_torch_stub

        _install_torch_stub()
        import xgrammar as xgr
        from xgrammar.kernels.apply_token_bitmask_mlx import apply_token_bitmask_mlx

        self._matcher = xgr.GrammarMatcher(compiled_grammar)
        self._apply_mask = apply_token_bitmask_mlx
        bitmask_width = (vocab_size + 31) // 32
        self._bitmask = np.full((1, bitmask_width), -1, dtype=np.int32)

    def _init_llguidance(self, ll_matcher, vocab_size: int):
        self._matcher = ll_matcher
        bitmask_width = (vocab_size + 31) // 32
        self._bitmask = np.full((1, bitmask_width), -1, dtype=np.int32)

    # ------------------------------------------------------------------
    # Per-request mode (original interface)
    # ------------------------------------------------------------------

    def __call__(self, tokens, logits: mx.array) -> mx.array:
        """Fill bitmask and apply to logits."""
        if self._terminated:
            return self._eos_only(logits)

        if self._backend == GrammarBackend.XGRAMMAR:
            return self._call_xgrammar(logits)
        elif self._backend == GrammarBackend.LLGUIDANCE:
            return self._call_llguidance(logits)
        return logits

    def _eos_only(self, logits: mx.array) -> mx.array:
        # Grammar complete: allow EOS and nothing else so generation stops on
        # a normal end-of-sequence instead of running on unconstrained.
        if not self._eos_ids:
            return logits
        width = int(logits.shape[-1])
        mask = np.full(width, float("-inf"), dtype=np.float32)
        for tid in self._eos_ids:
            if 0 <= tid < width:
                mask[tid] = 0.0
        mask_mx = mx.array(mask)
        if logits.ndim == 2:
            return logits + mask_mx.reshape(1, -1)
        return logits + mask_mx

    def _mask_vocab(self, logits: mx.array) -> int:
        # The mask must cover the ids the logits can emit. ``self._vocab_size``
        # comes from the model config, which for VLMs can exceed the
        # tokenizer's id range (padded embeddings); a llguidance matcher built
        # from that tokenizer then hands back a narrower bitmask than the
        # logits, and the xgrammar kernel raises broadcast_shapes. Whatever the
        # narrower number is, that is what we can safely mask against.
        width = int(logits.shape[-1]) if len(logits.shape) else self._vocab_size
        return min(self._vocab_size, width)

    @staticmethod
    def _align_words(mask_np: np.ndarray, vocab: int) -> np.ndarray:
        # Pad (with 0 words = deny) or truncate the bitmask to ``vocab``. Padded
        # ids are not decodable tokens, so denying them is correct, and it keeps
        # the bitmask and logits widths equal for the kernel.
        need = (vocab + 31) // 32
        have = int(mask_np.shape[-1])
        if have == need:
            return mask_np
        if have > need:
            logger.debug(
                "grammar: truncating bitmask %d -> %d words (vocab=%d)",
                have,
                need,
                vocab,
            )
            return mask_np[..., :need]
        logger.debug(
            "grammar: padding bitmask %d -> %d words (vocab=%d); padded ids masked out",
            have,
            need,
            vocab,
        )
        pad = np.zeros(mask_np.shape[:-1] + (need - have,), dtype=mask_np.dtype)
        return np.concatenate([mask_np, pad], axis=-1)

    def _call_xgrammar(self, logits: mx.array) -> mx.array:
        vocab = self._mask_vocab(logits)
        self._bitmask.fill(-1)
        self._matcher.fill_next_token_bitmask(self._bitmask)
        mask_np = self._align_words(self._bitmask, vocab)
        return self._apply_mask(mx.array(mask_np), logits, vocab)

    def _call_llguidance(self, logits: mx.array) -> mx.array:
        vocab = self._mask_vocab(logits)
        mask_np = None
        try:
            bitmask = self._matcher.compute_bitmask()
            if bitmask is not None:
                if isinstance(bitmask, bytes):
                    bitmask_width = len(bitmask) // 4
                    mask_np = np.frombuffer(bitmask, dtype=np.int32).reshape(
                        1, bitmask_width
                    )
                elif isinstance(bitmask, np.ndarray):
                    mask_np = bitmask
                else:
                    mask_np = np.array(bitmask, dtype=np.int32)
        except Exception:
            logger.debug("llguidance compute_bitmask failed, using permissive mask")
        if mask_np is None:
            self._bitmask.fill(-1)
            mask_np = self._bitmask

        mx_bitmask = mx.array(self._align_words(mask_np, vocab))
        try:
            from xgrammar.kernels.apply_token_bitmask_mlx import apply_token_bitmask_mlx

            return apply_token_bitmask_mlx(mx_bitmask, logits, vocab)
        except ImportError:
            return self._apply_bitmask_manual(mx_bitmask, logits, vocab)

    def _apply_bitmask_manual(
        self, bitmask: mx.array, logits: mx.array, vocab: int | None = None
    ) -> mx.array:
        """Fallback bitmask application when xgrammar kernel is unavailable."""
        neg_inf = float("-inf")
        if vocab is None:
            vocab = self._mask_vocab(logits)
        flat = np.array(bitmask).reshape(-1)
        allowed = np.zeros(vocab, dtype=bool)
        for i in range(0, vocab, 32):
            if i // 32 >= flat.shape[0]:
                break
            word = int(flat[i // 32])
            for bit in range(32):
                idx = i + bit
                if idx >= vocab:
                    break
                if word & (1 << bit):
                    allowed[idx] = True
        # Slots between ``vocab`` and the logits width are padding embeddings:
        # the tokenizer never emits them, so they stay masked out.
        width = int(logits.shape[-1]) if len(logits.shape) else vocab
        mask = np.full(width, neg_inf, dtype=np.float32)
        mask[:vocab] = np.where(allowed, 0.0, neg_inf)
        mask_mx = mx.array(mask)
        if logits.ndim == 2:
            return logits + mask_mx.reshape(1, -1)
        return logits + mask_mx

    def accept_token(self, token_id: int) -> None:
        """Accept a generated token to advance matcher state."""
        if self._terminated:
            return

        if self._backend == GrammarBackend.XGRAMMAR:
            if not self._matcher.accept_token(token_id):
                logger.warning("GrammarMatcher rejected token %d", token_id)
            if self._matcher.is_terminated():
                self._terminated = True
        elif self._backend == GrammarBackend.LLGUIDANCE:
            self._matcher.consume_token(token_id)
            if self._matcher.is_stopped() or self._matcher.is_error():
                self._terminated = True

    # ------------------------------------------------------------------
    # Batched mode helpers
    # ------------------------------------------------------------------

    @property
    def matcher(self):
        """Return the underlying matcher (xgrammar GrammarMatcher or llguidance LLMatcher)."""
        return self._matcher

    @property
    def backend(self) -> GrammarBackend:
        """Return the active grammar backend."""
        return self._backend

    @property
    def is_terminated(self) -> bool:
        return self._terminated

    def advance(self, tokens: mx.array) -> bool:
        """Accept the previous token and advance grammar state.

        Call this *instead of* ``__call__`` when using batched bitmask
        filling.  Returns ``True`` if the matcher is still active (not
        terminated) and should participate in the next
        ``batch_fill_next_token_bitmask`` call.
        """
        if self._terminated:
            return False

        if self._first_call:
            self._first_call = False
        elif len(tokens) > 0:
            last_token = int(tokens[-1])
            if self._backend == GrammarBackend.XGRAMMAR:
                if not self._matcher.accept_token(last_token):
                    logger.warning("GrammarMatcher rejected token %d", last_token)
                if self._matcher.is_terminated():
                    self._terminated = True
                    return False
            elif self._backend == GrammarBackend.LLGUIDANCE:
                self._matcher.consume_token(last_token)
                if self._matcher.is_stopped() or self._matcher.is_error():
                    self._terminated = True
                    return False

        return True
