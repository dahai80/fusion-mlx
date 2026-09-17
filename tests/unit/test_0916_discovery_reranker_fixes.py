# SPDX-License-Identifier: Apache-2.0
"""Unit tests for #0916 NER/reranker discovery + load fixes.

Covers:
- GLiNER NER models (gliner_config.json, no config.json) discovered + routed
  to NEREngine instead of rejected / mis-routed to BatchedEngine.
- CausalLM reranker name heuristic resolves HF-cache snapshot hash dirs
  (models--org--repo/snapshots/<hash>/) to the real repo id, so
  Qwen3-Reranker is detected as a reranker, not an LLM.
- HF-cache gliner entries accepted as MLX-compatible (gliner package loads
  HF-format safetensors natively, like mlx_audio).
- CausalLM reranker chat_template fallback when tokenizer_config ships none.
"""

from fusion_mlx.engines.reranker import _QWEN3_CHATML_TEMPLATE
from fusion_mlx.pool.model_discovery import (
    _effective_model_name,
    _is_causal_lm_embedding,
    _is_causal_lm_reranker,
    _is_hf_cache_mlx_compatible,
    _is_model_dir,
    detect_model_type,
)


class TestGlinerDiscovery:
    def test_is_model_dir_accepts_gliner_config(self, tmp_path):
        d = tmp_path / "gliner-model"
        d.mkdir()
        (d / "gliner_config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"\x00" * 64)
        assert _is_model_dir(d) is True

    def test_is_model_dir_rejects_empty_dir(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        assert _is_model_dir(d) is False

    def test_detect_model_type_gliner_returns_ner(self, tmp_path):
        d = tmp_path / "gliner-large"
        d.mkdir()
        (d / "gliner_config.json").write_text('{"lr_encoder": 1e-5}')
        (d / "model.safetensors").write_bytes(b"\x00" * 64)
        assert detect_model_type(d) == "ner"

    def test_detect_model_type_gliner_preferred_over_llm_fallback(self, tmp_path):
        # Without the gliner_config check, a dir with no config.json would
        # fall through to "llm". gliner_config.json must take precedence.
        d = tmp_path / "gliner-only"
        d.mkdir()
        (d / "gliner_config.json").write_text("{}")
        assert detect_model_type(d) == "ner"


class TestEffectiveModelName:
    def test_flat_local_layout_uses_dir_name(self, tmp_path):
        d = tmp_path / "Qwen3-Reranker-0.6B"
        d.mkdir()
        assert _effective_model_name(d) == "Qwen3-Reranker-0.6B"

    def test_hf_cache_snapshot_resolves_to_repo_id(self, tmp_path):
        # Real HF cache layout: models--org--repo/snapshots/<hash>/
        root = tmp_path / "models--mlx-community--Qwen3-Reranker-0.6B-4bit"
        snap = root / "snapshots" / "5f324548f1d20c2b5a450f126fc6ef2fb1126524"
        snap.mkdir(parents=True)
        assert _effective_model_name(snap) == "mlx-community/Qwen3-Reranker-0.6B-4bit"

    def test_is_causal_lm_reranker_hf_cache_snapshot(self, tmp_path):
        root = tmp_path / "models--mlx-community--Qwen3-Reranker-0.6B-4bit"
        snap = root / "snapshots" / "abc123hash"
        snap.mkdir(parents=True)
        # snapshot dir name is the hash, which does NOT contain "reranker";
        # the heuristic must resolve the ancestor models-- name.
        assert _is_causal_lm_reranker(snap) is True

    def test_is_causal_lm_reranker_plain_llm_rejected(self, tmp_path):
        root = tmp_path / "models--mlx-community--Qwen3-0.6B-4bit"
        snap = root / "snapshots" / "deadbeef"
        snap.mkdir(parents=True)
        assert _is_causal_lm_reranker(snap) is False

    def test_is_causal_lm_embedding_hf_cache_snapshot(self, tmp_path):
        root = tmp_path / "models--mlx-community--Qwen3-Embedding-0.6B-4bit"
        snap = root / "snapshots" / "feedface"
        snap.mkdir(parents=True)
        assert _is_causal_lm_embedding(snap) is True


class TestHfCacheGlinerCompat:
    def _gliner_snapshot(self, tmp_path):
        root = tmp_path / "models--gliner-community--gliner_large-v2.5"
        snap = root / "snapshots" / "3d6d1760be1c"
        snap.mkdir(parents=True)
        (snap / "gliner_config.json").write_text("{}")
        (snap / "model.fp16.safetensors").write_bytes(b"\x00" * 128)
        return snap

    def test_gliner_hf_cache_accepted(self, tmp_path):
        snap = self._gliner_snapshot(tmp_path)
        assert (
            _is_hf_cache_mlx_compatible(snap, "gliner-community/gliner_large-v2.5")
            is True
        )

    def test_gliner_without_weights_rejected(self, tmp_path):
        root = tmp_path / "models--gliner-community--gliner-x"
        snap = root / "snapshots" / "hash"
        snap.mkdir(parents=True)
        (snap / "gliner_config.json").write_text("{}")
        # no safetensors / bin -> not loadable
        assert _is_hf_cache_mlx_compatible(snap, "gliner-community/gliner-x") is False


class TestRerankerChatTemplateFallback:
    def test_qwen3_chatml_template_is_jinja(self):
        # Must be a valid Jinja2 expression containing the ChatML markers.
        assert "<|im_start|>" in _QWEN3_CHATML_TEMPLATE
        assert "<|im_end|>" in _QWEN3_CHATML_TEMPLATE
        assert "add_generation_prompt" in _QWEN3_CHATML_TEMPLATE

    def test_template_renders_with_sentinel_split(self):
        # The CausalLM reranker load splits the rendered template on a
        # content sentinel to derive prefix/suffix token sequences. Verify
        # the fallback template produces a 2-part split (sentinel appears
        # exactly once in the user turn).
        from jinja2 import Environment

        env = Environment()
        tmpl = env.from_string(_QWEN3_CHATML_TEMPLATE)
        sentinel = "<<__CONTENT_SENTINEL__>>"
        rendered = tmpl.render(
            messages=[
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": sentinel},
            ],
            add_generation_prompt=True,
        )
        parts = rendered.split(sentinel)
        assert len(parts) == 2, f"expected 2 parts, got {len(parts)}: {rendered!r}"
        assert "<|im_start|>assistant" in parts[1]


class TestDeepFilterNetDiscovery:
    def test_detect_model_type_deepfilternet_config_keys(self, tmp_path):
        # DeepFilterNet2-MLX ships config.json with df_order/nb_erb/nb_df and
        # no model_type/architectures. Without the dedicated check it falls
        # through to "llm" -> mlx_lm.load -> KeyError('model_type').
        d = tmp_path / "DeepFilterNet2-MLX"
        d.mkdir()
        (d / "config.json").write_text('{"df_order": 5, "nb_erb": 32, "nb_df": 96}')
        (d / "model.safetensors").write_bytes(b"\x00" * 64)
        assert detect_model_type(d) == "audio_sts"

    def test_detect_model_type_deepfilternet_hf_cache_name(self, tmp_path):
        # HF-cache snapshot dir name is the commit hash; _effective_model_name
        # must resolve the models-- ancestor so "deepfilternet" is detected.
        root = tmp_path / "models--iky1e--DeepFilterNet2-MLX"
        snap = root / "snapshots" / "5c0892ed7e3c"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text('{"sample_rate": 48000}')
        (snap / "model.safetensors").write_bytes(b"\x00" * 64)
        assert detect_model_type(snap) == "audio_sts"

    def test_detect_model_type_kokoro_uses_effective_name(self, tmp_path):
        # Kokoro in HF cache: snapshot hash dir, no istftnet/plbert in config
        # (some conversions). Must still resolve via repo id.
        root = tmp_path / "models--mlx-community--Kokoro-82M"
        snap = root / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text('{"some": "cfg"}')
        (snap / "model.safetensors").write_bytes(b"\x00" * 64)
        assert detect_model_type(snap) == "audio_tts"


class TestVlmChatTemplateFallback:
    def test_load_fallback_chat_template_returns_false_when_no_jinja(
        self, tmp_path, monkeypatch
    ):
        # No chat_template.jinja in cache -> helper returns False (caller raises).
        import huggingface_hub

        from fusion_mlx import _mirror
        from fusion_mlx.engines.vlm import VLMBatchedEngine

        monkeypatch.setattr(_mirror, "_hf_cache_root", lambda: tmp_path)
        monkeypatch.setattr(
            huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None
        )

        class _FakeTok:
            chat_template = None

        eng = VLMBatchedEngine.__new__(VLMBatchedEngine)
        eng._model_name = "org/nonexistent-model-xyz"
        assert eng._load_fallback_chat_template(_FakeTok()) is False

    def test_load_fallback_chat_template_loads_jinja(self, tmp_path, monkeypatch):
        # chat_template.jinja present in the model's HF-cache snapshot dir ->
        # helper sets it on the tokenizer. Build the real cache layout and
        # stub the HF cache lookup (conftest mocks huggingface_hub).
        import huggingface_hub

        from fusion_mlx import _mirror
        from fusion_mlx.engines.vlm import VLMBatchedEngine

        snap = tmp_path / "models--org--any-model" / "snapshots" / "abc123"
        refs = tmp_path / "models--org--any-model" / "refs"
        snap.mkdir(parents=True)
        refs.mkdir()
        (refs / "main").write_text("abc123")
        jinja_file = snap / "chat_template.jinja"
        jinja_file.write_text("{{ messages }}")
        monkeypatch.setattr(_mirror, "_hf_cache_root", lambda: tmp_path)
        monkeypatch.setattr(
            huggingface_hub,
            "try_to_load_from_cache",
            lambda repo, filename, cache_dir=None: str(jinja_file),
        )

        class _FakeTok:
            chat_template = None

        tok = _FakeTok()
        eng = VLMBatchedEngine.__new__(VLMBatchedEngine)
        eng._model_name = "org/any-model"
        assert eng._load_fallback_chat_template(tok) is True
        assert tok.chat_template == "{{ messages }}"
