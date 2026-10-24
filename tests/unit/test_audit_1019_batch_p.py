# SPDX-License-Identifier: Apache-2.0
"""Regression tests for audit 1019 L-series findings (batch P).

L5  — SpecPrefill state lives on ``Request`` only, not duplicated on the
      public ``SamplingParams`` dataclass.
L8  — stream error frames are JSON-encoded and every error path ends the
      stream with ``[DONE]``.
L9  — ``/api/version`` is authenticated and model digests are stable.
L27 — the chat subprocess inherits the bound socket instead of racing to
      rebind the port.
L30 — patch failures come back to the caller; spec-route decisions are
      recorded through one helper.
L31 — the video fallbacks log what they degraded to instead of passing.
"""

import ast
import inspect
import json
import logging
import pathlib

_REPO = pathlib.Path(__file__).resolve().parents[2]


class TestSamplingParamsHasNoSpecPrefillState:
    def test_no_specprefill_fields(self):
        from dataclasses import fields

        from fusion_mlx.request import SamplingParams

        names = {f.name for f in fields(SamplingParams)}
        assert not {n for n in names if "specprefill" in n}

    def test_request_still_carries_system_end(self):
        from dataclasses import fields

        from fusion_mlx.request import Request

        assert "specprefill_system_end" in {f.name for f in fields(Request)}


class TestStreamErrorFrames:
    def test_frame_is_valid_json_for_hostile_message(self):
        from fusion_mlx.api.openai.streaming import _sse_error_frame

        message = 'bad "quote" \\ back\nslash'
        frame = _sse_error_frame(message, 400, "server_busy")
        assert frame.startswith("data: ") and frame.endswith("\n\n")
        payload = json.loads(frame[len("data: ") : -2])
        assert payload["error"] == {
            "message": message,
            "status": 400,
            "type": "server_busy",
        }

    def test_no_type_key_when_omitted(self):
        from fusion_mlx.api.openai.streaming import _sse_error_frame

        payload = json.loads(_sse_error_frame("x", 404)[len("data: ") : -2])
        assert "type" not in payload["error"]

    def test_no_hand_built_error_fstrings_left(self):
        source = (_REPO / "fusion_mlx/api/openai/streaming.py").read_text()
        tree = ast.parse(source)
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.JoinedStr)
            and any(
                isinstance(v, ast.Constant)
                and isinstance(v.value, str)
                and '"error"' in v.value
                for v in node.values
            )
        ]
        assert offenders == []

    def test_error_arms_terminate_the_stream(self):
        source = (_REPO / "fusion_mlx/api/openai/streaming.py").read_text()
        tree = ast.parse(source)
        generator = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef)
            and n.name == "_stream_chat_generator"
        )
        error_handlers = [
            h
            for n in ast.walk(generator)
            if isinstance(n, ast.Try)
            for h in n.handlers
            if not _body_reraises(h)
            and any(isinstance(x, ast.Yield) for x in ast.walk(h))
        ]
        assert error_handlers
        for handler in error_handlers:
            text = ast.unparse(handler)
            assert "_SSE_DONE" in text, f"handler at line {handler.lineno} lacks [DONE]"


def _body_reraises(handler):
    return any(isinstance(n, ast.Raise) for n in ast.walk(handler))


class TestOllamaVersionAndDigests:
    def test_version_requires_auth(self):
        from fusion_mlx.api.ollama_routes import api_version
        from fusion_mlx.middleware.auth import verify_api_key

        params = inspect.signature(api_version).parameters
        deps = [
            p.default.dependency
            for p in params.values()
            if type(p.default).__name__ == "Depends"
        ]
        assert verify_api_key in deps

    def test_digest_is_stable_and_per_model(self):
        from fusion_mlx.api.ollama_routes import _model_digest

        assert _model_digest("qwen3.5-4b") == _model_digest("qwen3.5-4b")
        assert _model_digest("qwen3.5-4b") != _model_digest("qwen3.5-9b")
        assert _model_digest("qwen3.5-4b").startswith("sha256:")


class TestChatSpawnSocketActivation:
    def test_child_inherits_bound_socket(self, monkeypatch, tmp_path):
        import subprocess

        import fusion_mlx.cli_commands as cc

        captured = {}

        class _Proc:
            pass

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return _Proc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        cc._spawn_chat_server("m", str(tmp_path / "chat.log"))

        cmd = captured["cmd"]
        assert "--listen-fd" in cmd
        fd_arg = cmd[cmd.index("--listen-fd") + 1]
        assert captured["kwargs"]["pass_fds"] == (int(fd_arg),)
        assert "--port" in cmd

    def test_out_of_range_fd_falls_back_to_rebind(self, monkeypatch, tmp_path):
        import subprocess

        import fusion_mlx.cli_commands as cc

        class _Sock:
            def bind(self, addr):
                pass

            def getsockname(self):
                return ("127.0.0.1", 45678)

            def fileno(self):
                return 2000

            def close(self):
                pass

        import socket as socket_mod

        monkeypatch.setattr(socket_mod, "socket", lambda *a, **k: _Sock())
        captured = {}

        class _Proc:
            pass

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return _Proc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        cc._spawn_chat_server("m", str(tmp_path / "chat.log"))

        assert "--listen-fd" not in captured["cmd"]
        assert captured["kwargs"]["pass_fds"] == ()


class TestPatchRegistryReportsFailures:
    def test_failed_patch_is_returned_and_logged(self, caplog):
        from fusion_mlx.patches import registry

        registry.reset_for_tests()

        def boom(model, config):
            raise RuntimeError("patch exploded")

        registry.register("boom", "family-x", "raises", boom)
        with caplog.at_level(logging.ERROR, logger="fusion_mlx.patches.registry"):
            failed = registry.apply_all_for_model("family-x")

        assert failed == ["boom"]
        assert "patch exploded" in caplog.text
        assert "NOT active" in caplog.text
        assert registry.list_patches()[0]["is_applied"] is False

    def test_successful_patch_returns_empty(self):
        from fusion_mlx.patches import registry

        registry.reset_for_tests()
        registry.register("ok", "family-y", "noop", lambda m, c: None)
        assert registry.apply_all_for_model("family-y") == []


class TestSpecRouteDecisionRecords:
    @staticmethod
    def _capture(monkeypatch):
        from fusion_mlx.speculative import auto_router

        records = []
        monkeypatch.setattr(
            auto_router,
            "_append_spec_route_decision",
            lambda decision: records.append(decision),
        )
        return auto_router, records

    def test_long_doc_route_records_full_signal_set(self, monkeypatch):
        auto_router, records = self._capture(monkeypatch)
        router = auto_router.SpecAutoRouter()
        signals = auto_router.RouteSignals(prompt_token_count=9000)
        assert router.decide(signals) == auto_router.METHOD_DFLASH
        assert records == [
            {
                "method": auto_router.METHOD_DFLASH,
                "reason": "long_doc threshold",
                "candidates": sorted(auto_router.DEFAULT_AVAILABLE),
                "model_family": None,
                "prompt_tokens": 9000,
                "accept_rate": None,
                "kv_pressure": None,
            }
        ]

    def test_ngram_fallback_records_reason(self, monkeypatch):
        auto_router, records = self._capture(monkeypatch)
        router = auto_router.SpecAutoRouter()
        signals = auto_router.RouteSignals(
            prompt_token_count=10,
            available=frozenset({auto_router.METHOD_NGRAM}),
        )
        assert router.decide(signals) == auto_router.METHOD_NGRAM
        assert records[0]["reason"] == "ngram fallback"


class TestVideoFallbackVisibility:
    def test_model_path_logs_download_reason(self, monkeypatch, caplog):
        import huggingface_hub

        from fusion_mlx.video.ltx2 import utils

        calls = []

        def fake_snapshot_download(**kwargs):
            calls.append(kwargs)
            if kwargs.get("local_files_only"):
                raise RuntimeError("no local snapshot")
            return "/tmp/fake-model-dir"

        monkeypatch.setattr(
            huggingface_hub, "snapshot_download", fake_snapshot_download
        )
        with caplog.at_level(logging.INFO, logger="fusion_mlx.video.ltx2.utils"):
            utils.get_model_path("org/model-that-does-not-exist-locally")

        assert len(calls) == 2
        assert "downloading model weights" in caplog.text
        assert "no local snapshot" in caplog.text

    def test_tokenizer_fallback_warns(self, monkeypatch, caplog, tmp_path):
        import transformers

        from fusion_mlx.video.ltx2 import text_encoder

        class _Tok:
            padding_side = "right"

        def fake_from_pretrained(path, **kwargs):
            if path == "google/gemma-3-12b-it":
                return _Tok()
            raise OSError("tokenizer files missing")

        monkeypatch.setattr(
            transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained
        )
        with caplog.at_level(
            logging.WARNING, logger="fusion_mlx.video.ltx2.text_encoder"
        ):
            proc = text_encoder._load_processor(tmp_path, "/fake/text_encoder")

        assert isinstance(proc, _Tok)
        assert "falling back to google/gemma-3-12b-it" in caplog.text

    def test_tokenizer_dir_preferred_without_warning(
        self, monkeypatch, caplog, tmp_path
    ):
        import transformers

        from fusion_mlx.video.ltx2 import text_encoder

        (tmp_path / "tokenizer").mkdir()
        seen = []
        monkeypatch.setattr(
            transformers.AutoTokenizer,
            "from_pretrained",
            lambda path, **kw: seen.append(path) or "tok",
        )
        with caplog.at_level(
            logging.WARNING, logger="fusion_mlx.video.ltx2.text_encoder"
        ):
            assert text_encoder._load_processor(tmp_path, "/fake/te") == "tok"
        assert seen == [str(tmp_path / "tokenizer")]
        assert "falling back" not in caplog.text

    def test_decode_tiled_warns_when_tile_size_rejected(self, monkeypatch, caplog):
        from fusion_mlx.video.ltx2.video_vae import decoder as decoder_mod
        from fusion_mlx.video.ltx2_5.video_vae import LTX2_5VideoVAE

        class _Spatial:
            @property
            def tile_size_in_pixels(self):
                return 256

            @tile_size_in_pixels.setter
            def tile_size_in_pixels(self, value):
                raise AttributeError("tile size is frozen")

        class _Cfg:
            spatial_config = _Spatial()

            @staticmethod
            def default():
                return _Cfg()

        class _Decoder:
            def __init__(self):
                self.calls = []

            def decode_tiled(self, latent, tiling_config=None):
                self.calls.append(tiling_config)
                return "decoded"

        monkeypatch.setattr(decoder_mod, "TilingConfig", _Cfg)
        vae = object.__new__(LTX2_5VideoVAE)
        vae.decoder = _Decoder()
        with caplog.at_level(
            logging.WARNING, logger="fusion_mlx.video.ltx2_5.video_vae"
        ):
            assert vae.decode_tiled("latent", tile_size=512) == "decoded"
        assert len(vae.decoder.calls) == 1
        assert "could not set tile_size=512" in caplog.text
