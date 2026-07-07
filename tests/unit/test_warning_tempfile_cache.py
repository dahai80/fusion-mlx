# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.share.warning, _tempfile_safe, runtime.cache helpers."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

# ── share/warning.py ─────────────────────────────────────────────────
from fusion_mlx.share import warning as warn_mod


class TestSupportsColor:
    def test_tty_returns_true(self):
        with patch("sys.stdout.isatty", return_value=True):
            with patch("sys.platform", "linux"):
                assert warn_mod._supports_color() is True

    def test_non_tty_returns_false(self):
        with patch("sys.stdout.isatty", return_value=False):
            assert warn_mod._supports_color() is False

    def test_windows_returns_false(self):
        with patch("sys.stdout.isatty", return_value=True):
            with patch("sys.platform", "win32"):
                assert warn_mod._supports_color() is False


class TestRender:
    def test_minimal_no_chat_frontend(self):
        with patch.object(warn_mod, "_supports_color", return_value=False):
            result = warn_mod.render(
                url="http://x.com",
                api_key="key123",
                model="qwen",
                tunnel_id="t1",
                chat_frontend=None,
            )
        assert "Fusion-MLX share" in result
        assert "http://x.com" in result
        assert "key123" in result
        assert "qwen" in result
        assert "Chat:" not in result  # no chat_frontend

    def test_with_chat_frontend(self):
        with patch.object(warn_mod, "_supports_color", return_value=False):
            result = warn_mod.render(
                url="http://x.com",
                api_key="k",
                model="m",
                tunnel_id="t1",
                chat_frontend="https://chat.example.com",
            )
        assert "Chat:" in result
        assert "https://chat.example.com/#k=t1.k" in result

    def test_color_codes_present_when_tty(self):
        with patch.object(warn_mod, "_supports_color", return_value=True):
            result = warn_mod.render("u", "k", "m", "t", None)
        assert "\033[1;31m" in result  # red
        assert "\033[0m" in result  # reset

    def test_safe_url_quoted(self):
        with patch.object(warn_mod, "_supports_color", return_value=False):
            result = warn_mod.render("http://x.com", "k", "m", "t", None)
        assert "http://x.com/v1/chat/completions" in result


# ── _tempfile_safe.py ────────────────────────────────────────────────


# ── runtime/cache.py ─────────────────────────────────────────────────


from fusion_mlx.runtime import cache as cache_mod


class TestShutdownBudgetSec:
    def test_default_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_PREFIX_CACHE_SHUTDOWN_BUDGET", raising=False)
        assert cache_mod._shutdown_budget_sec() == 3.5

    def test_env_value_parsed(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_PREFIX_CACHE_SHUTDOWN_BUDGET", "10.5")
        assert cache_mod._shutdown_budget_sec() == 10.5

    def test_env_negative_clamped_to_zero(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_PREFIX_CACHE_SHUTDOWN_BUDGET", "-5")
        assert cache_mod._shutdown_budget_sec() == 0.0

    def test_env_invalid_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_PREFIX_CACHE_SHUTDOWN_BUDGET", "not-a-number")
        with patch("fusion_mlx.runtime.cache.logger.warning"):
            assert cache_mod._shutdown_budget_sec() == 3.5


class TestMakeShouldAbort:
    def test_returns_predicate(self):
        pred = cache_mod._make_should_abort(10.0)
        assert callable(pred)

    def test_predicate_false_within_budget(self):
        pred = cache_mod._make_should_abort(100.0)
        assert pred() is False

    def test_predicate_true_when_exceeded(self):
        pred = cache_mod._make_should_abort(0.0)  # already past deadline
        assert pred() is True

    def test_predicate_with_predicted_sec(self):
        pred = cache_mod._make_should_abort(100.0)
        assert pred(200.0) is True  # predicted pushes past deadline


class TestGetCacheDir:
    def test_returns_path_under_home(self, monkeypatch):
        cfg = MagicMock()
        cfg.scheduler.model_name = "qwen-test"
        monkeypatch.setattr(cache_mod, "get_config", lambda *a, **k: cfg)
        result = cache_mod.get_cache_dir()
        assert ".cache/fusion-mlx/prefix_cache" in result
        assert "qwen-test" in result

    def test_empty_model_name_uses_default(self, monkeypatch):
        cfg = MagicMock()
        cfg.scheduler.model_name = ""
        monkeypatch.setattr(cache_mod, "get_config", lambda *a, **k: cfg)
        result = cache_mod.get_cache_dir()
        assert "default" in result

    def test_slashes_replaced(self, monkeypatch):
        cfg = MagicMock()
        cfg.scheduler.model_name = "org/model/name"
        monkeypatch.setattr(cache_mod, "get_config", lambda *a, **k: cfg)
        result = cache_mod.get_cache_dir()
        assert "/" not in result.split("prefix_cache/")[-1]  # leaf has no slash

    def test_dot_dot_replaced(self, monkeypatch):
        cfg = MagicMock()
        cfg.scheduler.model_name = ".."
        monkeypatch.setattr(cache_mod, "get_config", lambda *a, **k: cfg)
        result = cache_mod.get_cache_dir()
        leaf = result.split("/")[-1]
        assert not leaf.startswith(".")  # lstrip removes leading dots

    def test_digest_in_path(self, monkeypatch):
        cfg = MagicMock()
        cfg.scheduler.model_name = "qwen"
        monkeypatch.setattr(cache_mod, "get_config", lambda *a, **k: cfg)
        import hashlib

        expected_digest = hashlib.sha256(b"qwen").hexdigest()[:8]
        result = cache_mod.get_cache_dir()
        assert expected_digest in result


class TestResolveMemoryAwareCache:
    def test_no_scheduler_returns_none(self):
        engine = MagicMock()
        del engine.scheduler  # no scheduler attr
        with patch.object(engine, "scheduler", None, create=True):
            assert cache_mod._resolve_memory_aware_cache(engine) is None

    def test_scheduler_no_cache_returns_none(self):
        engine = MagicMock()
        engine.scheduler.memory_aware_cache = None
        assert cache_mod._resolve_memory_aware_cache(engine) is None

    def test_returns_cache(self):
        engine = MagicMock()
        engine.scheduler.memory_aware_cache = MagicMock()
        assert (
            cache_mod._resolve_memory_aware_cache(engine)
            is engine.scheduler.memory_aware_cache
        )


class TestLoadRadixIndexAfterCache:
    def test_no_cache_returns(self):
        engine = MagicMock()
        with patch.object(cache_mod, "_resolve_memory_aware_cache", return_value=None):
            cache_mod._load_radix_index_after_cache(engine, "/tmp")  # no crash

    def test_no_radix_returns(self):
        engine = MagicMock()
        cache = MagicMock()
        cache._radix_index = None
        with patch.object(cache_mod, "_resolve_memory_aware_cache", return_value=cache):
            cache_mod._load_radix_index_after_cache(engine, "/tmp")

    def test_radix_load_success(self):
        engine = MagicMock()
        cache = MagicMock()
        radix = MagicMock()
        cache._radix_index = radix
        radix.load.return_value = True  # load succeeded
        with patch.object(cache_mod, "_resolve_memory_aware_cache", return_value=cache):
            cache_mod._load_radix_index_after_cache(engine, "/tmp")
            radix.load.assert_called_once()

    def test_radix_rebuild_from_keys(self):
        engine = MagicMock()
        cache = MagicMock()
        radix = MagicMock()
        cache._radix_index = radix
        radix.load.return_value = False  # load failed → rebuild
        cache._lock = MagicMock()
        cache._entries = {"k1": 1, "k2": 2}
        with patch.object(cache_mod, "_resolve_memory_aware_cache", return_value=cache):
            cache_mod._load_radix_index_after_cache(engine, "/tmp")
            radix.rebuild_from_keys.assert_called_once()

    def test_radix_rebuild_empty_entries_skips(self):
        engine = MagicMock()
        cache = MagicMock()
        radix = MagicMock()
        cache._radix_index = radix
        radix.load.return_value = False
        cache._lock = MagicMock()
        cache._entries = {}
        with patch.object(cache_mod, "_resolve_memory_aware_cache", return_value=cache):
            cache_mod._load_radix_index_after_cache(engine, "/tmp")
            radix.rebuild_from_keys.assert_not_called()

    def test_rebuild_exception_caught(self):
        engine = MagicMock()
        cache = MagicMock()
        radix = MagicMock()
        cache._radix_index = radix
        radix.load.return_value = False
        cache._lock = MagicMock()
        cache._entries = MagicMock()
        cache._entries.keys.side_effect = RuntimeError("boom")
        with patch.object(cache_mod, "_resolve_memory_aware_cache", return_value=cache):
            with patch("fusion_mlx.runtime.cache.logger.warning"):
                cache_mod._load_radix_index_after_cache(engine, "/tmp")  # no raise


class TestSaveRadixIndexAfterCache:
    def test_no_cache_returns(self):
        engine = MagicMock()
        with patch.object(cache_mod, "_resolve_memory_aware_cache", return_value=None):
            cache_mod._save_radix_index_after_cache(engine, "/tmp")

    def test_no_radix_returns(self):
        engine = MagicMock()
        cache = MagicMock()
        cache._radix_index = None
        with patch.object(cache_mod, "_resolve_memory_aware_cache", return_value=cache):
            cache_mod._save_radix_index_after_cache(engine, "/tmp")

    def test_save_success(self):
        engine = MagicMock()
        cache = MagicMock()
        radix = MagicMock()
        cache._radix_index = radix
        with patch.object(cache_mod, "_resolve_memory_aware_cache", return_value=cache):
            cache_mod._save_radix_index_after_cache(engine, "/tmp")
            radix.save.assert_called_once_with(os.path.join("/tmp", "radix.index"))

    def test_save_exception_caught(self):
        engine = MagicMock()
        cache = MagicMock()
        radix = MagicMock()
        cache._radix_index = radix
        radix.save.side_effect = OSError("disk full")
        with patch.object(cache_mod, "_resolve_memory_aware_cache", return_value=cache):
            with patch("fusion_mlx.runtime.cache.logger.warning"):
                cache_mod._save_radix_index_after_cache(engine, "/tmp")  # no raise


class TestLoadPrefixCacheFromDisk:
    def test_no_engine_returns(self, monkeypatch):
        cfg = MagicMock()
        monkeypatch.setattr(cache_mod, "get_config", lambda *a, **k: cfg)
        with patch.object(cache_mod, "_get_engine", return_value=None):
            cache_mod.load_prefix_cache_from_disk()  # no crash

    def test_engine_loads_cache(self, monkeypatch, tmp_path):
        cfg = MagicMock()
        monkeypatch.setattr(cache_mod, "get_config", lambda *a, **k: cfg)
        engine = MagicMock()
        engine.load_cache_from_disk.return_value = 5  # loaded 5 entries
        with patch.object(cache_mod, "_get_engine", return_value=engine):
            with patch.object(cache_mod, "get_cache_dir", return_value=str(tmp_path)):
                with patch.object(cache_mod, "_load_radix_index_after_cache"):
                    cache_mod.load_prefix_cache_from_disk()
                    engine.load_cache_from_disk.assert_called_once()

    def test_engine_no_entries(self, monkeypatch, tmp_path):
        cfg = MagicMock()
        monkeypatch.setattr(cache_mod, "get_config", lambda *a, **k: cfg)
        engine = MagicMock()
        engine.load_cache_from_disk.return_value = 0
        with patch.object(cache_mod, "_get_engine", return_value=engine):
            with patch.object(cache_mod, "get_cache_dir", return_value=str(tmp_path)):
                with patch.object(cache_mod, "_load_radix_index_after_cache"):
                    cache_mod.load_prefix_cache_from_disk()

    def test_load_exception_caught(self, monkeypatch, tmp_path):
        cfg = MagicMock()
        monkeypatch.setattr(cache_mod, "get_config", lambda *a, **k: cfg)
        engine = MagicMock()
        engine.load_cache_from_disk.side_effect = RuntimeError("boom")
        with patch.object(cache_mod, "_get_engine", return_value=engine):
            with patch.object(cache_mod, "get_cache_dir", return_value=str(tmp_path)):
                with patch("fusion_mlx.runtime.cache.logger.warning"):
                    cache_mod.load_prefix_cache_from_disk()  # no raise


class TestSavePrefixCacheToDisk:
    def test_no_engine_returns(self, monkeypatch):
        cfg = MagicMock()
        monkeypatch.setattr(cache_mod, "get_config", lambda *a, **k: cfg)
        with patch.object(cache_mod, "_get_engine", return_value=None):
            cache_mod.save_prefix_cache_to_disk()  # no crash

    def test_with_budget_calls_save(self, monkeypatch, tmp_path):
        cfg = MagicMock()
        monkeypatch.setattr(cache_mod, "get_config", lambda *a, **k: cfg)
        engine = MagicMock()
        with patch.object(cache_mod, "_get_engine", return_value=engine):
            with patch.object(cache_mod, "get_cache_dir", return_value=str(tmp_path)):
                with patch.object(
                    cache_mod, "_call_save_cache_to_disk", return_value=True
                ):
                    with patch.object(cache_mod, "_save_radix_index_after_cache"):
                        cache_mod.save_prefix_cache_to_disk(budget_sec=10.0)

    def test_no_budget_no_should_abort(self, monkeypatch, tmp_path):
        cfg = MagicMock()
        monkeypatch.setattr(cache_mod, "get_config", lambda *a, **k: cfg)
        engine = MagicMock()
        with patch.object(cache_mod, "_get_engine", return_value=engine):
            with patch.object(cache_mod, "get_cache_dir", return_value=str(tmp_path)):
                with patch.object(
                    cache_mod, "_call_save_cache_to_disk", return_value=True
                ):
                    with patch.object(cache_mod, "_save_radix_index_after_cache"):
                        cache_mod.save_prefix_cache_to_disk(budget_sec=0.0)

    def test_save_exception_caught(self, monkeypatch, tmp_path):
        cfg = MagicMock()
        monkeypatch.setattr(cache_mod, "get_config", lambda *a, **k: cfg)
        engine = MagicMock()
        with patch.object(cache_mod, "_get_engine", return_value=engine):
            with patch.object(cache_mod, "get_cache_dir", return_value=str(tmp_path)):
                with patch.object(
                    cache_mod,
                    "_call_save_cache_to_disk",
                    side_effect=RuntimeError("boom"),
                ):
                    with patch("fusion_mlx.runtime.cache.logger.warning"):
                        cache_mod.save_prefix_cache_to_disk()  # no raise
