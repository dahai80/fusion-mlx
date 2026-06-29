import pytest
import gc
import weakref
from unittest.mock import MagicMock

from fusion_mlx.model_registry import (
    ModelOwnershipError,
    ModelRegistry,
    get_registry,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    registry = get_registry()
    registry._owners.clear()
    yield
    registry._owners.clear()


class TestModelRegistry:
    def test_singleton(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_singleton_direct(self):
        r1 = ModelRegistry()
        r2 = ModelRegistry()
        assert r1 is r2

    def test_acquire_returns_true(self):
        registry = get_registry()
        engine = MagicMock()
        model = object()
        result = registry.acquire(model, engine, "test_engine")
        assert result is True

    def test_acquire_sets_ownership(self):
        registry = get_registry()
        engine = MagicMock()
        model = object()
        registry.acquire(model, engine, "test_engine")
        owned, owner = registry.is_owned(model)
        assert owned is True
        assert owner == "test_engine"

    def test_acquire_already_owned_same_engine(self):
        registry = get_registry()
        engine = MagicMock()
        model = object()
        registry.acquire(model, engine, "engine1")
        result = registry.acquire(model, engine, "engine1")
        assert result is True

    def test_acquire_already_owned_different_engine(self):
        registry = get_registry()
        engine1 = MagicMock()
        engine2 = MagicMock()
        model = object()
        registry.acquire(model, engine1, "engine1")
        with pytest.raises(ModelOwnershipError) as exc_info:
            registry.acquire(model, engine2, "engine2")
        assert "engine1" in str(exc_info.value)

    def test_release_owned_model(self):
        registry = get_registry()
        engine = MagicMock()
        model = object()
        registry.acquire(model, engine, "test_engine")
        result = registry.release(model, "test_engine")
        assert result is True
        owned, owner = registry.is_owned(model)
        assert owned is False
        assert owner is None

    def test_release_not_owned(self):
        registry = get_registry()
        model = object()
        result = registry.release(model, "some_engine")
        assert result is False

    def test_release_wrong_engine(self):
        registry = get_registry()
        engine = MagicMock()
        model = object()
        registry.acquire(model, engine, "engine1")
        result = registry.release(model, "engine2")
        assert result is False

    def test_is_owned_not_owned(self):
        registry = get_registry()
        model = object()
        owned, owner = registry.is_owned(model)
        assert owned is False
        assert owner is None

    def test_force_transfer_ownership(self):
        registry = get_registry()
        engine1 = MagicMock()
        engine2 = MagicMock()
        model = object()
        registry.acquire(model, engine1, "engine1")
        result = registry.acquire(model, engine2, "engine2", force=True)
        assert result is True
        owned, owner = registry.is_owned(model)
        assert owned is True
        assert owner == "engine2"

    def test_weak_reference_cleanup(self):
        registry = get_registry()
        engine = MagicMock()
        model = object()
        registry.acquire(model, engine, "test_engine")
        del engine
        gc.collect()
        owned, owner = registry.is_owned(model)
        assert owned is False


class TestMultiEngine:
    def test_multiple_models_same_engine(self):
        registry = get_registry()
        engine = MagicMock()
        model1 = object()
        model2 = object()
        registry.acquire(model1, engine, "engine1")
        registry.acquire(model2, engine, "engine1")
        owned1, _ = registry.is_owned(model1)
        owned2, _ = registry.is_owned(model2)
        assert owned1 is True
        assert owned2 is True

    def test_multiple_engines_different_models(self):
        registry = get_registry()
        engine1 = MagicMock()
        engine2 = MagicMock()
        model1 = object()
        model2 = object()
        registry.acquire(model1, engine1, "engine1")
        registry.acquire(model2, engine2, "engine2")
        owned1, owner1 = registry.is_owned(model1)
        owned2, owner2 = registry.is_owned(model2)
        assert owned1 is True
        assert owner1 == "engine1"
        assert owned2 is True
        assert owner2 == "engine2"

    def test_concurrent_acquire_attempt(self):
        registry = get_registry()
        engine1 = MagicMock()
        engine2 = MagicMock()
        model = object()
        registry.acquire(model, engine1, "engine1")
        with pytest.raises(ModelOwnershipError):
            registry.acquire(model, engine2, "engine2")


class TestCacheRecovery:
    def test_cleanup_removes_dead_references(self):
        registry = get_registry()
        engine = MagicMock()
        model = object()
        registry.acquire(model, engine, "test_engine")
        del engine
        gc.collect()
        removed = registry.cleanup()
        assert removed >= 1

    def test_cleanup_keeps_live_references(self):
        registry = get_registry()
        engine = MagicMock()
        model = object()
        registry.acquire(model, engine, "test_engine")
        removed = registry.cleanup()
        assert removed == 0
        owned, owner = registry.is_owned(model)
        assert owned is True
        assert owner == "test_engine"


class TestModelRegistryEdgeCases:
    @pytest.mark.skip(reason="Cannot create weak reference to NoneType")
    def test_acquire_with_none_engine(self):
        pass

    def test_get_stats_empty(self):
        registry = get_registry()
        stats = registry.get_stats()
        assert isinstance(stats, dict)
        assert "total_entries" in stats
        assert stats["total_entries"] == 0

    def test_get_stats_with_models(self):
        registry = get_registry()
        engine = MagicMock()
        model1 = object()
        model2 = object()
        registry.acquire(model1, engine, "engine1")
        registry.acquire(model2, engine, "engine1")
        stats = registry.get_stats()
        assert stats["total_entries"] >= 2

    def test_reacquire_after_release(self):
        registry = get_registry()
        engine1 = MagicMock()
        engine2 = MagicMock()
        model = object()
        registry.acquire(model, engine1, "engine1")
        registry.release(model, "engine1")
        result = registry.acquire(model, engine2, "engine2")
        assert result is True
        owned, owner = registry.is_owned(model)
        assert owned is True
        assert owner == "engine2"

    def test_model_ownership_error_message(self):
        registry = get_registry()
        engine1 = MagicMock()
        engine2 = MagicMock()
        model = object()
        registry.acquire(model, engine1, "engine1")
        with pytest.raises(ModelOwnershipError) as exc_info:
            registry.acquire(model, engine2, "engine2")
        error_msg = str(exc_info.value)
        assert "engine1" in error_msg
