from fusion_mlx.exceptions import (
    EnginePoolError,
    InsufficientMemoryError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
    is_cache_corruption_error,
)


class TestExceptionHierarchy:
     def test_engine_pool_error(self):
        err = EnginePoolError("pool error")
        assert str(err) == "pool error"
        assert isinstance(err, Exception)

     def test_insufficient_memory_inheritance(self):
        err = InsufficientMemoryError("no mem")
        assert isinstance(err, EnginePoolError)

     def test_model_loading_inheritance(self):
        err = ModelLoadingError("load fail")
        assert isinstance(err, EnginePoolError)

     def test_model_not_found_inheritance(self):
        err = ModelNotFoundError("not found")
        assert isinstance(err, EnginePoolError)

     def test_model_too_large_inheritance(self):
        err = ModelTooLargeError("model-x", 8_000_000_000, 4_000_000_000)
        assert isinstance(err, EnginePoolError)
        assert err.model_id == "model-x"
        assert err.model_size == 8_000_000_000
        assert err.ceiling == 4_000_000_000

     def test_is_cache_corruption_error(self):
        assert is_cache_corruption_error(Exception("test")) is False
