# SPDX-License-Identifier: Apache-2.0
"""Exceptions for fusion-mlx."""


class FusionMLXError(Exception):
    """Base exception for all fusion-mlx errors."""


class EnginePoolError(FusionMLXError):
    """Base exception for engine pool errors."""


class InsufficientMemoryError(EnginePoolError):
    """Not enough memory to load the requested model."""


class ModelLoadingError(EnginePoolError):
    """Failed to load a model."""


class ModelNotFoundError(EnginePoolError):
    """Requested model does not exist or is not loaded."""


class ModelTooLargeError(InsufficientMemoryError):
    """Model is too large to fit in available memory."""


class SchedulerError(FusionMLXError):
    """Base exception for scheduler errors."""


class RequestTimeoutError(SchedulerError):
    """A request timed out waiting for resources."""


class PreemptionError(SchedulerError):
    """Failed to preempt a request."""


class CacheError(FusionMLXError):
    """Base exception for cache errors."""


class CacheCorruptionError(CacheError):
    """Cache data is corrupted or invalid."""


class CacheWriteError(CacheError):
    """Failed to write to cache storage."""


class CacheReadError(CacheError):
    """Failed to read from cache storage."""


class RouterError(FusionMLXError):
    """Base exception for routing errors."""


class CloudRoutingError(RouterError):
    """Cloud routing failed."""


def is_cache_corruption_error(err):
    """Check if an error indicates cache corruption."""
    return isinstance(err, (
        CacheCorruptionError, ValueError, TypeError, AttributeError
     ))
