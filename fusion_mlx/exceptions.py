# SPDX-License-Identifier: Apache-2.0
"""Exceptions for fusion-mlx."""


class EnginePoolError(Exception):
    """Base exception for engine pool errors."""


class InsufficientMemoryError(EnginePoolError):
    """Not enough memory to load the requested model."""


class ModelLoadingError(EnginePoolError):
    """Failed to load a model."""


class ModelNotFoundError(EnginePoolError):
    """Requested model does not exist or is not loaded."""


class ModelTooLargeError(InsufficientMemoryError):
    """Model is too large to fit in available memory."""


def is_cache_corruption_error(err):
    return False
