# SPDX-License-Identifier: Apache-2.0
"""Tests for /v1/requests/{id}/cancel and BatchedEngine.abort_request routing.

Adapted from Rapid-MLX. Tests depending on ``fusion_mlx.engine.batched``,
``fusion_mlx.routes_internal.health``, or ``fusion_mlx.config.get_config`` are
skipped as those modules don't exist in fusion-mlx.
"""

import pytest


class _StubAsyncEngine:

    def __init__(self, returns: bool):
        self._returns = returns
        self.calls: list[str] = []

    async def abort_request(self, request_id: str) -> bool:
        self.calls.append(request_id)
        return self._returns


class _StubSyncMllmScheduler:

    def __init__(self, returns: bool):
        self._returns = returns
        self.calls: list[str] = []

    def abort_request(self, request_id: str) -> bool:
        self.calls.append(request_id)
        return self._returns


@pytest.mark.skip(
    reason="rapid-mlx-only: fusion_mlx.engine.batched.BatchedEngine does not exist"
)
class TestBatchedEngineAbortRouting:
    pass


@pytest.mark.skip(
    reason="rapid-mlx-only: fusion_mlx.engine.base.BaseEngine has no abort_request"
)
class TestBaseEngineDefaultAbort:
    pass


@pytest.mark.skip(
    reason="rapid-mlx-only: fusion_mlx.routes_internal.health does not exist"
)
class TestCancelRequestEndpoint:
    pass
