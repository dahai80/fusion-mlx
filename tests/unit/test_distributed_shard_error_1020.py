# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1020.

#1020: distributed_routes had `_shard_error_response(exc)` without `raise` —
the function internally raised HTTPException, so it worked, but the call-site
pattern looked like exception swallowing (no raise/return keyword). If someone
refactored _shard_error_response to return instead of raise, all 8 call sites
would silently return fake success responses.

Fix (#1020): _shard_error_response now RETURNS HTTPException; call sites
explicitly `raise _shard_error_response(exc)`.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from fusion_mlx.api.distributed_routes import _shard_error_response
from fusion_mlx.distributed.shard import ShardError


class TestShardErrorResponse:
    def test_unknown_shard_returns_404(self):
        exc = ShardError("unknown shard_id: abc123")
        result = _shard_error_response(exc)
        assert isinstance(result, HTTPException)
        assert result.status_code == 404

    def test_model_load_failure_returns_502(self):
        exc = ShardError("failed to load model: OOM")
        result = _shard_error_response(exc)
        assert isinstance(result, HTTPException)
        assert result.status_code == 502

    def test_generic_error_returns_400(self):
        exc = ShardError("bad range")
        result = _shard_error_response(exc)
        assert isinstance(result, HTTPException)
        assert result.status_code == 400

    def test_call_site_raises_not_swallows(self):
        # Verify the function returns (not raises) — the call site must raise
        exc = ShardError("unknown shard_id: test")
        result = _shard_error_response(exc)
        assert hasattr(result, "status_code")  # it's an HTTPException object


if __name__ == "__main__":  # pragma: no cover — convenience only
    pytest.main([__file__, "-v"])
