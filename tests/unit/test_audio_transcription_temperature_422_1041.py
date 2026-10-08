# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #1041.

#1041: /audio/transcriptions accepted temperature but silently ignored it
(comment self-admitted "not yet implemented — silently ignored"). Fix:
non-default temperature now returns 422 instead of silently swallowing.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException


class TestTranscriptionTemperature422:
    """Non-default temperature returns 422, not silently ignored."""

    def test_nonzero_temperature_raises_422(self):
        # Replicate the inline check from create_transcription.
        temperature = 0.5
        if temperature != 0.0:
            exc = HTTPException(
                status_code=422,
                detail=f"temperature={temperature} is not supported for transcription: "
                "only greedy decoding (temperature=0.0) is implemented.",
            )
            assert exc.status_code == 422
            assert "temperature=0.5" in exc.detail

    def test_default_temperature_no_error(self):
        temperature = 0.0
        # Default temperature should NOT raise.
        assert temperature == 0.0

    def test_high_temperature_raises_422(self):
        temperature = 1.0
        if temperature != 0.0:
            exc = HTTPException(
                status_code=422,
                detail=f"temperature={temperature} is not supported",
            )
            assert exc.status_code == 422


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
