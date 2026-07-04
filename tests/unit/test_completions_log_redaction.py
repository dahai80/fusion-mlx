# SPDX-License-Identifier: Apache-2.0
import logging

import pytest

logger = logging.getLogger(__name__)

SKIP_REASON = (
    "fusion-mlx does not have fusion_mlx.routes.completions; "
    "the /v1/completions route with prompt-redaction logging "
    "has not been ported from Rapid-MLX. Re-enable when the "
    "completions route lands in fusion-mlx."
)


@pytest.mark.skip(reason=SKIP_REASON)
def test_info_log_does_not_leak_prompt_body():
    pass


@pytest.mark.skip(reason=SKIP_REASON)
def test_info_log_carries_metadata_only():
    pass


@pytest.mark.skip(reason=SKIP_REASON)
def test_debug_log_carries_redacted_preview():
    pass
