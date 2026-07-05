# SPDX-License-Identifier: Apache-2.0
"""N-gram speculative decode coherence on hybrid GDN models.

Regression guard for the v0.4.1 corruption fix. The n-gram spec rejection
path (``ngram_spec._verify_drafts`` + rollback) must produce token-for-token
identical output to pure decode on a hybrid recurrent model
(Qwen3.6-27B-mxfp8: 48 GatedDeltaNet / ArraysCache layers + 16 KVCache
layers), with rejections actually exercising the rollback.

The earlier corruption ("1,2,...,11,21,24,93,5,6,7,8,9..." repetition on
count tasks) came from two rejection-path bugs, both fixed:
  (1) KVCache not trimmed on rejection — ``trim_prompt_cache`` is a no-op
      for hybrid caches (ArraysCache is non-trimmable) and trimmed the
      wrong count; the path now trims each trimmable cache by K directly.
  (2) ``resample_idx`` off-by-one — bonus token read the pred after the
      first REJECTED draft instead of the last ACCEPTED one.

Skipped unless the model is present (set FUSION_NGRAM_E2E_MODEL or rely on
the default path below). Run with: pytest tests/integration/test_ngram_spec_gdn_coherence.py -s
"""

import os
import re
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration

DEFAULT_MODEL = "/Users/dahai/.fusion-mlx/models/Qwen3.6-27B-mxfp8"
PROMPT = "Count from 1 to 50, separated by commas: 1, 2, 3,"
MAX_TOKENS = 90

_GEN_SCRIPT = """
import asyncio, os
from fusion_mlx.engines.batched import BatchedEngine

async def main():
    engine = BatchedEngine(model_name=os.environ["FUSION_NGRAM_E2E_MODEL"])
    try:
        out = await engine.generate({prompt!r}, max_tokens={max_tokens}, temperature=0.0, top_p=1.0)
        print(out.text, flush=True)
    finally:
        try:
            await engine.stop()
        except Exception:
            pass

asyncio.run(main())
"""


def _model_path():
    return os.environ.get("FUSION_NGRAM_E2E_MODEL", DEFAULT_MODEL)


def _run(spec_on: bool, model: str) -> str:
    env = dict(os.environ)
    env["FUSION_NGRAM_SPEC_ENABLED"] = "1" if spec_on else "0"
    env["FUSION_NGRAM_E2E_MODEL"] = model
    script = _GEN_SCRIPT.format(prompt=PROMPT, max_tokens=MAX_TOKENS)
    proc = subprocess.run(
        [sys.executable, "-u", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"spec_on={spec_on} failed: {proc.stderr[-2000:]}"
        )
    return proc.stdout.strip()


@pytest.mark.skipif(
    not os.path.isdir(_model_path()),
    reason=f"model not present at {_model_path()}",
)
def test_ngram_spec_gdn_coherent():
    model = _model_path()
    spec_on = _run(spec_on=True, model=model)
    spec_off = _run(spec_on=False, model=model)
    assert spec_on, f"spec-on produced empty output; stderr check needed"
    assert spec_off, f"spec-off produced empty output; stderr check needed"
    assert spec_on == spec_off, (
        f"n-gram spec diverged from pure decode on GDN model:\n"
        f"SPEC_ON : {spec_on!r}\nSPEC_OFF: {spec_off!r}"
    )
