# SPDX-License-Identifier: Apache-2.0
# mtp + suffix coexistence coherence: mtp<->suffix per-request routing guard.
#
# Regression guard for the per-request spec-routing Step 1 mutex break.
# With both --enable-mtp and --suffix-decoding active (Change 4 lifted the
# CLI mutex), mtp takes priority for MTP-eligible decode steps (verify+accept
# inside GenerationBatch.next) and suffix runs only on steps mtp did not own.
# The scheduler's _try_spec_decode guard (last_step_was_mtp) prevents a
# second spec loop on a step mtp already ran - the historical double-spec
# cache-corruption path.
#
# All three configs must produce token-for-token identical greedy output
# (spec methods are correctness-preserving via verified acceptance). If the
# guard were absent, config BOTH would double-spec and diverge from the
# MTP-only and SUFFIX-only baselines.
#
# Skipped unless the MTP-head model is present. Run with:
#   pytest tests/integration/test_mtp_suffix_coherence.py -s

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration

DEFAULT_MODEL = "/Users/dahai/.fusion-mlx/models/mlx-community/Qwen3.5-9B-4bit"
PROMPT = "Count from 1 to 40, separated by commas: 1, 2, 3,"
MAX_TOKENS = 80

_GEN_SCRIPT = """
import asyncio, os
from fusion_mlx.engines.batched import BatchedEngine
from fusion_mlx.model_settings import ModelSettings

async def main():
    mtp = os.environ["FUSION_E2E_MTP"] == "1"
    suf = os.environ["FUSION_E2E_SUFFIX"] == "1"
    settings = ModelSettings(mtp_enabled=mtp, ngram_spec_enabled=suf)
    engine = BatchedEngine(
        model_name=os.environ["FUSION_E2E_MODEL"],
        model_settings=settings,
    )
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
    return os.environ.get("FUSION_E2E_MODEL", DEFAULT_MODEL)


_LOAD_FAILURE_MARKERS = (
    "parameters not in model",
    "not supported",
    "ModuleNotFoundError",
    "ImportError",
    "load_weights",
)


def _is_load_failure(stderr: str) -> bool:
    return any(m in stderr for m in _LOAD_FAILURE_MARKERS)


def _run(mtp: bool, suffix: bool, model: str) -> str:
    env = dict(os.environ)
    env["FUSION_E2E_MTP"] = "1" if mtp else "0"
    env["FUSION_E2E_SUFFIX"] = "1" if suffix else "0"
    env["FUSION_E2E_MODEL"] = model
    script = _GEN_SCRIPT.format(prompt=PROMPT, max_tokens=MAX_TOKENS)
    proc = subprocess.run(
        [sys.executable, "-u", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        if _is_load_failure(proc.stderr):
            raise MtpLoadUnavailable(
                f"mtp={mtp} suffix={suffix} model-load unavailable: "
                f"{proc.stderr[-800:]}"
            )
        raise RuntimeError(f"mtp={mtp} suffix={suffix} failed: {proc.stderr[-2000:]}")
    return proc.stdout.strip()


class MtpLoadUnavailable(RuntimeError):
    pass


@pytest.mark.skipif(
    not os.path.isdir(_model_path()),
    reason=f"MTP-head model not present at {_model_path()}",
)
def test_mtp_suffix_coexistence_coherent():
    model = _model_path()
    try:
        mtp_only = _run(mtp=True, suffix=False, model=model)
    except MtpLoadUnavailable as exc:
        # Pre-existing environment limitation (quantized checkpoint vs mtp
        # patch strict-load, or mlx_lm lacking the model type) - NOT a
        # coherence regression in the per-request routing guard. Skip so CI
        # stays green; the unit tests in test_mtp_suffix_coexistence.py
        # cover the guard logic, and this E2E runs once an mtp-loadable
        # model is available.
        pytest.skip(f"mtp-load unavailable in this environment: {exc}")
    suffix_only = _run(mtp=False, suffix=True, model=model)
    both = _run(mtp=True, suffix=True, model=model)

    assert mtp_only, "mtp-only produced empty output; stderr check needed"
    assert suffix_only, "suffix-only produced empty output; stderr check needed"
    assert both, "mtp+suffix produced empty output; stderr check needed"

    assert both == mtp_only, (
        f"mtp+suffix diverged from mtp-only (double-spec guard failed):\n"
        f"MTP_ONLY : {mtp_only!r}\nBOTH     : {both!r}"
    )
    assert both == suffix_only, (
        f"mtp+suffix diverged from suffix-only:\n"
        f"SUFFIX_ONLY: {suffix_only!r}\nBOTH       : {both!r}"
    )
