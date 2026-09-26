# SPDX-License-Identifier: Apache-2.0
# #995: ltx2_5 dev pipeline joint A/V generation wiring. Verifies the dev
# branch calls denoise_dev_av + _write_mp4_av when audio=True, and no longer
# raises NotImplementedError. Real e2e gated by FUSION_MLX_REAL_MODEL_TESTS.
from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

import pytest

from fusion_mlx.video.ltx2_5 import generate as gen_mod


def _dev_branch_source() -> str:
    src = inspect.getsource(gen_mod.generate_video)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "var_str"
            and test.ops
            and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "dev"
        ):
            return ast.unparse(ast.Module(body=node.body, type_ignores=[]))
    raise AssertionError("dev branch not found")


def test_dev_branch_no_notimplemented_error():
    # #995: the audio NotImplementedError is gone — dev A/V is wired.
    dev_src = _dev_branch_source()
    tree = ast.parse(dev_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            func = node.exc.func
            name = func.id if isinstance(func, ast.Name) else ""
            if name == "NotImplementedError":
                pytest.fail(
                    "dev branch still raises NotImplementedError — "
                    "A/V path not wired (#995): " + ast.unparse(node)
                )


def test_dev_branch_calls_denoise_dev_av_when_audio():
    dev_src = _dev_branch_source()
    assert (
        "denoise_dev_av(" in dev_src
    ), "dev branch must call denoise_dev_av for joint A/V (#995)"
    # call is gated by `if audio:`
    idx = dev_src.find("denoise_dev_av(")
    before = dev_src[:idx]
    assert "if audio:" in before, "denoise_dev_av call must be under `if audio:`"


def test_dev_branch_calls_write_mp4_av_when_audio():
    dev_src = _dev_branch_source()
    assert (
        "_write_mp4_av(" in dev_src
    ), "dev branch must call _write_mp4_av to mux audio (#995)"


def test_denoise_dev_av_imported():
    # import surface: generate module must import denoise_dev_av.
    assert hasattr(
        gen_mod, "denoise_dev_av"
    ), "generate module must import denoise_dev_av from ..ltx2.denoise (#995)"


_LTX25_Q8_REPO = next(
    (Path.home() / ".fusion-mlx" / "models" / "models--dgrauet--ltx-2.5-mlx-q8").glob(
        "snapshots/*"
    ),
    None,
)
# q8 repo has all components + transformer-dev.ref -> 40GB bf16 dev transformer.
_REAL = (
    os.environ.get("FUSION_MLX_REAL_MODEL_TESTS") == "1"
    and _LTX25_Q8_REPO is not None
    and (_LTX25_Q8_REPO / "transformer-dev.ref").exists()
)
real_model = pytest.mark.skipif(
    not _REAL, reason="needs ltx2_5 q8 repo + FUSION_MLX_REAL_MODEL_TESTS=1"
)


@real_model
def test_dev_av_e2e_mp4_has_audio_stream(tmp_path):
    # Real e2e: dev + audio at tiny res → mp4 must contain an aac stream.
    # Heavy (22B load); gated. Verifies #995 end-to-end on production path.

    out = str(tmp_path / "dev_av.mp4")
    mp4 = gen_mod.generate_video(
        model_repo=str(_LTX25_Q8_REPO),
        prompt="a cat meowing",
        num_frames=9,
        height=256,
        width=256,
        fps=24,
        seed=0,
        audio=True,
        variant="dev",
        num_inference_steps=4,
        output_path=out,
    )
    assert os.path.exists(out)
    assert len(mp4) > 1000
    # ffmpeg probe: must report an audio stream.
    import subprocess

    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            out,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (
        "audio" in r.stdout
    ), f"dev AV mp4 has no audio stream (#995): stdout={r.stdout!r} stderr={r.stderr!r}"
