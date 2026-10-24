# SPDX-License-Identifier: Apache-2.0
"""Regression tests for audit 1019 L11: one shared SD image pipeline core.

The SD1.5 / SD2 / SDXL pipelines were ~300-line copies. The shared pieces
now live in fusion_mlx.image.pipeline_base; these tests pin the numerics of
the shared denoise loop against the pre-refactor implementation and check
that the per-model modules keep only their real differences.
"""

import ast
import pathlib

import mlx.core as mx
import numpy as np
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_MODULES = {
    "sd15": _REPO / "fusion_mlx/image/sd15/generate.py",
    "sd2": _REPO / "fusion_mlx/image/sd2/generate.py",
    "sdxl": _REPO / "fusion_mlx/image/sdxl/generate.py",
}


class _FakeScheduler:
    def scale_model_input(self, latent):
        return latent

    def step(self, noise, latent):
        return latent - 0.1 * noise


def _reference_loop(latent, timesteps, guidance, context, context_un, unet):
    # Verbatim copy of the pre-refactor loop (SD1.5/SD2 shape).
    for t in timesteps:
        t_arr = mx.array([float(t)])
        if guidance > 1:
            latent_in = mx.concatenate([latent, latent], axis=0)
            context_in = mx.concatenate([context, context_un], axis=0)
            t_in = mx.concatenate([t_arr, t_arr], axis=0)
            noise = unet(latent_in, t_in, context_in)
            noise_cond, noise_un = mx.split(noise, 2, axis=0)
            noise = noise_un + guidance * (noise_cond - noise_un)
        else:
            noise = unet(latent, t_arr, context)
        latent = _FakeScheduler().step(noise, latent)
        mx.eval(latent)
    return latent


class TestDenoiseLoopEquivalence:
    @staticmethod
    def _unet(latent, t_arr, context):
        # Shape-safe stand-in: depends on both latent and conditioning.
        return latent * 0.5 + context.mean() * 0.01 + t_arr[0] * 0.001

    @pytest.mark.parametrize("guidance", [7.5, 1.0])
    def test_matches_pre_refactor_numerics(self, guidance):
        from fusion_mlx.image.pipeline_base import denoise_loop

        mx.random.seed(7)
        latent = mx.random.normal((1, 4, 8, 8), dtype=mx.float32)
        context = mx.random.normal((1, 77, 8), dtype=mx.float32)
        context_un = mx.random.normal((1, 77, 8), dtype=mx.float32)
        timesteps = [900.0, 700.0, 500.0]

        expected = _reference_loop(
            latent, timesteps, guidance, context, context_un, self._unet
        )
        uncond = {"context": context_un} if guidance > 1 else None
        got = denoise_loop(
            "SD",
            _FakeScheduler(),
            timesteps,
            latent,
            guidance,
            {"context": context},
            uncond,
            lambda lat, t, args: self._unet(lat, t, args["context"]),
        )

        assert np.array_equal(np.array(got), np.array(expected))

    def test_sdxl_style_extra_conditioning_is_merged(self):
        from fusion_mlx.image.pipeline_base import denoise_loop

        seen = {}

        def unet(latent, t_arr, context, pooled, time_ids):
            seen["batched"] = latent.shape[0]
            seen["pooled"] = pooled.shape
            return latent * 0.25 + time_ids.sum() * 0

        mx.random.seed(3)
        latent = mx.random.normal((1, 4, 4, 4), dtype=mx.float32)
        cond = {
            "context": mx.zeros((1, 77, 8)),
            "pooled": mx.zeros((1, 8)),
            "time_ids": mx.zeros((1, 6)),
        }
        uncond = {k: v for k, v in cond.items()}
        denoise_loop(
            "SDXL",
            _FakeScheduler(),
            [500.0],
            latent,
            7.5,
            cond,
            uncond,
            lambda lat, t, args: unet(
                lat, t, args["context"], args["pooled"], args["time_ids"]
            ),
        )
        # CFG batches the conditional and unconditional rows.
        assert seen["batched"] == 2
        assert seen["pooled"] == (2, 8)


class TestScheduleHelpers:
    def test_strength_clamped_and_steps_rounded(self, caplog):
        import logging

        from fusion_mlx.image.pipeline_base import prepare_schedule

        with caplog.at_level(logging.WARNING, logger="fusion_mlx.image.pipeline_base"):
            is_img2img, eff_steps, strength = prepare_schedule(
                "SD2", "p", 30, "init.png", 2.0, 512, 512, 7.5, 0
            )
        assert is_img2img is True
        assert strength == 1.0
        assert eff_steps == 30
        assert "out of (0,1], clamping" in caplog.text

    def test_negative_strength_clamps_to_floor(self):
        from fusion_mlx.image.pipeline_base import prepare_schedule

        _, eff_steps, strength = prepare_schedule(
            "SD15", "p", 20, "init.png", 0.0, 512, 512, 7.5, 0
        )
        assert strength == 0.001
        assert eff_steps == 1

    def test_txt2img_keeps_all_steps(self, caplog):
        import logging

        from fusion_mlx.image.pipeline_base import prepare_schedule

        with caplog.at_level(logging.INFO, logger="fusion_mlx.image.pipeline_base"):
            is_img2img, eff_steps, strength = prepare_schedule(
                "SDXL", "p", 25, None, None, 1024, 1024, 7.5, 42
            )
        assert (is_img2img, eff_steps, strength) == (False, 25, 1.0)
        assert "SDXL txt2img prompt=" in caplog.text

    def test_slice_realigns_sigmas_and_step_index(self):
        from fusion_mlx.image.pipeline_base import slice_timesteps

        class _S:
            sigmas = mx.array([0.0, 1.0, 2.0, 3.0, 4.0])
            _step_index = 3

        scheduler = _S()
        timesteps = [800.0, 600.0, 400.0, 200.0]
        got = slice_timesteps(scheduler, timesteps, 2)
        assert got == [400.0, 200.0]
        # sigmas keep n+1 entries: the tail from the same start index.
        assert np.array_equal(np.array(scheduler.sigmas), np.array([2.0, 3.0, 4.0]))
        assert scheduler._step_index == 0

    def test_slice_noop_when_all_steps_run(self):
        from fusion_mlx.image.pipeline_base import slice_timesteps

        class _S:
            sigmas = mx.array([1.0, 2.0])
            _step_index = 1

        scheduler = _S()
        assert slice_timesteps(scheduler, [1.0, 2.0], 2) == [1.0, 2.0]
        assert scheduler._step_index == 1

    def test_init_noise_latent_uses_scheduler_sigma(self):
        from fusion_mlx.image.pipeline_base import init_noise_latent

        class _Cfg:
            in_channels = 4

        class _S:
            init_noise_sigma = 0.5

        mx.random.seed(1)
        latent = init_noise_latent(_Cfg(), 8, 6, _S())
        assert latent.shape == (1, 4, 8, 6)
        mx.random.seed(1)
        raw = mx.random.normal((1, 4, 8, 6), dtype=mx.float32)
        assert np.array_equal(np.array(latent), np.array(raw * 0.5))


class TestSharedTokenizerDirResolution:
    # The old SDXL copy returned the snapshot ROOT instead of root/<subfolder>,
    # the #482 degenerate-tokenizer bug that SD1.5/SD2 had already fixed.
    @pytest.mark.parametrize("name", ["sd15", "sd2", "sdxl"])
    def test_resolve_dir_appends_subfolder(self, name, monkeypatch, tmp_path):
        import huggingface_hub

        from fusion_mlx.image import sd2, sd15, sdxl

        module = {"sd15": sd15, "sd2": sd2, "sdxl": sdxl}[name].generate
        calls = []
        monkeypatch.setattr(
            huggingface_hub,
            "snapshot_download",
            lambda repo, **kw: calls.append(kw) or str(tmp_path),
        )
        got = module._resolve_dir("org/model", "tokenizer")
        assert got == str(tmp_path / "tokenizer")
        assert calls[0]["allow_patterns"] == ["tokenizer/*"]


class TestDuplicationRemoved:
    @pytest.mark.parametrize("name", sorted(_MODULES))
    def test_no_pipeline_helpers_left_in_model_modules(self, name):
        tree = ast.parse(_MODULES[name].read_text())
        defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        moved = {
            "_local_path",
            "_load_safetensors",
            "_to_pil",
            "_pil_to_nchw",
        }
        assert not (defined & moved)
        assert defined <= {"_resolve", "_resolve_dir"}

    @pytest.mark.parametrize("name", sorted(_MODULES))
    def test_gen_result_is_shared(self, name):
        from fusion_mlx.image import pipeline_base, sd2, sd15, sdxl

        module = {"sd15": sd15, "sd2": sd2, "sdxl": sdxl}[name].generate
        assert module.GenResult is pipeline_base.GenResult
