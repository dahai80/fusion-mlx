# SPDX-License-Identifier: Apache-2.0
# Issue #761: an i2v checkpoint (model_type=i2v, in_dim>vae_z_dim) cannot run
# pure text-to-video — its patch_embedding expects more channels than a noise
# latent provides, so the DiT addmm crashes. generate_video must fail visibly
# with a remediation message instead of the cryptic addmm traceback.
import json

import pytest

from fusion_mlx.video.wan2.generate import generate_video


def _make_i2v_model_dir(tmp_path, *, in_dim=36, vae_z_dim=16):
    # Minimal i2v config.json (matches Wan2.1-14B i2v checkpoint signature).
    # No safetensors -> correct_in_dim probe returns None, config is preserved
    # at model_type=i2v, in_dim>vae_z_dim, so the #761 guard fires before any
    # T5 / model load.
    d = tmp_path / "Wan2.1-14B-i2v"
    d.mkdir()
    config = {
        "model_type": "i2v",
        "in_dim": in_dim,
        "vae_z_dim": vae_z_dim,
        "dim": 5120,
        "patch_size": [1, 2, 2],
        "num_heads": 40,
        "num_layers": 40,
        "text_len": 512,
        "vae_stride": [4, 8, 8],
        "num_train_timesteps": 1000,
        "sample_steps": 1,
        "sample_shift": 5.0,
        "sample_guide_scale": 1.0,
    }
    (d / "config.json").write_text(json.dumps(config))
    return d


class TestI2vNoImageGuard:
    def test_guard_present_in_source(self):
        # Fail-visible guard for issue #761 exists in generate_video.
        import inspect

        src = inspect.getsource(generate_video)
        assert "issue #761" in src
        assert 'model_type == "i2v"' in src

    def test_i2v_no_image_raises_valueerror(self, tmp_path):
        # i2v checkpoint, no image/camera/control/reference -> fail visibly.
        d = _make_i2v_model_dir(tmp_path)
        with pytest.raises(ValueError, match="(?i)i2v model.*text-to-video"):
            generate_video(
                model_dir=d,
                prompt="a red circle bouncing",
                num_frames=9,
                width=256,
                height=256,
                steps=1,
                guide_scale=1.0,
                seed=0,
                scheduler="unipc",
                no_compile=True,
            )

    def test_i2v_with_image_not_guarded(self, tmp_path):
        # image= satisfies the i2v channel-concat path; the guard must NOT
        # fire (model load may then fail for the missing weights, but that is
        # a different error class — not the #761 guard ValueError).
        d = _make_i2v_model_dir(tmp_path)
        raised = None
        try:
            generate_video(
                model_dir=d,
                prompt="test",
                image=str(tmp_path / "img.png"),
                num_frames=9,
                width=256,
                height=256,
                steps=1,
                guide_scale=1.0,
                seed=0,
                scheduler="unipc",
                no_compile=True,
            )
        except ValueError as e:
            raised = e
        except Exception:
            # A non-ValueError (e.g. FileNotFoundError on weights) is fine —
            # the guard did not fire.
            return
        if isinstance(raised, ValueError):
            # If a ValueError surfaces it must NOT be the #761 guard message.
            assert "text-to-video" not in str(raised).lower() or "i2v model" not in str(
                raised
            )

    def test_t2v_model_not_guarded(self, tmp_path):
        # A genuine t2v model (model_type=t2v, in_dim==vae_z_dim) runs without
        # the guard. Missing weights -> non-guard error, which is acceptable.
        d = _make_i2v_model_dir(tmp_path, in_dim=16, vae_z_dim=16)
        # Override model_type to t2v.
        cfg = json.loads((d / "config.json").read_text())
        cfg["model_type"] = "t2v"
        (d / "config.json").write_text(json.dumps(cfg))
        raised = None
        try:
            generate_video(
                model_dir=d,
                prompt="test",
                num_frames=9,
                width=256,
                height=256,
                steps=1,
                guide_scale=1.0,
                seed=0,
                scheduler="unipc",
                no_compile=True,
            )
        except ValueError as e:
            raised = e
        except Exception:
            return
        if isinstance(raised, ValueError):
            assert "i2v model" not in str(raised)
