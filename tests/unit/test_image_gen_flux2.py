from fusion_mlx.engines.image_gen import _infer_flux2_config


def test_flux2_config_default_9b():
    assert _infer_flux2_config("flux-2") == "flux2_klein_9b"
    assert _infer_flux2_config("black-forest-labs/FLUX.2-klein-9B") == "flux2_klein_9b"
    assert _infer_flux2_config("flux2-klein-9b") == "flux2_klein_9b"


def test_flux2_config_4b():
    assert _infer_flux2_config("FLUX.2-klein-4B") == "flux2_klein_4b"
    assert _infer_flux2_config("flux2-klein-4b") == "flux2_klein_4b"


def test_flux2_config_9b_kv():
    assert _infer_flux2_config("flux2-klein-9b-kv") == "flux2_klein_9b_kv"
    assert _infer_flux2_config("FLUX.2-klein-9B-kv") == "flux2_klein_9b_kv"


def test_flux2_config_empty_or_none_defaults_to_9b():
    assert _infer_flux2_config("") == "flux2_klein_9b"
    assert _infer_flux2_config(None) == "flux2_klein_9b"


def test_flux2_config_base_4b():
    assert _infer_flux2_config("FLUX.2-klein-base-4B") == "flux2_klein_base_4b"
    assert _infer_flux2_config("flux2-klein-base-4b") == "flux2_klein_base_4b"


def test_flux2_config_base_9b():
    assert _infer_flux2_config("FLUX.2-klein-base-9B") == "flux2_klein_base_9b"
    assert _infer_flux2_config("flux2-klein-base-9b") == "flux2_klein_base_9b"
