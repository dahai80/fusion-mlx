import numpy as np
import pytest


@pytest.fixture
def tiny_sr_safetensors(tmp_path):
    # Build a minimal RRDBNet (1 block, scale=2) and dump its params as
    # safetensors with the MLX-NHWC layout load_sr_model expects.
    # IMPORTANT: use tree_flatten (not dict(parameters())) — dict() returns
    # only ~6 top-level keys, but load_sr_model flattens to ~40 dotted keys
    # (body.0.rdb1.conv1.weight ...). Matching the flat layout here makes
    # the fixture keys line up 1:1 with the loader's lookup.
    from mlx.utils import tree_flatten
    from safetensors.numpy import save_file

    from fusion_mlx.image.sr.config import RealESRGANConfig
    from fusion_mlx.image.sr.rrdb import RRDBNet

    cfg = RealESRGANConfig(num_block=1, scale=2)
    net = RRDBNet(cfg)
    flat = {}
    for k, v in tree_flatten(net.parameters()):
        flat[k] = np.array(v)
    path = tmp_path / "tiny.safetensors"
    save_file(flat, str(path))
    return str(path), cfg, flat


def test_load_sr_model_matches_all_keys(tiny_sr_safetensors):
    from fusion_mlx.image.sr.weights import load_sr_model

    path, cfg, fixture_flat = tiny_sr_safetensors
    net = load_sr_model(path, config=cfg)
    assert net is not None

    # Strengthened assertion: a loader that matches 0 keys would pass the
    # weak `net is not None` check. Re-flatten the loaded net's params and
    # verify (1) every fixture key is present in the loaded net, and
    # (2) values match for a sample of keys including the first conv and
    # an RDB conv deep in the body.
    from mlx.utils import tree_flatten

    loaded_flat = {k: np.array(v) for k, v in tree_flatten(net.parameters())}
    assert set(loaded_flat.keys()) == set(fixture_flat.keys()), (
        "loaded keys != fixture keys: "
        f"missing={set(fixture_flat) - set(loaded_flat)} "
        f"extra={set(loaded_flat) - set(fixture_flat)}"
    )

    sample_keys = ["conv_first.weight", "body.0.rdb1.conv1.weight", "conv_last.bias"]
    for k in sample_keys:
        assert np.allclose(loaded_flat[k], fixture_flat[k], atol=1e-6), (
            f"value mismatch at {k}: loaded={loaded_flat[k].flatten()[:5]} "
            f"fixture={fixture_flat[k].flatten()[:5]}"
        )


def test_load_sr_model_strips_params_ema_prefix(tmp_path, tiny_sr_safetensors):
    # Re-save with a 'params_ema.' prefix to mimic the raw .pth convention.
    from safetensors import safe_open
    from safetensors.numpy import save_file

    from fusion_mlx.image.sr.weights import load_sr_model

    base_path, cfg, fixture_flat = tiny_sr_safetensors
    prefixed = {}
    with safe_open(base_path, framework="numpy") as f:
        for k in f.keys():  # noqa: SIM118 - safe_open is not a dict, .keys() required
            prefixed["params_ema." + k] = f.get_tensor(k)
    p = tmp_path / "prefixed.safetensors"
    save_file(prefixed, str(p))
    net = load_sr_model(str(p), config=cfg)
    assert net is not None

    # Same strengthened check: after prefix strip, all keys must match.
    from mlx.utils import tree_flatten

    loaded_flat = {k: np.array(v) for k, v in tree_flatten(net.parameters())}
    assert set(loaded_flat.keys()) == set(fixture_flat.keys()), (
        "prefixed-load keys != fixture keys: "
        f"missing={set(fixture_flat) - set(loaded_flat)} "
        f"extra={set(loaded_flat) - set(fixture_flat)}"
    )
    assert np.allclose(
        loaded_flat["conv_first.weight"], fixture_flat["conv_first.weight"], atol=1e-6
    )


def test_load_sr_model_missing_file_returns_init_net():
    # Missing path -> loader logs a warning and returns a random-init net
    # (does not raise). The net must still be a usable RRDBNet.
    from fusion_mlx.image.sr.config import RealESRGANConfig
    from fusion_mlx.image.sr.rrdb import RRDBNet
    from fusion_mlx.image.sr.weights import load_sr_model

    net = load_sr_model("/nonexistent/path/none.safetensors", config=RealESRGANConfig(num_block=1, scale=2))
    assert isinstance(net, RRDBNet)
