import importlib

import pytest

from fusion_mlx.engines import image_gen


def _reload_with_env(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("FUSION_FLUX_QUANT", raising=False)
    else:
        monkeypatch.setenv("FUSION_FLUX_QUANT", value)
    importlib.reload(image_gen)
    return image_gen._flux_quantize_from_env()


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    yield
    monkeypatch.delenv("FUSION_FLUX_QUANT", raising=False)
    importlib.reload(image_gen)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("w8a16", 8),
        ("w8", 8),
        ("int8", 8),
        ("8", 8),
        ("W8A16", 8),
        (" w8a16 ", 8),
        ("w4", 4),
        ("nf4", 4),
        ("int4", 4),
        ("4", 4),
        ("W4", 4),
    ],
)
def test_quant_aliases(monkeypatch, raw, expected):
    assert _reload_with_env(monkeypatch, raw) == expected


@pytest.mark.parametrize("raw", ["", "0", "off", "none", "bf16", "OFF", "None"])
def test_quant_off(monkeypatch, raw):
    assert _reload_with_env(monkeypatch, raw) is None


@pytest.mark.parametrize("raw", ["7", "fp16", "garbage", "w16"])
def test_quant_invalid_warns(monkeypatch, caplog, raw):
    with caplog.at_level("WARNING"):
        result = _reload_with_env(monkeypatch, raw)
    assert result is None
    assert "FUSION_FLUX_QUANT" in caplog.text


def test_quant_unset(monkeypatch):
    assert _reload_with_env(monkeypatch, None) is None


def test_engine_init_picks_env(monkeypatch):
    monkeypatch.setenv("FUSION_FLUX_QUANT", "w8a16")
    importlib.reload(image_gen)
    eng = image_gen.ImageGenEngine(model_name="FLUX.2-klein-base-4B")
    assert eng._quantize == 8
