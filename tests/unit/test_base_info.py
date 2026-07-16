from __future__ import annotations

from fusion_mlx._version import __version__
from fusion_mlx.server import _runtime_base_info


class _FakePool:
    def __init__(self, ids):
        self._ids = ids

    def get_loaded_model_ids(self):
        return list(self._ids)


def test_base_info_has_required_fields():
    info = _runtime_base_info(None)
    for key in (
        "version",
        "metal_available",
        "metal_family",
        "kv_cache_supported",
        "quantization_formats",
        "max_context_length",
        "gpu_info",
        "compatible_models",
    ):
        assert key in info, f"missing field: {key}"


def test_base_info_version_matches_package():
    assert _runtime_base_info(None)["version"] == __version__


def test_base_info_quant_formats_include_core_set():
    fmts = _runtime_base_info(None)["quantization_formats"]
    for expected in ("mxfp4", "mxfp8", "quant2"):
        assert expected in fmts


def test_base_info_metal_available_is_bool():
    assert isinstance(_runtime_base_info(None)["metal_available"], bool)


def test_base_info_gpu_info_shape():
    gpu = _runtime_base_info(None)["gpu_info"]
    assert set(gpu.keys()) == {"chip_name", "gpu_cores", "memory_gb"}
    # gpu_cores is not reported by mlx.device_info -> stays None (not fabricated)
    assert gpu["gpu_cores"] is None


def test_base_info_metal_family_not_fabricated():
    # metal_family is not exposed by mlx -> None rather than a guessed value
    assert _runtime_base_info(None)["metal_family"] is None


def test_base_info_compatible_models_empty_when_no_pool():
    assert _runtime_base_info(None)["compatible_models"] == []


def test_base_info_compatible_models_from_pool():
    pool = _FakePool(["qwen3.5-9b", "bge-m3"])
    assert _runtime_base_info(pool)["compatible_models"] == ["qwen3.5-9b", "bge-m3"]


def test_base_info_pool_error_does_not_raise():
    class _BadPool:
        def get_loaded_model_ids(self):
            raise RuntimeError("boom")

    info = _runtime_base_info(_BadPool())
    assert info["compatible_models"] == []


def test_base_info_metal_probe_failure_safe():
    # Force the mlx import path to fail -> metal_available False, still valid
    import fusion_mlx.server as srv

    orig = srv.__dict__.get("_runtime_base_info")
    try:
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "mlx.core":
                raise ImportError("forced")
            return real_import(name, *a, **k)

        builtins.__import__ = fake_import
        info = srv._runtime_base_info(None)
        assert info["metal_available"] is False
        assert info["gpu_info"]["chip_name"] is None
    finally:
        builtins.__import__ = real_import
        assert orig is srv._runtime_base_info
