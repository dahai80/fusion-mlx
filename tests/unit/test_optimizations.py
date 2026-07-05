from fusion_mlx.optimizations import (
    HardwareInfo,
    detect_hardware,
    get_optimization_status,
    get_system_memory_gb,
)


def test_detect_hardware_returns_hardware_info():
    hw = detect_hardware()
    assert isinstance(hw, HardwareInfo)
    assert hw.chip_name
    assert hw.total_memory_gb > 0


def test_get_system_memory_gb():
    mem = get_system_memory_gb()
    assert mem > 0


def test_get_optimization_status():
    status = get_optimization_status()
    assert "hardware" in status
    assert "mlx_lm_features" in status
    assert "chip" in status["hardware"]
    assert "total_memory_gb" in status["hardware"]


def test_optimization_status_flash_attention():
    status = get_optimization_status()
    fa = status["mlx_lm_features"]["flash_attention"]
    assert fa in ("built-in", "not available")
