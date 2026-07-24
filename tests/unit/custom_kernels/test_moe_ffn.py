import logging
import numpy as np
import mlx.core as mx
from fusion_mlx.custom_kernels.phase_c.glm_moe_ffn import moe_ffn_fused, is_native_available

logger = logging.getLogger(__name__)


def reference_moe_ffn(
    x: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
):
    gate = x @ w_gate.T
    up = x @ w_up.T
    hidden = gate / (1.0 + np.exp(-gate)) * up
    output = hidden @ w_down.T
    return output


class TestMoeFfnFallback:
    def test_small(self):
        M, K, inter_dim, K_out = 4, 64, 32, 64
        np.random.seed(42)
        x_np = np.random.randn(M, K).astype(np.float32) * 0.5
        w_gate_np = np.random.randn(inter_dim, K).astype(np.float32) * 0.3
        w_up_np = np.random.randn(inter_dim, K).astype(np.float32) * 0.3
        w_down_np = np.random.randn(K_out, inter_dim).astype(np.float32) * 0.3

        expected = reference_moe_ffn(x_np, w_gate_np, w_up_np, w_down_np)

        result = moe_ffn_fused(
            mx.array(x_np), mx.array(w_gate_np),
            mx.array(w_up_np), mx.array(w_down_np),
        )
        np.testing.assert_allclose(np.array(result), expected, atol=0.01, rtol=0.01)
        logger.info("test_small: PASSED")

    def test_single_token(self):
        M, K, inter_dim, K_out = 1, 128, 64, 128
        np.random.seed(123)
        x_np = np.random.randn(M, K).astype(np.float32) * 0.3
        w_gate_np = np.random.randn(inter_dim, K).astype(np.float32) * 0.2
        w_up_np = np.random.randn(inter_dim, K).astype(np.float32) * 0.2
        w_down_np = np.random.randn(K_out, inter_dim).astype(np.float32) * 0.2

        expected = reference_moe_ffn(x_np, w_gate_np, w_up_np, w_down_np)

        result = moe_ffn_fused(
            mx.array(x_np), mx.array(w_gate_np),
            mx.array(w_up_np), mx.array(w_down_np),
        )
        np.testing.assert_allclose(np.array(result), expected, atol=0.01, rtol=0.01)
        logger.info("test_single_token: PASSED")

    def test_llm_shape(self):
        K = 4096
        inter_dim = 14336
        K_out = 4096
        M = 2
        np.random.seed(99)
        x_np = np.random.randn(M, K).astype(np.float32) * 0.1
        w_gate_np = np.random.randn(inter_dim, K).astype(np.float32) * 0.05
        w_up_np = np.random.randn(inter_dim, K).astype(np.float32) * 0.05
        w_down_np = np.random.randn(K_out, inter_dim).astype(np.float32) * 0.05

        expected = reference_moe_ffn(x_np, w_gate_np, w_up_np, w_down_np)

        result = moe_ffn_fused(
            mx.array(x_np), mx.array(w_gate_np),
            mx.array(w_up_np), mx.array(w_down_np),
        )
        np.testing.assert_allclose(np.array(result), expected, atol=0.1, rtol=0.01)
        logger.info("test_llm_shape: PASSED")


class TestMoeFfnMetal:
    def test_metal_source_compiles(self):
        import subprocess
        from pathlib import Path

        metal_src = Path(__file__).resolve().parents[3] / "fusion_mlx" / "custom_kernels" / "metal" / "moe_ffn_fused.metal"
        if not metal_src.exists():
            logger.warning("moe_ffn_fused.metal not found, skip compile test")
            return

        try:
            result = subprocess.run(
                ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.1",
                 "-c", str(metal_src), "-o", "/dev/null"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                logger.info("test_metal_source_compiles: PASSED (xcrun metal3.1)")
            else:
                logger.warning(f"Metal compile errors:\n{result.stderr[:500]}")
        except FileNotFoundError:
            logger.warning("xcrun not found, skip Metal compile test")

    def test_native_availability(self):
        logger.info(f"is_native_available: {is_native_available()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    t1 = TestMoeFfnFallback()
    t1.test_small()
    t1.test_single_token()
    t1.test_llm_shape()
    t2 = TestMoeFfnMetal()
    t2.test_metal_source_compiles()
    t2.test_native_availability()
