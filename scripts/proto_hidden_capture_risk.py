#!/usr/bin/env python3
"""Prototype: validate HiddenStateCapture risks (R2/R3/R5/R6/R8).

R2: Shape/Dtype mismatch after capture
R3: DFlash drafter numerical drift
R5: Capture performance overhead
R6: Cache corruption from patched forwards
R8: Memory leak from lazy eval graph retention

Usage:
    python scripts/proto_hidden_capture_risk.py --target <model_path>
"""
import argparse
import gc
import logging
import time
import traceback

import mlx.core as mx
import mlx.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("proto_risk")


class _CapturedLayer(nn.Module):
    """Wrapper that replaces model.layers[idx] to capture hidden output."""
    def __init__(self, original_layer, layer_idx, capture_store):
        super().__init__()
        self._original = original_layer
        self._layer_idx = layer_idx
        self._capture_store = capture_store

    def __call__(self, *args, **kwargs):
        out = self._original(*args, **kwargs)
        if isinstance(out, (tuple, list)):
            self._capture_store[self._layer_idx] = out[0]
        else:
            self._capture_store[self._layer_idx] = out
        return out

    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattr__(name)
        return getattr(self._original, name)


class HiddenStateCapture:
    def __init__(self, model, layer_ids):
        self.model = model
        self.layer_ids = set(layer_ids)
        self._captured = {}
        self._original_layers = {}
        self._installed = False

    def install(self):
        layers = self._get_layers()
        for idx in self.layer_ids:
            if idx < len(layers):
                self._original_layers[idx] = layers[idx]
                layers[idx] = _CapturedLayer(layers[idx], idx, self._captured)
        self._installed = True
        logger.info(
            "HiddenStateCapture installed (layer replace): %d layers: %s",
            len(self._original_layers),
            sorted(self._original_layers.keys()),
        )

    def uninstall(self):
        layers = self._get_layers()
        for idx, original in self._original_layers.items():
            if idx < len(layers):
                layers[idx] = original
        self._captured.clear()
        self._original_layers.clear()
        self._installed = False
        logger.info("HiddenStateCapture uninstalled (layer restore)")

    def _get_layers(self):
        if hasattr(self.model, 'layers'):
            return self.model.layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        return []

    @property
    def last_hidden(self):
        if not self._captured:
            return None
        return self._captured[max(self._captured.keys())]

    def get_hiddens(self, layer_ids, dtype=None):
        result = []
        for i in layer_ids:
            h = self._captured.get(i)
            if h is not None:
                if dtype is not None and h.dtype != dtype:
                    h = h.astype(dtype)
                result.append(h)
        return result

    def clear(self):
        for idx in list(self._captured.keys()):
            mx.eval(self._captured[idx])
        self._captured.clear()


def test_r2_shape_dtype(model):
    """R2: Verify captured hidden states have correct shape/dtype."""
    logger.info("=" * 60)
    logger.info("R2: Shape/Dtype Mismatch Test")
    logger.info("=" * 60)

    layers = model.model.layers if hasattr(model, 'model') else model.layers
    n_layers = len(layers)
    hidden_size = 0
    if hasattr(model, 'model') and hasattr(model.model, 'args'):
        hidden_size = getattr(model.model.args, 'hidden_size', 0)

    capture_ids = [n_layers - 1]
    if n_layers > 4:
        capture_ids.append(n_layers // 2)
    if n_layers > 8:
        capture_ids.append(n_layers // 4)

    capture = HiddenStateCapture(model, layer_ids=capture_ids)
    capture.install()

    try:
        dummy = mx.array([[1]], mx.uint32)
        out = model(dummy, cache=None)
        mx.eval(out)

        passed = True
        for idx in sorted(capture._captured.keys()):
            h = capture._captured[idx]
            shape_str = str(h.shape)
            dtype_str = str(h.dtype)
            ndim_ok = h.ndim == 3
            hidden_ok = hidden_size == 0 or h.shape[-1] == hidden_size
            status = "OK" if (ndim_ok and hidden_ok) else "FAIL"
            if status == "FAIL":
                passed = False
            logger.info(
                "  layer %2d: shape=%-20s dtype=%-10s ndim=%d hidden_match=%s [%s]",
                idx, shape_str, dtype_str, h.ndim, hidden_ok, status,
            )

        if capture._captured:
            sample = list(capture._captured.values())[0]
            if sample.dtype != mx.float32:
                converted = sample.astype(mx.float32)
                logger.info(
                    "  dtype conversion: %s -> %s OK",
                    sample.dtype, converted.dtype,
                )

        logger.info("R2 RESULT: %s", "PASS" if passed else "FAIL")
        return passed

    finally:
        capture.uninstall()


def test_r5_overhead(model, n_steps=30, warmup=5):
    """R5: Measure capture overhead vs baseline."""
    logger.info("=" * 60)
    logger.info("R5: Performance Overhead Test")
    logger.info("=" * 60)

    layers = model.model.layers if hasattr(model, 'model') else model.layers
    n_layers = len(layers)
    capture_ids = [n_layers - 1]

    for _ in range(warmup):
        out = model(mx.array([[1]], mx.uint32), cache=None)
        mx.eval(out)

    times_baseline = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        out = model(mx.array([[1]], mx.uint32), cache=None)
        mx.eval(out)
        times_baseline.append(time.perf_counter() - t0)

    capture = HiddenStateCapture(model, layer_ids=capture_ids)
    capture.install()

    for _ in range(warmup):
        out = model(mx.array([[1]], mx.uint32), cache=None)
        mx.eval(out)
        capture.clear()

    times_capture = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        out = model(mx.array([[1]], mx.uint32), cache=None)
        mx.eval(out)
        capture.clear()
        times_capture.append(time.perf_counter() - t0)

    capture.uninstall()

    import numpy as np
    b_mean = np.mean(times_baseline) * 1000
    b_std = np.std(times_baseline) * 1000
    c_mean = np.mean(times_capture) * 1000
    c_std = np.std(times_capture) * 1000
    overhead_pct = (c_mean - b_mean) / b_mean * 100

    logger.info("  baseline: %.2f +/- %.2f ms/step", b_mean, b_std)
    logger.info("  capture:  %.2f +/- %.2f ms/step", c_mean, c_std)
    logger.info("  overhead: %.1f%%", overhead_pct)

    passed = overhead_pct < 5.0
    logger.info("R5 RESULT: %s (overhead=%.1f%%, threshold=5%%)", "PASS" if passed else "FAIL", overhead_pct)
    return passed


def test_r6_cache_integrity(model):
    """R6: Verify patched forwards don't corrupt KVCache."""
    logger.info("=" * 60)
    logger.info("R6: Cache Integrity Test")
    logger.info("=" * 60)

    layers = model.model.layers if hasattr(model, 'model') else model.layers
    n_layers = len(layers)

    from mlx_lm.models.cache import KVCache
    prompt_cache = [KVCache() for _ in range(n_layers)]

    tokens = mx.array([[1, 2, 3, 4, 5]], mx.uint32)
    out_baseline = model(tokens, cache=prompt_cache)
    mx.eval(out_baseline)

    baseline_offsets = [c.offset for c in prompt_cache]
    logger.info("  baseline offsets after prefill: %s", baseline_offsets)

    tok1 = mx.array([[6]], mx.uint32)
    out1 = model(tok1, cache=prompt_cache)
    mx.eval(out1)
    offsets_no_capture = [c.offset for c in prompt_cache]
    logger.info("  offsets after step (no capture): %s", offsets_no_capture)

    capture = HiddenStateCapture(model, layer_ids=[n_layers - 1])
    capture.install()

    tok2 = mx.array([[7]], mx.uint32)
    out2 = model(tok2, cache=prompt_cache)
    mx.eval(out2)
    offsets_with_capture = [c.offset for c in prompt_cache]
    logger.info("  offsets after step (with capture): %s", offsets_with_capture)

    captured = capture.last_hidden
    capture.uninstall()

    passed = True
    for i in range(n_layers):
        expected_no_capture = baseline_offsets[i] + 1
        expected_with_capture = expected_no_capture + 1
        if offsets_no_capture[i] != expected_no_capture:
            logger.error(
                "  layer %d: no-capture offset %d != expected %d",
                i, offsets_no_capture[i], expected_no_capture,
            )
            passed = False
        if offsets_with_capture[i] != expected_with_capture:
            logger.error(
                "  layer %d: with-capture offset %d != expected %d",
                i, offsets_with_capture[i], expected_with_capture,
            )
            passed = False

    if captured is not None:
        logger.info("  captured hidden: shape=%s dtype=%s", captured.shape, captured.dtype)
    else:
        logger.error("  no captured hidden state!")
        passed = False

    trim_result = prompt_cache[0].trim(1)
    logger.info("  cache[0].trim(1) = %d, offset now = %d", trim_result, prompt_cache[0].offset)
    if trim_result != 1:
        logger.error("  cache trim returned unexpected value: %d", trim_result)
        passed = False

    logger.info("R6 RESULT: %s", "PASS" if passed else "FAIL")
    return passed


def test_r8_memory_leak(model, n_steps=50):
    """R8: Verify clear() releases graph references."""
    logger.info("=" * 60)
    logger.info("R8: Memory Leak Test")
    logger.info("=" * 60)

    layers = model.model.layers if hasattr(model, 'model') else model.layers
    n_layers = len(layers)
    capture = HiddenStateCapture(model, layer_ids=[n_layers - 1])
    capture.install()

    try:
        for _ in range(3):
            out = model(mx.array([[1]], mx.uint32), cache=None)
            mx.eval(out)
            capture.clear()

        mx.synchronize()
        gc.collect()
        baseline_mem = mx.get_active_memory() / 1024 / 1024
        logger.info("  baseline active memory: %.1f MB", baseline_mem)

        for step in range(n_steps):
            out = model(mx.array([[1]], mx.uint32), cache=None)
            mx.eval(out)
            capture.clear()
            if step % 10 == 9:
                mx.synchronize()
                gc.collect()
                mem = mx.get_active_memory() / 1024 / 1024
                logger.info("  step %3d: active memory: %.1f MB", step + 1, mem)

        mx.synchronize()
        gc.collect()
        final_mem = mx.get_active_memory() / 1024 / 1024
        growth = final_mem - baseline_mem
        growth_pct = (growth / baseline_mem * 100) if baseline_mem > 0 else 0

        logger.info("  final active memory: %.1f MB (growth: %.1f MB, %.1f%%)",
                     final_mem, growth, growth_pct)

        for _ in range(10):
            out = model(mx.array([[1]], mx.uint32), cache=None)
            mx.eval(out)

        mx.synchronize()
        gc.collect()
        leaky_mem = mx.get_active_memory() / 1024 / 1024
        logger.info("  without clear(): %.1f MB (delta: +%.1f MB)",
                     leaky_mem, leaky_mem - final_mem)

        passed = growth_pct < 5.0
        logger.info("R8 RESULT: %s (growth=%.1f%%, threshold=5%%)", "PASS" if passed else "FAIL", growth_pct)
        return passed

    finally:
        capture.clear()
        capture.uninstall()


def test_r3_numerical_drift(model):
    """R3: Test captured hidden state numerical quality for drafter input."""
    logger.info("=" * 60)
    logger.info("R3: Numerical Drift Test (captured hidden quality)")
    logger.info("=" * 60)

    layers = model.model.layers if hasattr(model, 'model') else model.layers
    n_layers = len(layers)

    step = max(1, n_layers // 6)
    capture_ids = sorted(set([1] + [min(i, n_layers - 1) for i in range(step, n_layers, step)][:4]))
    logger.info("  capturing layers: %s", capture_ids)

    capture = HiddenStateCapture(model, layer_ids=capture_ids)
    capture.install()

    try:
        out = model(mx.array([[1]], mx.uint32), cache=None)
        mx.eval(out)

        passed = True
        for idx in sorted(capture._captured.keys()):
            h = capture._captured[idx]
            mx.eval(h)
            h_mean = float(mx.mean(h).item())
            h_std = float(mx.std(h).item())
            h_norm = float(mx.sum(h * h).sqrt().item())
            has_nan = bool(mx.any(mx.isnan(h)).item())
            has_inf = bool(mx.any(mx.isinf(h)).item())

            status = "OK"
            if has_nan:
                status = "FAIL (NaN)"
                passed = False
            elif has_inf:
                status = "FAIL (Inf)"
                passed = False
            elif h_std < 0.001:
                status = "WARN (collapsed)"
                passed = False

            logger.info(
                "  layer %2d: mean=%8.4f std=%8.4f norm=%8.2f dtype=%s [%s]",
                idx, h_mean, h_std, h_norm, h.dtype, status,
            )

        if len(capture._captured) >= 2:
            norms = {}
            for idx, h in capture._captured.items():
                mx.eval(h)
                norms[idx] = float(mx.sum(h * h).sqrt().item())
            norm_vals = list(norms.values())
            max_norm = max(norm_vals)
            min_norm = min(norm_vals)
            ratio = max_norm / min_norm if min_norm > 0 else float('inf')
            logger.info("  cross-layer norm ratio: %.2f (max/min)", ratio)
            # Note: ratio up to 200x is NORMAL for transformers with residual
            # connections in bf16. Drafters (Eagle3/DFly) have built-in
            # RMSNorm that handles this. Only flag as FAIL if NaN/Inf.
            if ratio > 10000:
                logger.warning("  extreme norm ratio (>10000x) suggests real instability")
                passed = False

        logger.info("R3 RESULT: %s", "PASS" if passed else "FAIL")
        return passed

    finally:
        capture.uninstall()


def load_model(model_path):
    logger.info("Loading model from %s ...", model_path)
    from mlx_lm import load
    model, tokenizer = load(model_path)
    logger.info("Model loaded: %s", type(model).__name__)
    if hasattr(model, 'model') and hasattr(model.model, 'args'):
        args = model.model.args
        logger.info(
            "  hidden_size=%d n_layers=%d model_type=%s",
            getattr(args, 'hidden_size', '?'),
            getattr(args, 'num_hidden_layers', '?'),
            getattr(args, 'model_type', '?'),
        )
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Prototype risk validation")
    parser.add_argument("--target", required=True, help="Path to target model")
    parser.add_argument("--skip-r3", action="store_true", help="Skip R3 test")
    parser.add_argument("--skip-r5", action="store_true", help="Skip R5 (slow)")
    args = parser.parse_args()

    model, tokenizer = load_model(args.target)

    results = {}

    tests = [
        ("R2", lambda: test_r2_shape_dtype(model)),
        ("R3", lambda: test_r3_numerical_drift(model)),
        ("R5", lambda: test_r5_overhead(model)),
        ("R6", lambda: test_r6_cache_integrity(model)),
        ("R8", lambda: test_r8_memory_leak(model)),
    ]

    for name, test_fn in tests:
        if name == "R3" and args.skip_r3:
            logger.info("Skipping R3 (requested)")
            results[name] = "SKIP"
            continue
        if name == "R5" and args.skip_r5:
            logger.info("Skipping R5 (requested)")
            results[name] = "SKIP"
            continue
        try:
            results[name] = test_fn()
        except Exception:
            logger.error("%s FAILED with exception:", name)
            traceback.print_exc()
            results[name] = False

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for name, passed in results.items():
        if passed == "SKIP":
            status = "SKIP"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"
        logger.info("  %s: %s", name, status)

    n_pass = sum(1 for v in results.values() if v is True)
    n_fail = sum(1 for v in results.values() if v is False)
    n_skip = sum(1 for v in results.values() if v == "SKIP")
    logger.info("  Total: %d pass, %d fail, %d skip", n_pass, n_fail, n_skip)


if __name__ == "__main__":
    main()
