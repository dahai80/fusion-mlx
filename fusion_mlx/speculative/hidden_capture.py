# SPDX-License-Identifier: Apache-2.0
"""HiddenStateCapture — layer-replacement middleware for capturing
intermediate hidden states from the target model during forward pass.

Used by Eagle3 and DFly speculative decoders to feed real hidden states
into their drafters instead of zero tensors.

CRITICAL: Uses layer replacement (model.model.layers[idx] = wrapper),
NOT monkey-patching layer.__call__. Python dunder methods are looked up
on the type, not the instance, so monkey-patching __call__ is a silent
no-op.
"""

import logging
import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class _CapturedLayer(nn.Module):
    """Layer wrapper that intercepts forward output and stores it.

    Replaces the original layer in model.model.layers so that every
    forward call passes through this wrapper transparently.
    """

    def __init__(
        self,
        original_layer: nn.Module,
        layer_idx: int,
        capture_store: dict[int, mx.array],
        prefill_store: dict[int, mx.array] | None = None,
    ):
        super().__init__()
        self._original = original_layer
        self._layer_idx = layer_idx
        self._capture_store = capture_store
        self._prefill_store = prefill_store

    def __call__(self, *args, **kwargs):
        out = self._original(*args, **kwargs)
        if isinstance(out, (tuple, list)):
            captured_val = out[0]
        else:
            captured_val = out
        self._capture_store[self._layer_idx] = captured_val
        seq_len = captured_val.shape[1] if captured_val.ndim >= 2 else -1
        ps = getattr(self, "_prefill_store", None)
        if seq_len > 1 and ps is not None:
            ps[self._layer_idx] = captured_val
        return out

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattr__(name)
        return getattr(self._original, name)


class HiddenStateCapture:
    """Middleware that captures hidden states from specified layers
    of the target model during its regular forward pass.

    Usage:
        capture = HiddenStateCapture(model, layer_ids=[1, 20, 39, 58, 77])
        capture.install()
        # ... regular forward pass runs, capture._captured fills ...
        hidden_states = capture.get_captured()
        capture.uninstall()
    """

    def __init__(self, model: Any, layer_ids: list[int] | None = None):
        self._model = model
        self._layer_ids = layer_ids or []
        self._captured: dict[int, mx.array] = {}
        self._prefill_captured: dict[int, mx.array] = {}
        self._original_layers: dict[int, nn.Module] = {}
        self._installed = False
        self._capture_count = 0
        self._install_overhead_ms = 0.0

    @property
    def layer_ids(self) -> list[int]:
        return list(self._layer_ids)

    @property
    def installed(self) -> bool:
        return self._installed

    def _get_layers(self) -> list[nn.Module]:
        """Navigate to model.model.layers regardless of model wrapper."""
        inner = self._model
        if hasattr(inner, "model"):
            inner = inner.model
        if hasattr(inner, "layers"):
            return inner.layers
        return []

    def _get_inner_model(self) -> Any:
        """Return the inner model that holds .layers."""
        inner = self._model
        if hasattr(inner, "model"):
            inner = inner.model
        return inner

    def install(self) -> None:
        if self._installed:
            logger.debug("hidden_capture: already installed, skipping")
            return

        if not self._layer_ids:
            logger.debug("hidden_capture: no layer_ids, nothing to install")
            return

        t0 = time.perf_counter()
        layers = self._get_layers()
        inner = self._get_inner_model()

        installed_count = 0
        for idx in self._layer_ids:
            if idx < len(layers):
                self._original_layers[idx] = layers[idx]
                wrapped = _CapturedLayer(
                    layers[idx],
                    idx,
                    self._captured,
                    self._prefill_captured,
                )
                inner.layers[idx] = wrapped
                installed_count += 1
            else:
                logger.warning(
                    "hidden_capture: layer_idx=%d exceeds model layers=%d, skipping",
                    idx,
                    len(layers),
                )

        dt = (time.perf_counter() - t0) * 1000
        self._install_overhead_ms = dt
        self._installed = True
        logger.info(
            "hidden_capture: installed %d/%d wrappers in %.1fms (layers=%s, ps_id=%s, captured_id=%s)",
            installed_count,
            len(self._layer_ids),
            dt,
            self._layer_ids,
            id(self._prefill_captured),
            id(self._captured),
        )

    def uninstall(self) -> None:
        if not self._installed:
            return

        inner = self._get_inner_model()
        for idx, original in self._original_layers.items():
            inner.layers[idx] = original

        self._original_layers.clear()
        self._captured.clear()
        self._prefill_captured.clear()
        self._installed = False
        self._capture_count = 0
        logger.info("hidden_capture: uninstalled, restored original layers")

    def get_captured(self) -> dict[int, mx.array]:
        return dict(self._captured)

    def get_captured_list(self) -> list[mx.array]:
        result = []
        for idx in self._layer_ids:
            if idx in self._captured:
                result.append(self._captured[idx])
        return result

    def clear_captured(self) -> None:
        self._captured.clear()

    def on_new_request(self) -> None:
        self.clear_captured()
        self._capture_count = 0

    def get_prefill_captured(self) -> dict[int, mx.array]:
        return dict(self._prefill_captured)

    def clear_prefill_captured(self) -> None:
        self._prefill_captured.clear()

    def get_stats(self) -> dict:
        return {
            "installed": self._installed,
            "layer_ids": self.layer_ids,
            "captures": self._capture_count,
            "install_overhead_ms": self._install_overhead_ms,
            "captured_layers": list(self._captured.keys()),
        }
