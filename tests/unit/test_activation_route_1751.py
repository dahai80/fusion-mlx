"""Unit tests for admin activation readout + guarded delta apply (#1751)."""

import base64
import inspect

import numpy as np
import pytest

from fusion_mlx.admin import activation_route as ar

# CI stubs mlx.core as a MagicMock; the delta shape-validation tests need
# real array ops. KVCache being a real class is the reliable signal
# (same pattern as test_spec_decode_resample_fix).
try:
    from mlx_lm.models.cache import KVCache as _KVC

    _HAS_REAL_MLX = inspect.isclass(_KVC)
except Exception:  # noqa: BLE001
    _HAS_REAL_MLX = False

pytestmark = pytest.mark.skipif(
    not _HAS_REAL_MLX,
    reason="needs real mlx array ops (stubbed as MagicMock on CI)",
)


@pytest.fixture(autouse=True)
def _clean_registry():
    ar._DELTA_REGISTRY.clear()
    yield
    ar._DELTA_REGISTRY.clear()


class _FakeTokenizer:
    def apply_chat_template(self, conv, tokenize=True, add_generation_prompt=False):
        assert tokenize is True
        return [1, 2, 3, 4]

    def encode(self, text, add_special_tokens=True):
        return [5, 6, 7]

    def decode(self, ids):
        return "tok"


class _FakeInner:
    def __init__(self, num_layers):
        self.layers = [object() for _ in range(num_layers)]
        self.embed_tokens = lambda ids: ids


class _FakeModel:
    def __init__(self, num_layers=8):
        self.model = _FakeInner(num_layers)
        tree = {}
        for i in range(num_layers):
            tree[f"model.layers.{i}.mlp.down_proj.weight"] = f"W{i}"
        self._tree = tree

    def parameters(self):
        return self._tree


class BatchedEngine:  # name matches the production guard in _extract_lm
    def __init__(self, num_layers=8):
        self._model = _FakeModel(num_layers)
        self.tokenizer = _FakeTokenizer()

    def load_weights(self, pairs):
        self.loaded = list(pairs)


class _FakePool:
    def __init__(self, engine):
        self.engine = engine
        self.leased = 0
        self.released = 0

    async def get_engine(self, model_id, _lease=False, adapter_path=None):
        self.leased += 1
        return self.engine

    async def release_engine(self, model_id, adapter_path=None):
        self.released += 1


def _wire(pool):
    ar.set_activation_context(pool)
    return pool


# ─── layer index resolution ───────────────────────────────────────────────────


def test_resolve_layer_negative_index():
    assert ar._resolve_layer_index(-4, 32) == 28
    assert ar._resolve_layer_index(0, 32) == 0
    assert ar._resolve_layer_index(31, 32) == 31


def test_resolve_layer_out_of_range_rejected():
    import fastapi

    with pytest.raises(fastapi.HTTPException) as ei:
        ar._resolve_layer_index(32, 32)
    assert ei.value.status_code == 400
    with pytest.raises(fastapi.HTTPException):
        ar._resolve_layer_index(-33, 32)


# ─── position picking ─────────────────────────────────────────────────────────


def _req(**kw):
    return ar.ActivationRequest(model="m", **kw)


def test_positions_all_and_last():
    assert ar._pick_positions(_req(positions="all"), 5) == [0, 1, 2, 3, 4]
    assert ar._pick_positions(_req(positions="last"), 5) == [4]
    assert ar._pick_positions(_req(positions=[1, -1]), 5) == [1, 4]


def test_positions_out_of_range_rejected():
    import fastapi

    with pytest.raises(fastapi.HTTPException):
        ar._pick_positions(_req(positions=[5]), 5)


def test_positions_invalid_spec_rejected():
    import fastapi

    with pytest.raises(fastapi.HTTPException):
        ar._pick_positions(_req(positions="middle"), 5)


# ─── b64 encoding ─────────────────────────────────────────────────────────────


def test_encode_f32_b64_roundtrip():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    raw = base64.b64decode(ar._encode_f32_b64(arr))
    out = np.frombuffer(raw, dtype="<f4").reshape(2, 2)
    assert np.allclose(out, arr)


# ─── param path eligibility (#1751 guard) ────────────────────────────────────


def test_param_path_non_mlp_rejected():
    import fastapi

    model = _FakeModel()
    for path in (
        "model.layers.7.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
        "layers.7.mlp.down_proj.weight",
    ):
        with pytest.raises(fastapi.HTTPException) as ei:
            ar._validate_param_path(path, model, late_margin=4)
        assert ei.value.status_code == 400


def test_param_path_early_layer_rejected():
    import fastapi

    model = _FakeModel(num_layers=8)
    with pytest.raises(fastapi.HTTPException) as ei:
        ar._validate_param_path(
            "model.layers.3.mlp.down_proj.weight", model, late_margin=4
        )
    assert "not late" in ei.value.detail


def test_param_path_late_layer_ok():
    model = _FakeModel(num_layers=8)
    target = ar._validate_param_path(
        "model.layers.5.mlp.down_proj.weight", model, late_margin=4
    )
    assert target == "W5"


def test_param_path_unknown_rejected():
    import fastapi

    model = _FakeModel(num_layers=8)
    with pytest.raises(fastapi.HTTPException) as ei:
        ar._validate_param_path(
            "model.layers.7.mlp.no_such_param.weight", model, late_margin=4
        )
    assert ei.value.status_code == 400


# ─── engine type guard ────────────────────────────────────────────────────────


def test_extract_lm_refuses_non_lm_engine():
    import fastapi

    class VLMBatchedEngine:
        pass

    with pytest.raises(fastapi.HTTPException) as ei:
        ar._extract_lm(VLMBatchedEngine())
    assert ei.value.status_code == 400


# ─── readout handler flow (fake forward) ─────────────────────────────────────


def test_readout_activations_happy_path(monkeypatch):
    import asyncio

    pool = _wire(_FakePool(BatchedEngine(num_layers=8)))
    hidden = np.ones((4, 16), dtype=np.float32)
    monkeypatch.setattr(ar, "_forward_hidden_states", lambda m, t, li: hidden)
    monkeypatch.setattr(ar, "_executor_for", lambda e: _FakeExecutor())

    req = _req(
        messages=[[{"role": "user", "content": "hi"}]],
        layer=-2,
        positions="all",
    )
    resp = asyncio.run(ar.readout_activations(req, is_admin=True))
    assert resp["layer_index"] == 6
    assert resp["num_layers"] == 8
    assert resp["encoding"] == "base64_f32_le"
    seq = resp["sequences"][0]
    assert seq["shape"] == [4, 16]
    assert seq["token_ids"] == [1, 2, 3, 4]
    assert pool.released == 1


def test_readout_releases_lease_on_forward_error(monkeypatch):
    import asyncio

    pool = _wire(_FakePool(BatchedEngine(num_layers=8)))

    def boom(m, t, li):
        raise RuntimeError("forward failed")

    monkeypatch.setattr(ar, "_forward_hidden_states", boom)
    monkeypatch.setattr(ar, "_executor_for", lambda e: _FakeExecutor())
    req = _req(messages=[[{"role": "user", "content": "hi"}]], layer=-1)
    with pytest.raises(RuntimeError):
        asyncio.run(ar.readout_activations(req, is_admin=True))
    assert pool.released == 1


def test_readout_requires_wired_pool():
    import asyncio

    import fastapi

    old = ar._pool
    ar._pool = None
    try:
        with pytest.raises(fastapi.HTTPException) as ei:
            asyncio.run(ar.readout_activations(_req(input=["x"]), is_admin=True))
        assert ei.value.status_code == 503
    finally:
        ar._pool = old


# ─── apply-delta handler flow (fake apply) ───────────────────────────────────


class _FakeArr:
    def __init__(self, shape):
        self.shape = shape


def test_apply_delta_registers_and_reports(monkeypatch):
    import asyncio

    engine = BatchedEngine(num_layers=8)
    pool = _wire(_FakePool(engine))

    def fake_apply(eng, request):
        assert eng is engine
        return {
            "old_snapshot": _FakeArr((16, 16)),
            "new_value": None,
            "delta": None,
        }

    monkeypatch.setattr(ar, "_apply_on_engine", fake_apply)
    monkeypatch.setattr(ar, "_executor_for", lambda e: _FakeExecutor())
    req = ar.ApplyDeltaRequest(
        model="m",
        param_path="model.layers.7.mlp.down_proj.weight",
        delta_b64=base64.b64encode(b"\x00" * 16).decode(),
    )
    resp = asyncio.run(ar.apply_delta(req, is_admin=True))
    assert resp["ephemeral"] is True
    assert resp["shape"] == [16, 16]
    assert resp["apply_id"] in ar._DELTA_REGISTRY
    assert pool.released == 1


class _FakeExecutor:
    def submit(self, fn, *args):
        import concurrent.futures

        fut = concurrent.futures.Future()
        try:
            fut.set_result(fn(*args))
        except Exception as e:
            fut.set_exception(e)
        return fut


def test_apply_delta_validation_rejects_bad_shape(tmp_path):
    import mlx.core as mx

    target = mx.zeros((4, 4), dtype=mx.float32)
    path = tmp_path / "delta.safetensors"
    mx.save_safetensors(str(path), {"delta": mx.zeros((8, 8), dtype=mx.float32)})
    req = ar.ApplyDeltaRequest(
        model="m",
        param_path="model.layers.7.mlp.down_proj.weight",
        delta_path=str(path),
    )
    import fastapi

    with pytest.raises(fastapi.HTTPException) as ei:
        ar._load_delta(req, target)
    assert "delta shape" in ei.value.detail


def test_apply_delta_tensor_key_missing_rejected(tmp_path):
    import mlx.core as mx

    target = mx.zeros((4, 4), dtype=mx.float32)
    path = tmp_path / "delta.safetensors"
    mx.save_safetensors(str(path), {"delta": mx.zeros((4, 4), dtype=mx.float32)})
    req = ar.ApplyDeltaRequest(
        model="m",
        param_path="model.layers.7.mlp.down_proj.weight",
        delta_path=str(path),
        tensor_key="nope",
    )
    import fastapi

    with pytest.raises(fastapi.HTTPException) as ei:
        ar._load_delta(req, target)
    assert ei.value.status_code == 400


def test_apply_delta_file_happy_path(tmp_path):
    import mlx.core as mx

    delta = mx.ones((4, 4), dtype=mx.float32) * 2.0
    path = tmp_path / "delta.safetensors"
    mx.save_safetensors(str(path), {"delta": delta})
    target = mx.zeros((4, 4), dtype=mx.float32)
    req = ar.ApplyDeltaRequest(
        model="m",
        param_path="model.layers.7.mlp.down_proj.weight",
        delta_path=str(path),
    )
    loaded = ar._load_delta(req, target)
    assert float(loaded[0, 0]) == 2.0


def test_apply_delta_requires_source():
    import fastapi
    import mlx.core as mx

    req = ar.ApplyDeltaRequest(model="m", param_path="p")
    with pytest.raises(fastapi.HTTPException) as ei:
        ar._load_delta(req, mx.zeros((2, 2)))
    assert ei.value.status_code == 400


# ─── rollback + status ────────────────────────────────────────────────────────


def test_delta_status_lists_registry():
    import asyncio

    ar._DELTA_REGISTRY["a1"] = {
        "apply_id": "a1",
        "model": "m1",
        "param_path": "p",
        "scale": 1.0,
        "comment": "",
        "applied_at": "t",
        "old_snapshot": _FakeArr((2, 2)),
        "shape": [2, 2],
        "ephemeral": True,
    }
    resp = asyncio.run(ar.delta_status(model="m1", is_admin=True))
    assert len(resp["deltas"]) == 1
    assert resp["deltas"][0]["apply_id"] == "a1"
    assert "old_snapshot" not in resp["deltas"][0]
    resp_other = asyncio.run(ar.delta_status(model="m2", is_admin=True))
    assert resp_other["deltas"] == []


def test_rollback_unknown_apply_id_404():
    import asyncio

    import fastapi

    _wire(_FakePool(BatchedEngine()))
    req = ar.RollbackDeltaRequest(model="m", apply_id="nope")
    with pytest.raises(fastapi.HTTPException) as ei:
        asyncio.run(ar.rollback_delta(req, is_admin=True))
    assert ei.value.status_code == 404


def test_rollback_requires_selection():
    import asyncio

    import fastapi

    _wire(_FakePool(BatchedEngine()))
    req = ar.RollbackDeltaRequest(model="m")
    with pytest.raises(fastapi.HTTPException) as ei:
        asyncio.run(ar.rollback_delta(req, is_admin=True))
    assert ei.value.status_code == 400


def test_rollback_happy_path_restores_snapshot(monkeypatch):
    import asyncio

    engine = BatchedEngine(num_layers=8)
    pool = _wire(_FakePool(engine))
    snap = _FakeArr((2, 2))
    ar._DELTA_REGISTRY["a1"] = {
        "apply_id": "a1",
        "model": "m",
        "param_path": "model.layers.7.mlp.down_proj.weight",
        "scale": 1.0,
        "comment": "",
        "applied_at": "t",
        "old_snapshot": snap,
        "shape": [2, 2],
        "ephemeral": True,
    }

    def fake_rb(eng, path, old):
        assert eng is engine
        assert path == "model.layers.7.mlp.down_proj.weight"
        assert old is snap
        engine.rolled_back = True

    monkeypatch.setattr(ar, "_rollback_on_engine", fake_rb)
    monkeypatch.setattr(ar, "_executor_for", lambda e: _FakeExecutor())
    req = ar.RollbackDeltaRequest(model="m", apply_id="a1")
    resp = asyncio.run(ar.rollback_delta(req, is_admin=True))
    assert resp["rolled_back"] == ["a1"]
    assert "a1" not in ar._DELTA_REGISTRY
    assert engine.rolled_back is True
    assert pool.released == 1
