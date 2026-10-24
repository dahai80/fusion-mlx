# SPDX-License-Identifier: Apache-2.0
"""#1347: single-model serve must adopt the registry entry for the same
weights instead of registering a duplicate pool entry under the raw
settings string.

Covers Server._canonical_single_model_target: repo-key/slashed spellings
resolve to one entry, already-loaded entries are left alone, and every
failure path degrades to the previous behaviour.
"""

from __future__ import annotations

import types
import unittest

from fusion_mlx.server import Server


class _Entry:
    def __init__(self, model_path, engine=None):
        self.model_path = model_path
        self.engine = engine


class _Pool:
    def __init__(self, entries, resolve):
        self._entries = entries
        self._resolve = resolve
        self.resolved_calls = []

    def _resolve_hf_repo_id(self, model_id):
        self.resolved_calls.append(model_id)
        return self._resolve(model_id)


def _server(pool):
    return types.SimpleNamespace(pool=pool)


class TestCanonicalSingleModelTarget(unittest.TestCase):
    def _call(self, pool, served, model_path):
        return Server._canonical_single_model_target(_server(pool), served, model_path)

    def test_adopts_registry_entry(self):
        entry = _Entry("/models/models--org--m")
        pool = _Pool({"org--m": entry}, lambda mid: "org--m" if mid == "org/m" else mid)
        served, model_path = self._call(pool, "org/m", "org/m")
        self.assertEqual(served, "org--m")
        self.assertEqual(model_path, "/models/models--org--m")
        self.assertEqual(pool.resolved_calls, ["org/m"])

    def test_already_loaded_entry_keeps_original_key(self):
        entry = _Entry("/models/models--org--m", engine=object())
        pool = _Pool({"org--m": entry}, lambda mid: "org--m" if mid == "org/m" else mid)
        served, model_path = self._call(pool, "org/m", "org/m")
        self.assertEqual(served, "org/m")
        self.assertEqual(model_path, "org/m")

    def test_unresolvable_target_unchanged(self):
        pool = _Pool({}, lambda mid: mid)
        served, model_path = self._call(pool, "local/dir", "local/dir")
        self.assertEqual(served, "local/dir")
        self.assertEqual(model_path, "local/dir")

    def test_resolved_key_without_entry_unchanged(self):
        pool = _Pool({}, lambda mid: "org--m")
        served, model_path = self._call(pool, "org/m", "org/m")
        self.assertEqual(served, "org/m")
        self.assertEqual(model_path, "org/m")

    def test_resolve_failure_unchanged(self):
        def _boom(mid):
            raise RuntimeError("registry exploded")

        pool = _Pool({}, _boom)
        served, model_path = self._call(pool, "org/m", "org/m")
        self.assertEqual(served, "org/m")
        self.assertEqual(model_path, "org/m")

    def test_missing_pool_unchanged(self):
        served, model_path = Server._canonical_single_model_target(
            types.SimpleNamespace(), "org/m", "org/m"
        )
        self.assertEqual(served, "org/m")
        self.assertEqual(model_path, "org/m")

    def test_explicit_served_name_replaced_by_registry_key(self):
        # The explicit --served-model-name is kept as an alias by
        # _load_single_model; the primary key becomes the registry entry.
        entry = _Entry("/models/models--org--m")
        pool = _Pool({"org--m": entry}, lambda mid: "org--m" if mid == "org/m" else mid)
        served, model_path = self._call(pool, "my-alias", "org/m")
        self.assertEqual(served, "org--m")
        self.assertEqual(model_path, "/models/models--org--m")


class _FakeEngine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def start(self):
        return None


class _LoadPool(_Pool):
    def __init__(self, entries, resolve, settings_manager=None):
        super().__init__(entries, resolve)
        self._settings_manager = settings_manager
        self.registered = []

    def check_tier_budget(self, tier, size):
        return True, ""

    def register_engine(self, model_id, engine, model_path="", estimated_size=0):
        self.registered.append((model_id, engine, model_path, estimated_size))


class TestSingleModelEngineKind(unittest.TestCase):
    # #1349: a vlm-typed registry entry must load as VLMBatchedEngine, not a
    # text-only BatchedEngine, unless force_text is set.
    def _run(self, engine_type, force_text):
        import asyncio
        from unittest.mock import patch

        captured = {}

        class _B(_FakeEngine):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                captured["kind"] = "batched"

        class _V(_FakeEngine):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                captured["kind"] = "vlm"

        entry = _Entry("org/m")
        entry.engine_type = engine_type
        pool = _LoadPool(
            {"org--m": entry}, lambda mid: "org--m" if mid == "org/m" else mid
        )

        class _Srv:
            _canonical_single_model_target = Server._canonical_single_model_target

        srv = _Srv()
        srv.pool = pool
        srv.engine_cores = {}
        srv.config = types.SimpleNamespace(scheduler=None)
        pending = {
            "model_path": "org/m",
            "original_name": "org/m",
            "served_model_name": None,
            "stream_interval": 1,
            "force_text": force_text,
        }
        with (
            patch("fusion_mlx.engines.batched.BatchedEngine", _B),
            patch("fusion_mlx.engines.vlm.VLMBatchedEngine", _V),
        ):
            asyncio.run(Server._load_single_model(srv, pending))
        return captured, pool

    def test_vlm_entry_loads_vlm_engine(self):
        captured, pool = self._run("vlm", force_text=False)
        self.assertEqual(captured.get("kind"), "vlm")
        self.assertEqual(pool.registered[0][0], "org--m")

    def test_force_text_keeps_batched_engine(self):
        captured, _ = self._run("vlm", force_text=True)
        self.assertEqual(captured.get("kind"), "batched")

    def test_batched_entry_loads_batched_engine(self):
        captured, _ = self._run("batched", force_text=False)
        self.assertEqual(captured.get("kind"), "batched")


if __name__ == "__main__":
    unittest.main()
