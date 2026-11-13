import threading

from fusion_mlx.pool.engine_pool import EngineEntry, EnginePool


def _pool() -> EnginePool:
    # Minimal stand-in: the reporting accessors only touch _entries,
    # _current_model_memory and the discover lock.
    pool = EnginePool.__new__(EnginePool)
    pool._entries = {}
    pool._current_model_memory = 0
    pool._discover_sync_lock = threading.Lock()
    return pool


def _entry(mid, engine=None, **kw):
    kw.setdefault("estimated_size", 0)
    return EngineEntry(
        model_id=mid,
        model_path=f"/models/{mid}",
        model_type="llm",
        engine_type="batched",
        **kw,
        engine=engine,
    )


class _FakeEngine:
    pass


class TestAliasDedupe1592:
    def test_loaded_ids_skip_alias_entries(self):
        pool = _pool()
        eng = _FakeEngine()
        pool._entries["canon--model"] = _entry("canon--model", engine=eng)
        pool._entries["org/model"] = _entry(
            "org/model", engine=eng, alias_of="canon--model"
        )
        assert pool.get_loaded_model_ids() == ["canon--model"]

    def test_alias_without_engine_not_listed(self):
        pool = _pool()
        pool._entries["canon--model"] = _entry("canon--model", engine=_FakeEngine())
        pool._entries["org/model"] = _entry("org/model", alias_of="canon--model")
        assert pool.get_loaded_model_ids() == ["canon--model"]

    def test_memory_tiers_skip_alias_entries(self):
        pool = _pool()
        eng = _FakeEngine()
        pool._entries["canon--model"] = _entry(
            "canon--model", engine=eng, estimated_size=2 * 1024**3
        )
        pool._entries["org/model"] = _entry(
            "org/model", engine=eng, alias_of="canon--model"
        )
        rows = pool.get_memory_tiers()
        all_models = [r["model"] for r in rows["P0"] + rows["P1"]]
        assert all_models == ["canon--model"]

    def test_register_engine_marks_alias_only_on_new_entries(self):
        pool = _pool()
        eng = _FakeEngine()
        pool.register_engine(
            "canon--model", eng, model_path="/models/x", estimated_size=8
        )
        pool.register_engine(
            "org/model", eng, model_path="/models/x", alias_of="canon--model"
        )
        assert pool._entries["org/model"].alias_of == "canon--model"
        assert pool.get_loaded_model_ids() == ["canon--model"]

    def test_register_alias_onto_existing_entry_does_not_tag_it(self):
        # Discovery created a real entry first; an alias registration with
        # the same name must attach the engine but NOT mark it alias_of,
        # or a legit model disappears from reporting.
        pool = _pool()
        existing = _entry("shared-name")
        pool._entries["shared-name"] = existing
        pool.register_engine(
            "shared-name", _FakeEngine(), model_path="/models/x", alias_of="canon"
        )
        assert existing.alias_of == ""
        assert pool.get_loaded_model_ids() == ["shared-name"]

    def test_alias_registration_does_not_double_count_memory(self):
        pool = _pool()
        eng = _FakeEngine()
        pool.register_engine(
            "canon--model", eng, model_path="/models/x", estimated_size=8
        )
        assert pool._current_model_memory == 8
        pool.register_engine(
            "org/model", eng, model_path="/models/x", alias_of="canon--model"
        )
        # Alias entry carries no estimated_size — residency counted once.
        assert pool._current_model_memory == 8
