from types import SimpleNamespace

from fusion_mlx.engines.vlm import VLMBatchedEngine


def _make_engine(model_settings, scheduler=None):
    engine = VLMBatchedEngine.__new__(VLMBatchedEngine)
    engine._model_settings = model_settings
    engine._model_name = "test_vlm"
    sched = scheduler if scheduler is not None else SimpleNamespace()
    engine._engine = SimpleNamespace(engine=SimpleNamespace(scheduler=sched))
    return engine


def test_enabled_sets_scheduler_bits():
    sched = SimpleNamespace()
    engine = _make_engine(
        SimpleNamespace(
            turboquant_kv_enabled=True,
            turboquant_kv_bits=4,
            turboquant_skip_last=True,
            kv_cache_turboquant_mode=None,
        ),
        scheduler=sched,
    )
    engine._apply_turboquant_kv()
    assert sched._turboquant_kv_bits == 4.0
    assert sched._turboquant_skip_last is True
    assert sched._turboquant_kv_mode == "v4"


def test_disabled_does_not_set_scheduler():
    sched = SimpleNamespace()
    engine = _make_engine(
        SimpleNamespace(
            turboquant_kv_enabled=False,
            turboquant_kv_bits=4,
            turboquant_skip_last=True,
            kv_cache_turboquant_mode=None,
        ),
        scheduler=sched,
    )
    engine._apply_turboquant_kv()
    assert not hasattr(sched, "_turboquant_kv_bits")
    assert not hasattr(sched, "_turboquant_skip_last")
    assert not hasattr(sched, "_turboquant_kv_mode")


def test_none_settings_is_noop():
    sched = SimpleNamespace()
    engine = _make_engine(None, scheduler=sched)
    engine._apply_turboquant_kv()
    assert not hasattr(sched, "_turboquant_kv_bits")


def test_defaults_when_bits_skip_unset():
    sched = SimpleNamespace()
    engine = _make_engine(
        SimpleNamespace(
            turboquant_kv_enabled=True,
            turboquant_kv_bits=None,
            turboquant_skip_last=None,
            kv_cache_turboquant_mode=None,
        ),
        scheduler=sched,
    )
    engine._apply_turboquant_kv()
    assert sched._turboquant_kv_bits == 4.0
    assert sched._turboquant_skip_last is None
    assert sched._turboquant_kv_mode == "v4"


def test_custom_bits_and_k8v4_mode_pass_through():
    sched = SimpleNamespace()
    engine = _make_engine(
        SimpleNamespace(
            turboquant_kv_enabled=True,
            turboquant_kv_bits=2,
            turboquant_skip_last=False,
            kv_cache_turboquant_mode="k8v4",
        ),
        scheduler=sched,
    )
    engine._apply_turboquant_kv()
    assert sched._turboquant_kv_bits == 2.0
    assert sched._turboquant_skip_last is False
    assert sched._turboquant_kv_mode == "k8v4"


def test_invalid_mode_defaults_to_v4(caplog):
    sched = SimpleNamespace()
    engine = _make_engine(
        SimpleNamespace(
            turboquant_kv_enabled=True,
            turboquant_kv_bits=4,
            turboquant_skip_last=True,
            kv_cache_turboquant_mode="bogus",
        ),
        scheduler=sched,
    )
    engine._apply_turboquant_kv()
    assert sched._turboquant_kv_mode == "v4"
    assert any("not in ('v4', 'k8v4')" in r.message for r in caplog.records)


def test_scheduler_missing_is_swallowed_and_logged(caplog):
    # engine.engine has no scheduler attr -> AttributeError caught, logged
    engine = VLMBatchedEngine.__new__(VLMBatchedEngine)
    engine._model_settings = SimpleNamespace(
        turboquant_kv_enabled=True,
        turboquant_kv_bits=4,
        turboquant_skip_last=True,
        kv_cache_turboquant_mode=None,
    )
    engine._model_name = "test_vlm"
    engine._engine = SimpleNamespace(engine=SimpleNamespace())
    engine._apply_turboquant_kv()
    assert any("TurboQuant KV init failed" in r.message for r in caplog.records)
