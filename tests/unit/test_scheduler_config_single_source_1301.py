# SPDX-License-Identifier: Apache-2.0
# #1301 guard: ONE SchedulerConfig source of truth across the serve path.
#
# The serve path builds a single SchedulerConfig (cli_serve.serve_command),
# applies the settings.json cache section to it, publishes it on the config
# singleton, and load_model() stamps that same object onto the live Server.
# The EnginePool (on-demand loads) and the single-model engine consume it.
#
# Before the refactor Server._convert_scheduler_config() copied a
# hand-maintained list of field names into a fresh SchedulerConfig. Any field
# missing from that list was silently dropped, so an on-demand pool load could
# boot with defaults (1GiB pure-memory prefix cache) while the boot-time model
# ran the settings.json SSD config.
#
# These tests are the mechanical guard for that trap:
#
# * every field declared on SchedulerConfig (reflected via dataclasses.fields,
#   no manual list) is stamped with a sentinel and must survive
#   ServerConfig.scheduler -> Server conversion -> EnginePool -> engine;
#   failures are reported by field name;
# * the serve path constructs SchedulerConfig exactly once (AST count).
#
# Run the first test against a partial-copy implementation and it fails with
# the dropped field names — that is the sensitivity the issue asks for.

from __future__ import annotations

import ast
import dataclasses
import types
import typing
from pathlib import Path

from fusion_mlx.config import (
    SchedulerConfig,
    SchedulingPolicy,
    ServerConfig,
    get_config,
    reset_config,
)

_HINTS = typing.get_type_hints(SchedulerConfig)


def _annotation_sentinel(hint):
    origin = typing.get_origin(hint)
    if isinstance(hint, types.UnionType) or origin is typing.Union:
        for arg in typing.get_args(hint):
            if arg is not type(None):
                return _annotation_sentinel(arg)
        return None
    if hint is bool:
        return True
    if hint is int:
        return 101
    if hint is float:
        return 0.37
    if hint is str:
        return "sentinel-1301"
    if hint is SchedulingPolicy:
        return SchedulingPolicy.PRIORITY
    # Any (pflash_config): a plain marker is safe — nothing in the chain
    # parses it, it only has to round-trip.
    return "sentinel-1301"


def _raw_sentinel(field, current):
    # Values are derived from the field's DEFAULT, not from a name list, so a
    # newly added field is automatically covered.
    if isinstance(current, bool):
        return not current
    if isinstance(current, SchedulingPolicy):
        return (
            SchedulingPolicy.PRIORITY
            if current is SchedulingPolicy.FCFS
            else SchedulingPolicy.FCFS
        )
    if isinstance(current, int):
        # 4/8/16 bit widths are re-clamped by __post_init__; stay inside the
        # valid set so a copy-based conversion is not reported as drift.
        if current in (4, 8, 16):
            return 4 if current != 4 else 16
        return current + 101
    if isinstance(current, float):
        return round(current + 0.07, 4)
    if isinstance(current, str):
        return f"{current}-sentinel-1301"
    return _annotation_sentinel(_HINTS[field.name])


def _accepted_values(field, raw):
    # A conversion that rebuilds the config (dataclasses.replace) re-runs
    # __post_init__, which may normalize a sentinel (chunked_prefill is
    # re-derived from chunked_prefill_tokens). Both forms are accepted; a
    # DROPPED field lands on the dataclass default instead and matches
    # neither.
    accepted = [raw]
    try:
        rebuilt = SchedulerConfig(**{field.name: raw})
    except Exception:
        return accepted
    accepted.append(getattr(rebuilt, field.name))
    return accepted


def _build_chain(cfg):
    from fusion_mlx.engines.batched import BatchedEngine
    from fusion_mlx.pool.engine_pool import EnginePool
    from fusion_mlx.server import Server

    server_stub = types.SimpleNamespace(config=ServerConfig(scheduler=cfg))
    converted = Server._convert_scheduler_config(server_stub)
    pool = EnginePool(scheduler_config=converted)
    engine = BatchedEngine(
        model_name="sentinel-1301/model",
        scheduler_config=pool._scheduler_config,
    )
    return [
        ("Server._convert_scheduler_config", converted),
        ("EnginePool._scheduler_config", pool._scheduler_config),
        ("BatchedEngine._scheduler_config", engine._scheduler_config),
    ]


def test_every_scheduler_config_field_survives_conversion_chain():
    fields = dataclasses.fields(SchedulerConfig)
    assert fields, "SchedulerConfig declares no fields"

    cfg = SchedulerConfig()
    raw = {}
    for field in fields:
        raw[field.name] = _raw_sentinel(field, getattr(cfg, field.name))
        setattr(cfg, field.name, raw[field.name])

    dropped = []
    for stage, obj in _build_chain(cfg):
        for field in fields:
            accepted = _accepted_values(field, raw[field.name])
            got = getattr(obj, field.name)
            if not any(got == want for want in accepted):
                dropped.append(
                    f"{stage}.{field.name}: got {got!r}, "
                    f"expected {raw[field.name]!r}"
                )

    assert not dropped, (
        "scheduler config fields were dropped between the serve-built "
        "SchedulerConfig and the engine — a conversion point still copies a "
        "subset of fields:\n  " + "\n  ".join(dropped)
    )


def test_serve_path_builds_exactly_one_scheduler_config():
    import importlib

    module = importlib.import_module("fusion_mlx.cli_serve.serve_command")
    tree = ast.parse(Path(module.__file__).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SchedulerConfig"
    ]
    assert len(calls) == 1, (
        "serve_command must build exactly one SchedulerConfig "
        f"(found {len(calls)} at lines {[c.lineno for c in calls]})"
    )


def test_load_model_publishes_serve_config_as_the_single_source(monkeypatch):
    from fusion_mlx import server

    reset_config()
    built = SchedulerConfig(
        max_num_seqs=17,
        paged_ssd_cache_dir="/tmp/sentinel-1301-ssd",
        hot_cache_max_size=42 * 1024 * 1024,
        initial_cache_blocks=4242,
    )
    live = types.SimpleNamespace(config=ServerConfig())
    monkeypatch.setattr(server, "_server_instance", live)
    monkeypatch.setattr(server, "_pending_single_model", None)
    monkeypatch.setattr(server, "_resolve_single_model_path", lambda name: name)
    monkeypatch.setattr(server, "get_app", lambda: None)

    try:
        server.load_model("sentinel-1301/model", scheduler_config=built)

        assert get_config().scheduler is built
        assert live.config.scheduler is built
        assert server._pending_single_model is not None
        assert "scheduler_config" not in server._pending_single_model
    finally:
        reset_config()
