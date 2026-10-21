# SPDX-License-Identifier: Apache-2.0
"""#1288: single-model serve must apply the settings.json cache section.

Prod wedge (2026-10-20): single-model serve built SchedulerConfig without the
cache section (only --model-dir read it), so the engine booted the 1GiB
in-memory LRU (hot_cache_only, no SSD dir) while settings.json configured a
20GB hot cache + SSD dir. One 38k-token store thrashed the hot LRU, defeated
dedup, filled the 4096-block paged pool (17.9k OOB warnings, 29 force-drains).
"""

import json

from fusion_mlx.cli_serve.serve_command import _apply_settings_cache_section
from fusion_mlx.config import SchedulerConfig


def _write_settings(tmp_path, cache_section):
    settings = {}
    if cache_section is not None:
        settings["cache"] = cache_section
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    return str(tmp_path)


def test_applies_full_cache_section(tmp_path):
    d = _write_settings(
        tmp_path,
        {
            "enabled": True,
            "hot_cache_only": False,
            "hot_cache_max_size": "20GB",
            "paged_ssd_cache_dir": "/tmp/ssd-test",
            "initial_cache_blocks": 384,
            "ssd_cache_max_size": "500GB",
        },
    )
    cfg = SchedulerConfig()
    assert _apply_settings_cache_section(cfg, d) is True
    assert cfg.paged_ssd_cache_dir == "/tmp/ssd-test"
    assert cfg.hot_cache_max_size == 20 * 1000**3
    assert cfg.hot_cache_only is False
    assert cfg.initial_cache_blocks == 384
    assert cfg.paged_ssd_cache_max_size == 500 * 1000**3


def test_missing_file_is_noop(tmp_path):
    cfg = SchedulerConfig()
    result = _apply_settings_cache_section(cfg, str(tmp_path))
    assert result is False
    assert cfg.paged_ssd_cache_dir is None
    assert cfg.hot_cache_max_size == 0


def test_malformed_json_is_noop(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    cfg = SchedulerConfig()
    result = _apply_settings_cache_section(cfg, str(tmp_path))
    assert result is False
    assert cfg.hot_cache_max_size == 0


def test_no_cache_section_is_noop(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"server": {}}), encoding="utf-8")
    cfg = SchedulerConfig()
    result = _apply_settings_cache_section(cfg, str(tmp_path))
    assert result is False
    assert cfg.hot_cache_max_size == 0


def test_partial_section_applies_only_present_fields(tmp_path):
    d = _write_settings(tmp_path, {"hot_cache_max_size": "2GB"})
    cfg = SchedulerConfig()
    result = _apply_settings_cache_section(cfg, d)
    assert result is True
    assert cfg.hot_cache_max_size == 2 * 1000**3
    assert cfg.paged_ssd_cache_dir is None
    assert cfg.hot_cache_only is False


def test_singleton_scheduler_config_takes_cache_section(tmp_path):
    from fusion_mlx.config import get_config

    d = _write_settings(
        tmp_path,
        {
            "enabled": True,
            "hot_cache_only": False,
            "hot_cache_max_size": "20GB",
            "paged_ssd_cache_dir": "/tmp/ssd-pool-test",
        },
    )
    from dataclasses import fields, replace

    original = get_config().scheduler
    snapshot = replace(original)
    try:
        assert _apply_settings_cache_section(original, d) is True
        assert original.paged_ssd_cache_dir == "/tmp/ssd-pool-test"
        assert original.hot_cache_max_size == 20 * 1000**3
        assert original.hot_cache_only is False
    finally:
        for f in fields(original):
            setattr(original, f.name, getattr(snapshot, f.name))
