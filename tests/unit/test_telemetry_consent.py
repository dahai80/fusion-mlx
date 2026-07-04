# SPDX-License-Identifier: Apache-2.0
import importlib
import logging

import pytest

logger = logging.getLogger(__name__)


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RAPID_MLX_TELEMETRY", raising=False)
    import fusion_mlx.telemetry.state as state

    importlib.reload(state)
    return tmp_path


def test_default_is_off(fake_home):
    from fusion_mlx.telemetry.state import is_enabled

    assert is_enabled() is False


def test_consent_round_trip(fake_home):
    from fusion_mlx.telemetry.state import (
        get_consent_state,
        is_enabled,
        record_consent,
    )

    assert get_consent_state() is None
    record_consent(True, rapid_mlx_version="0.6.33")
    state = get_consent_state()
    assert state is not None
    assert state.consent is True
    assert state.prompted_version == "0.6.33"
    assert state.schema_version == 1
    assert state.prompted_at.endswith("Z")
    assert is_enabled() is True


def test_env_kill_switch_wins_over_consent(fake_home, monkeypatch):
    from fusion_mlx.telemetry.state import is_enabled, record_consent

    record_consent(True, rapid_mlx_version="0.6.33")
    assert is_enabled() is True
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "0")
    assert is_enabled() is False


def test_cli_flag_wins_over_consent(fake_home):
    from fusion_mlx.telemetry.state import is_enabled, record_consent

    record_consent(True, rapid_mlx_version="0.6.33")
    assert is_enabled() is True
    assert is_enabled(cli_no_telemetry=True) is False


def test_env_force_on_is_ignored(fake_home, monkeypatch):
    from fusion_mlx.telemetry.state import is_enabled

    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "1")
    assert is_enabled() is False
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "true")
    assert is_enabled() is False


@pytest.mark.parametrize("falsy", ["0", "false", "FALSE", "no", "off", "  0  ", ""])
def test_env_falsy_values_all_disable(fake_home, monkeypatch, falsy):
    from fusion_mlx.telemetry.state import is_enabled, record_consent

    record_consent(True, rapid_mlx_version="0.6.33")
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", falsy)
    assert is_enabled() is False, f"falsy value {falsy!r} should kill-switch"


def test_client_id_idempotent(fake_home):
    from fusion_mlx.telemetry.state import get_or_create_client_id

    first = get_or_create_client_id()
    assert first
    assert len(first) == 36
    assert get_or_create_client_id() == first


def test_client_id_user_zeroed_uuid_preserved(fake_home):
    from fusion_mlx.telemetry.state import client_id_path, get_or_create_client_id

    zero = "00000000-0000-0000-0000-000000000000"
    path = client_id_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(zero + "\n")
    assert get_or_create_client_id() == zero


def test_reset_state_removes_both_files(fake_home):
    from fusion_mlx.telemetry.state import (
        client_id_path,
        consent_path,
        get_or_create_client_id,
        record_consent,
        reset_state,
    )

    record_consent(True, rapid_mlx_version="0.6.33")
    get_or_create_client_id()
    assert consent_path().exists()
    assert client_id_path().exists()
    reset_state()
    assert not consent_path().exists()
    assert not client_id_path().exists()
    reset_state()


def test_consent_source_reports_origin(fake_home, monkeypatch):
    from fusion_mlx.telemetry.state import consent_source, record_consent

    assert "default" in consent_source()
    record_consent(True, rapid_mlx_version="0.6.33")
    assert "consent-file" in consent_source()
    monkeypatch.setenv("RAPID_MLX_TELEMETRY", "0")
    assert "env-var" in consent_source()
    monkeypatch.delenv("RAPID_MLX_TELEMETRY")
    assert "cli-flag" in consent_source(cli_no_telemetry=True)


def test_corrupt_consent_file_treated_as_unprompted(fake_home):
    from fusion_mlx.telemetry.state import consent_path, get_consent_state

    path = consent_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(":\n  not valid yaml :: at all")
    assert get_consent_state() is None


def test_consent_file_atomic_write(fake_home):
    from fusion_mlx.telemetry.state import consent_path, record_consent

    record_consent(True, rapid_mlx_version="0.6.33")
    leftover = consent_path().with_suffix(consent_path().suffix + ".tmp")
    assert not leftover.exists()


def test_record_consent_cleans_up_stale_tmp(fake_home):
    import yaml

    from fusion_mlx.telemetry.state import (
        consent_path,
        get_consent_state,
        record_consent,
    )

    cpath = consent_path()
    cpath.parent.mkdir(parents=True, exist_ok=True)
    stale = cpath.with_suffix(cpath.suffix + ".tmp")
    stale.write_text("partial: junk\nthis is not valid")
    assert stale.exists()

    record_consent(True, rapid_mlx_version="0.6.33")
    assert not stale.exists(), "stale .tmp should be cleaned up"
    state = get_consent_state()
    assert state is not None
    assert state.consent is True
    parsed = yaml.safe_load(cpath.read_text())
    assert parsed["consent"] is True


def test_schema_version_mismatch_treated_as_unprompted(fake_home):
    import yaml

    from fusion_mlx.telemetry.state import consent_path, get_consent_state

    cpath = consent_path()
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(
        yaml.safe_dump(
            {
                "consent": True,
                "prompted_at": "2026-05-10T00:00:00Z",
                "prompted_version": "0.6.33",
                "schema_version": 99,
            }
        )
    )
    assert get_consent_state() is None
