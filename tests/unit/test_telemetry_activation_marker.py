# SPDX-License-Identifier: Apache-2.0
import pytest

from fusion_mlx.telemetry import state


@pytest.fixture
def tmp_telemetry_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "_default_telemetry_dir", lambda: tmp_path)
    monkeypatch.setattr(state, "_activation_latch", set())
    return tmp_path


def test_claim_once_per_install(tmp_telemetry_dir):
    assert state.claim_activation_marker("first_inference") is True
    assert state.claim_activation_marker("first_inference") is False


def test_claim_rejects_unknown_kind(tmp_telemetry_dir):
    with pytest.raises(ValueError):
        state.claim_activation_marker("bogus_kind")


def test_reset_wipes_markers(tmp_telemetry_dir):
    state.claim_activation_marker("first_inference")
    state.reset_state()
    assert state.claim_activation_marker("first_inference") is True


def test_consent_schema_version_is_3():
    assert state.CURRENT_CONSENT_SCHEMA_VERSION == 3
