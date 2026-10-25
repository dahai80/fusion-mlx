# SPDX-License-Identifier: Apache-2.0
# The abort path (#1038) warns when a present spec state lacks
# cleanup_request, treating it as a drifted class that leaks per-request
# state. NGramSpecState had no such method, so every aborted n-gram request
# logged "draft state may leak" even though nothing leaked.

import logging

from fusion_mlx.scheduler.ngram_spec import NGramSpecState


def _active_state(request_id: str) -> NGramSpecState:
    state = NGramSpecState(order=2, num_draft=2)
    state.on_new_request(request_id)
    for token in (1, 2, 3, 4):
        state.add_token(token)
    state.track_d1_match(4)
    state._paused = True
    return state


def test_ngram_state_exposes_cleanup_request():
    # The scheduler's abort path checks hasattr(..., "cleanup_request"); the
    # missing method was what produced the warning.
    assert hasattr(NGramSpecState, "cleanup_request")


def test_cleanup_request_drops_ownership_and_adaptive_state():
    state = _active_state("req-a")
    state.cleanup_request("req-a")

    assert state._last_request_id is None
    assert state.steps == 0
    assert state._paused is False
    assert state._recent_rates == []
    assert state._d1_rates == []
    assert state._probing is False


def test_cleanup_request_ignores_other_requests():
    state = _active_state("req-a")
    state.cleanup_request("req-b")

    assert state._last_request_id == "req-a"


def test_abort_path_no_longer_warns(caplog):
    state = _active_state("req-a")
    with caplog.at_level(logging.WARNING, logger="fusion_mlx.scheduler.sched_trim"):
        if hasattr(state, "cleanup_request"):
            state.cleanup_request("req-a")
        else:
            logging.getLogger("fusion_mlx.scheduler.sched_trim").warning(
                "ngram spec state present but lacks cleanup_request for "
                "aborted %s — draft state may leak",
                "req-a",
            )

    assert "draft state may leak" not in caplog.text
