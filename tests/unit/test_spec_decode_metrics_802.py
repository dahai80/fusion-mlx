# SPDX-License-Identifier: Apache-2.0
# Tests for issue #802: speculative-decoding draft accept-rate metrics exposed
# on /metrics. No real model weights — the dflash/dfly global counters and the
# mtp controller registry are driven directly and render_prometheus_metrics is
# asserted to emit the expected Prometheus series.

from fusion_mlx.routes_internal.metrics import render_prometheus_metrics
from fusion_mlx.speculative.dflash.accept_counter import (
    get_global_counter as get_dflash,
)
from fusion_mlx.speculative.dflash.accept_counter import (
    reset_global_counter_for_tests as reset_dflash,
)
from fusion_mlx.speculative.dfly.accept_counter import (
    get_global_counter as get_dfly,
)
from fusion_mlx.speculative.dfly.accept_counter import (
    reset_global_counter_for_tests as reset_dfly,
)
from fusion_mlx.speculative.mtp.draft_k_controller_v2 import (
    reset_controllers as reset_mtp,
)
from fusion_mlx.speculative.mtp.draft_k_controller_v2 import (
    sum_across_controllers,
)


def test_dflash_accept_rate_rendered():
    reset_dflash()
    ctr = get_dflash()
    # 10 draft attempts, 8 accepted -> accept_ratio 0.8
    for _ in range(10):
        ctr.record_attempt()
    for _ in range(8):
        ctr.record_accept(tokens_saved=1)
    body = render_prometheus_metrics()
    assert 'fusion_mlx_spec_decode_accepted_total{strategy="dflash2"} 8' in body
    assert 'fusion_mlx_spec_decode_drafted_total{strategy="dflash2"} 10' in body
    assert 'fusion_mlx_spec_decode_accept_rate{strategy="dflash2"} 0.8' in body


def test_dfly_accept_rate_rendered():
    reset_dfly()
    ctr = get_dfly()
    ctr.record(accepted=3, drafted=9)
    body = render_prometheus_metrics()
    assert 'fusion_mlx_spec_decode_accepted_total{strategy="dspark"} 3' in body
    assert 'fusion_mlx_spec_decode_drafted_total{strategy="dspark"} 9' in body
    assert 'fusion_mlx_spec_decode_accept_rate{strategy="dspark"} 0.333333' in body


def test_zero_drafted_accept_rate_is_zero_not_nan():
    # Guard: drafted=0 must yield 0.0, never NaN/divide-by-zero.
    reset_dflash()
    get_dflash()  # counter exists, no records
    body = render_prometheus_metrics()
    assert 'fusion_mlx_spec_decode_accept_rate{strategy="dflash2"} 0.0' in body


def test_mtp_round_counters_rendered():
    # MTP keeps a per-model DepthController; sum_across_controllers aggregates
    # round_count + park_count. No direct accept count is exported, so the
    # metric is rounds + park rounds only (scale signal).
    reset_mtp()
    assert sum_across_controllers() == (0, 0, {})
    body = render_prometheus_metrics()
    assert 'fusion_mlx_spec_decode_rounds_total{strategy="mtp"} 0' in body
    assert 'fusion_mlx_spec_decode_park_rounds_total{strategy="mtp"} 0' in body


def test_metrics_endpoint_does_not_error_when_strategies_absent():
    # /metrics must stay green even when no speculative strategy has run.
    # All counters at zero (fresh reset) must render cleanly.
    reset_dflash()
    reset_dfly()
    reset_mtp()
    body = render_prometheus_metrics()
    # All three strategy labels present with zero counters.
    assert 'strategy="dflash2"' in body
    assert 'strategy="dspark"' in body
    assert 'strategy="mtp"' in body
