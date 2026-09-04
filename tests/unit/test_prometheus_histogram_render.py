# SPDX-License-Identifier: Apache-2.0
import logging

import pytest

from fusion_mlx.server_metrics import Histogram, get_server_metrics

logger = logging.getLogger(__name__)


def test_histogram_observe_and_render():
    h = Histogram(buckets=[0.1, 0.5, 1.0, float("inf")])
    h.observe(0.05)
    h.observe(0.3)
    h.observe(2.0)
    lines = h.render("fusion_mlx_model_ttft_seconds", labels={"model": "m"})
    text = "\n".join(lines)
    logger.info("rendered histogram:\n%s", text)
    assert "_bucket" in text
    assert "_count" in text
    assert "_sum" in text
    assert "_max" in text
    assert 'le="0.1"' in text
    assert 'le="+Inf"' in text
    assert h.count == 3
    assert h.sum == pytest.approx(2.35)
    assert h.max == 2.0
    bucket_counts = {}
    for line in lines:
        if "_bucket" in line:
            if 'le="0.1"' in line:
                bucket_counts["0.1"] = int(line.rsplit(" ", 1)[-1])
            elif 'le="0.5"' in line:
                bucket_counts["0.5"] = int(line.rsplit(" ", 1)[-1])
            elif 'le="1.0"' in line:
                bucket_counts["1.0"] = int(line.rsplit(" ", 1)[-1])
            elif 'le="+Inf"' in line:
                bucket_counts["+Inf"] = int(line.rsplit(" ", 1)[-1])
    assert bucket_counts["0.1"] == 1
    assert bucket_counts["0.5"] == 2
    assert bucket_counts["1.0"] == 2
    assert bucket_counts["+Inf"] == 3


def test_server_metrics_modality_counters():
    m = get_server_metrics()
    m.clear_metrics()
    m.record_modality_request("vision")
    m.record_modality_request("vision")
    m.record_modality_request("audio")
    m.record_modality_request("video")
    m.record_modality_request("image_generation")
    d = m.to_dict()
    logger.info(
        "modality dict: %s",
        {
            k: d[k]
            for k in (
                "vision_requests",
                "audio_requests",
                "video_requests",
                "image_generation_requests",
            )
        },
    )
    assert d["vision_requests"] == 2
    assert d["audio_requests"] == 1
    assert d["video_requests"] == 1
    assert d["image_generation_requests"] == 1
    m.clear_metrics()


def test_server_metrics_lifespan_timestamps():
    m = get_server_metrics()
    m.clear_metrics()
    m.record_startup()
    d = m.to_dict()
    startup = d["startup_epoch"]
    logger.info("startup_epoch=%s", startup)
    assert isinstance(startup, float)
    assert startup > 0.0
    m.record_shutdown()
    d = m.to_dict()
    shutdown = d["shutdown_epoch"]
    logger.info("shutdown_epoch=%s", shutdown)
    assert isinstance(shutdown, float)
    assert shutdown >= startup
    m.clear_metrics()


def test_record_request_complete_feeds_histograms():
    m = get_server_metrics()
    m.clear_metrics()
    m.record_request_complete(
        prompt_tokens=10,
        completion_tokens=5,
        generation_duration=0.5,
        ttft_ms=120.0,
        model_id="test-model",
    )
    ttft = m.get_ttft_histograms()
    tps = m.get_tps_histograms()
    logger.info("ttft hist keys=%s tps hist keys=%s", list(ttft), list(tps))
    assert "test-model" in ttft
    assert "test-model" in tps
    assert ttft["test-model"].count == 1
    assert tps["test-model"].count == 1
    assert ttft["test-model"].sum == pytest.approx(0.12)
    assert tps["test-model"].sum == pytest.approx(10.0)
    m.clear_metrics()


def test_record_request_complete_no_ttft_skips_histogram():
    m = get_server_metrics()
    m.clear_metrics()
    m.record_request_complete(
        prompt_tokens=10,
        completion_tokens=5,
        generation_duration=0.5,
        model_id="m2",
    )
    ttft = m.get_ttft_histograms()
    tps = m.get_tps_histograms()
    logger.info("ttft hist=%s tps hist=%s", list(ttft), list(tps))
    assert "m2" not in ttft
    assert "m2" in tps
    assert tps["m2"].count == 1
    m.clear_metrics()


def test_histogram_snapshot():
    h = Histogram(buckets=[0.1, 1.0, float("inf")])
    h.observe(0.05)
    h.observe(0.5)
    h.observe(2.0)
    snap = h.snapshot()
    logger.info("snapshot=%s", snap)
    assert "buckets" in snap
    assert "counts" in snap
    assert "sum" in snap
    assert "count" in snap
    assert "max" in snap
    assert snap["count"] == 3
    assert snap["sum"] == pytest.approx(2.55)
    assert snap["max"] == 2.0
    assert snap["buckets"] == [0.1, 1.0, float("inf")]
