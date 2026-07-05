from __future__ import annotations

import logging

from fusion_mlx.reasoning.think_detector import (
    ThinkDetector,
    looks_like_autonomous_think,
)

logger = logging.getLogger(__name__)


def test_looks_like_autonomous_think_simple():
    assert looks_like_autonomous_think("<think>")
    assert looks_like_autonomous_think("<think>Step 1: ...")


def test_looks_like_autonomous_think_with_leading_whitespace():
    """Distilled-on-Qwen models often emit a couple of space tokens
    before the opener."""
    assert looks_like_autonomous_think("  <think>")
    assert looks_like_autonomous_think("\n\n<think>")


def test_looks_like_autonomous_think_negatives():
    assert not looks_like_autonomous_think("")
    assert not looks_like_autonomous_think("The answer is 42")
    assert not looks_like_autonomous_think("Okay let me think.")
    # ``<think>`` deep inside a response is NOT an autonomous opener.
    long_prefix = "A" * 200 + "<think>"
    assert not looks_like_autonomous_think(long_prefix)


def test_detector_routes_to_thinking_on_opener():
    det = ThinkDetector()
    assert det.feed("<think>")
    assert det.is_decided


def test_detector_commits_to_content_after_threshold():
    det = ThinkDetector()
    # First 30 chars don't open with <think>
    assert not det.feed("The answer is 42 right now")
    assert not det.is_decided
    # Past the inspection window
    assert not det.feed("A" * (ThinkDetector.INSPECT_BYTES + 5))
    assert det.is_decided


def test_detector_decision_is_sticky():
    det = ThinkDetector()
    det.feed("<think>reasoning here")
    # Even if subsequent text doesn't look thinking-shaped, the routing
    # stays.
    assert det.feed("regular content from now on")


def test_detector_reset():
    det = ThinkDetector()
    det.feed("<think>")
    det.reset()
    assert not det.is_decided
    assert not det.feed("plain text without opener" + "A" * 100)
