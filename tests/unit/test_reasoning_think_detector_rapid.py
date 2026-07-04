# SPDX-License-Identifier: Apache-2.0
import logging

from fusion_mlx.reasoning.think_detector import (
    ThinkDetector,
    looks_like_autonomous_think,
)

logger = logging.getLogger(__name__)

THINK_OPEN = "<think>"


def test_looks_like_autonomous_think_simple():
    assert looks_like_autonomous_think(THINK_OPEN)
    assert looks_like_autonomous_think(THINK_OPEN + "Step 1: ...")


def test_looks_like_autonomous_think_with_leading_whitespace():
    assert looks_like_autonomous_think("   " + THINK_OPEN)
    assert looks_like_autonomous_think("\n\n" + THINK_OPEN)


def test_looks_like_autonomous_think_negatives():
    assert not looks_like_autonomous_think("")
    assert not looks_like_autonomous_think("The answer is 42")
    assert not looks_like_autonomous_think("Okay let me think.")
    long_prefix = "A" * 200 + THINK_OPEN
    assert not looks_like_autonomous_think(long_prefix)


def test_detector_routes_to_thinking_on_opener():
    det = ThinkDetector()
    assert det.feed(THINK_OPEN)
    assert det.is_decided


def test_detector_commits_to_content_after_threshold():
    det = ThinkDetector()
    assert not det.feed("The answer is 42 right now")
    assert not det.is_decided
    assert not det.feed("A" * (ThinkDetector.INSPECT_BYTES + 5))
    assert det.is_decided


def test_detector_decision_is_sticky():
    det = ThinkDetector()
    det.feed(THINK_OPEN + "reasoning here")
    assert det.feed("regular content from now on")


def test_detector_reset():
    det = ThinkDetector()
    det.feed(THINK_OPEN)
    det.reset()
    assert not det.is_decided
    assert not det.feed("plain text without opener" + "A" * 100)
