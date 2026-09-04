# SPDX-License-Identifier: Apache-2.0
from fusion_mlx.telemetry.coherence import (
    is_abnormally_short,
    is_empty,
    looks_like_garbage,
)


def test_empty_when_zero_tokens():
    assert is_empty(0) is True
    assert is_empty(5) is False


def test_garbage_repetition():
    assert looks_like_garbage("aaaaaaaaaaaaaaaa", 16) is True
    assert looks_like_garbage("hello world this is fine", 10) is False


def test_garbage_when_tokens_but_empty_text():
    assert looks_like_garbage("", 10) is True


def test_abnormally_short():
    assert is_abnormally_short("hi", 2) is True
    assert is_abnormally_short("hello world paragraph", 20) is False
