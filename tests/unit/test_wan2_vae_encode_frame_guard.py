import pytest

from fusion_mlx.video.wan2.vae import _validate_wan2_encode_frames


# Valid frame counts: 1+4N with N>=1 -> {5,9,13,17,21,...} (odd iter_).
@pytest.mark.parametrize("t", [5, 9, 13, 17, 21, 25, 29, 33])
def test_validate_accepts_valid_frame_counts(t):
    # Should not raise; returns None.
    assert _validate_wan2_encode_frames(t) is None


# Even-iter counts (t == 3 mod 4): {3,7,11,15,19,...} -> empty-stack flush.
@pytest.mark.parametrize("t", [3, 7, 11, 15, 19, 23])
def test_validate_rejects_even_iter_counts(t):
    with pytest.raises(ValueError, match=r"1\+4N"):
        _validate_wan2_encode_frames(t)


# Degenerate: T=1 (and T=0/negative) cannot satisfy downsample3d kt=3.
@pytest.mark.parametrize("t", [1, 0, -1, -5])
def test_validate_rejects_degenerate_counts(t):
    with pytest.raises(ValueError, match=r"1\+4N"):
        _validate_wan2_encode_frames(t)


def test_validate_error_message_names_issue_and_frame_count():
    with pytest.raises(ValueError) as exc_info:
        _validate_wan2_encode_frames(7)
    msg = str(exc_info.value)
    assert "T=7" in msg
    assert "#669" in msg
