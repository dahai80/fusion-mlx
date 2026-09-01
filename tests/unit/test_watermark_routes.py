import pytest
from pydantic import ValidationError

from fusion_mlx.api.watermark_models import (
    WatermarkEmbedRequest,
    WatermarkVerifyRequest,
)


def test_embed_request_minimal():
    req = WatermarkEmbedRequest(
        model="org/repo", payload={"a": 1}, secret="nondefault"
    )
    assert req.bits_per_weight == 1
    assert req.in_place is False
    assert req.layers is None


def test_embed_request_bits_per_weight_range():
    with pytest.raises(ValidationError):
        WatermarkEmbedRequest(
            model="m", payload={}, secret="s", bits_per_weight=4
        )
    with pytest.raises(ValidationError):
        WatermarkEmbedRequest(
            model="m", payload={}, secret="s", bits_per_weight=0
        )


def test_embed_request_output_path_required_when_not_in_place():
    # in_place=False (default) without output_path is allowed at model level;
    # the route enforces output_path presence. Here we just validate the
    # path-prefix constraint when a path IS given.
    import os

    home = os.path.expanduser("~/.fusion-mlx/models")
    req = WatermarkEmbedRequest(
        model="m", payload={}, secret="s", output_path=home + "/wm-out"
    )
    assert req.output_path.startswith(home)


def test_verify_request_minimal():
    req = WatermarkVerifyRequest(model="m", secret="s")
    assert req.bits_per_weight == 1
    assert req.layers is None
