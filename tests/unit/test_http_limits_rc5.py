import os

import httpx

from fusion_mlx._http_limits import bounded_limits


def test_default_limits_bounded():
    for k in (
        "FUSION_HTTP_MAX_CONNECTIONS",
        "FUSION_HTTP_MAX_KEEPALIVE",
        "FUSION_HTTP_KEEPALIVE_EXPIRY",
    ):
        os.environ.pop(k, None)
    lim = bounded_limits()
    assert isinstance(lim, httpx.Limits)
    assert lim.max_connections == 64
    assert lim.max_keepalive_connections == 16
    assert lim.keepalive_expiry == 5.0


def test_env_overrides_respected():
    os.environ["FUSION_HTTP_MAX_CONNECTIONS"] = "32"
    os.environ["FUSION_HTTP_MAX_KEEPALIVE"] = "8"
    os.environ["FUSION_HTTP_KEEPALIVE_EXPIRY"] = "2.5"
    try:
        lim = bounded_limits()
        assert lim.max_connections == 32
        assert lim.max_keepalive_connections == 8
        assert lim.keepalive_expiry == 2.5
    finally:
        for k in (
            "FUSION_HTTP_MAX_CONNECTIONS",
            "FUSION_HTTP_MAX_KEEPALIVE",
            "FUSION_HTTP_KEEPALIVE_EXPIRY",
        ):
            os.environ.pop(k, None)


def test_invalid_env_falls_back_to_default():
    os.environ["FUSION_HTTP_MAX_CONNECTIONS"] = "0"
    os.environ["FUSION_HTTP_MAX_KEEPALIVE"] = "not-a-number"
    os.environ["FUSION_HTTP_KEEPALIVE_EXPIRY"] = "-1"
    try:
        lim = bounded_limits()
        assert lim.max_connections == 64
        assert lim.max_keepalive_connections == 16
        assert lim.keepalive_expiry == 5.0
    finally:
        for k in (
            "FUSION_HTTP_MAX_CONNECTIONS",
            "FUSION_HTTP_MAX_KEEPALIVE",
            "FUSION_HTTP_KEEPALIVE_EXPIRY",
        ):
            os.environ.pop(k, None)


def test_limits_have_finite_cap():
    # RC-5 core contract: must never return the unbounded httpx default
    # (max_connections is a finite int, not None).
    lim = bounded_limits()
    assert lim.max_connections is not None
    assert lim.max_connections > 0
