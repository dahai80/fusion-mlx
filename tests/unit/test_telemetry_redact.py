# SPDX-License-Identifier: Apache-2.0
import logging

import pytest

from fusion_mlx.telemetry.redact import (
    bucket_memory_gb,
    bucket_tokens,
    bucket_tps,
    bucket_ttft_ms,
    fingerprint_traceback,
    hash_flag_names,
    normalize_model_path,
    platform_info,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "n,expected",
    [
        (-1, "0-256"),
        (0, "0-256"),
        (255, "0-256"),
        (256, "256-1k"),
        (1023, "256-1k"),
        (1024, "1k-4k"),
        (4095, "1k-4k"),
        (4096, "4k-16k"),
        (16384, "16k-64k"),
        (65535, "16k-64k"),
        (65536, "64k+"),
        (1_000_000, "64k+"),
    ],
)
def test_bucket_tokens_boundaries(n, expected):
    assert bucket_tokens(n) == expected


@pytest.mark.parametrize(
    "ms,expected",
    [
        (-1, "<100ms"),
        (0, "<100ms"),
        (99, "<100ms"),
        (100, "100-500ms"),
        (499, "100-500ms"),
        (500, "500-1500ms"),
        (1500, "1.5-5s"),
        (4999, "1.5-5s"),
        (5000, ">5s"),
    ],
)
def test_bucket_ttft_boundaries(ms, expected):
    assert bucket_ttft_ms(ms) == expected


@pytest.mark.parametrize(
    "tps,expected",
    [
        (-1, "<10"),
        (0, "<10"),
        (9.999, "<10"),
        (10, "10-30"),
        (29.99, "10-30"),
        (30, "30-50"),
        (50, "50-100"),
        (99.99, "50-100"),
        (100, ">100"),
        (1000, ">100"),
    ],
)
def test_bucket_tps_boundaries(tps, expected):
    assert bucket_tps(tps) == expected


def test_bucket_memory_rounds_and_clamps():
    assert bucket_memory_gb(0) == 0
    assert bucket_memory_gb(-1) == 0
    assert bucket_memory_gb(1024**3) == 1
    assert bucket_memory_gb(int(1.4 * 1024**3)) == 1
    assert bucket_memory_gb(int(1.6 * 1024**3)) == 2
    assert bucket_memory_gb(256 * 1024**3) == 256


@pytest.mark.parametrize(
    "raw",
    [
        "mlx-community/Qwen3.5-9B-4bit",
        "huggingface/transformers",
        "user_name/model.v2",
        "a/b",
    ],
)
def test_normalize_model_path_passes_hf_repo_ids(raw):
    assert normalize_model_path(raw) == raw


@pytest.mark.parametrize(
    "raw",
    [
        "/Users/alice/models/foo",
        "./local-checkout",
        "../sibling",
        "~/models/foo",
        "C:\\Users\\bob\\model",
        "file:///tmp/model",
        "org/path/with/extra/slashes",
        "weird name with spaces/x",
    ],
)
def test_normalize_model_path_redacts_local(raw):
    assert normalize_model_path(raw) == "<local>"


def test_normalize_model_path_bare_alias_passes():
    assert normalize_model_path("qwen3.5-9b-4bit") == "qwen3.5-9b-4bit"
    assert normalize_model_path("hermes3-8b-4bit") == "hermes3-8b-4bit"


def test_normalize_model_path_empty():
    assert normalize_model_path("") == "<empty>"


def test_hash_flag_names_extracts_only_names():
    argv = ["--api-key", "sk-real-secret", "--port", "8000", "model-name"]
    assert hash_flag_names(argv) == ["api-key", "port"]


def test_hash_flag_names_handles_equals_form():
    argv = ["--api-key=sk-secret", "--port=8000"]
    result = hash_flag_names(argv)
    assert "api-key" in result
    assert "port" in result
    assert all("sk-secret" not in name for name in result)
    assert all("8000" not in name for name in result)


def test_hash_flag_names_value_with_equals():
    argv = ["--filter=key=value", "--header=X-Trace=abc123"]
    result = hash_flag_names(argv)
    assert result == ["filter", "header"]
    for needle in ("key", "value", "X-Trace", "abc123"):
        assert needle not in result


def test_hash_flag_names_short_flags():
    argv = ["-V", "-y", "--verbose"]
    assert hash_flag_names(argv) == ["V", "verbose", "y"]


def test_hash_flag_names_returns_sorted_unique():
    argv = ["--zebra", "--alpha", "--zebra", "--mike"]
    assert hash_flag_names(argv) == ["alpha", "mike", "zebra"]


def test_hash_flag_names_empty_and_non_strings():
    assert hash_flag_names([]) == []
    assert hash_flag_names(["not-a-flag", "another-positional"]) == []
    assert hash_flag_names([None, 42, "--real"]) == ["real"]


def test_fingerprint_traceback_is_deterministic():
    def trigger_and_fingerprint() -> str:
        try:
            raise ValueError("user secret leaked here")
        except ValueError as e:
            return fingerprint_traceback(e)

    fp1 = trigger_and_fingerprint()
    fp2 = trigger_and_fingerprint()

    assert fp1 == fp2
    assert len(fp1) == 16


def test_fingerprint_traceback_omits_message_text():
    try:
        raise RuntimeError("user secret leaked here in the message")
    except RuntimeError as e:
        fp = fingerprint_traceback(e)

    assert "user" not in fp
    assert "secret" not in fp
    assert "leaked" not in fp
    assert all(c in "0123456789abcdef" for c in fp)


def test_fingerprint_traceback_excludes_exception_module_path():
    err1 = type("CustomError", (Exception,), {"__module__": "pkg_a.sub"})
    err2 = type("CustomError", (Exception,), {"__module__": "pkg_b.deep.nested"})

    def trigger(cls) -> str:
        try:
            raise cls("x")
        except Exception as e:
            return fingerprint_traceback(e)

    assert trigger(err1) == trigger(err2)


def test_fingerprint_traceback_omits_local_paths():
    try:
        raise RuntimeError("x")
    except RuntimeError as e:
        fp = fingerprint_traceback(e)

    def site_a():
        raise ValueError("a")

    def site_b():
        raise ValueError("b")

    try:
        site_a()
    except ValueError as e:
        fp_a = fingerprint_traceback(e)
    try:
        site_b()
    except ValueError as e:
        fp_b = fingerprint_traceback(e)

    assert fp_a != fp_b
    for f in (fp, fp_a, fp_b):
        assert len(f) == 16


def test_platform_info_no_full_kernel_string():
    info = platform_info()
    assert isinstance(info["os_version"], str)
    assert info["os_version"].count(".") <= 1


def test_platform_info_python_version_short():
    info = platform_info()
    assert info["python_version"].count(".") == 1


def test_platform_info_memory_is_rounded_int():
    info = platform_info()
    assert isinstance(info["memory_gb"], int)
    assert info["memory_gb"] >= 0


def test_platform_info_no_unbounded_strings():
    info = platform_info()
    for key in ("os", "os_version", "arch", "chip", "python_version"):
        assert isinstance(info[key], str)
        assert len(info[key]) < 200, f"{key} suspiciously long: {info[key]!r}"
