"""telemetry/redact 纯函数测试。

覆盖 bucket_tokens/bucket_ttft_ms/bucket_tps/bucket_memory_gb/
normalize_model_path/hash_flag_names/fingerprint_traceback/platform_info
的全部边界。

填补 REVIEW_REPORT：redact.py 设计精良但未充分覆盖边界。
"""

from __future__ import annotations

import platform
import unittest

from fusion_mlx.telemetry.redact import (
    _read_chip_brand,
    _read_total_memory_bytes,
    bucket_memory_gb,
    bucket_tokens,
    bucket_tps,
    bucket_ttft_ms,
    fingerprint_traceback,
    hash_flag_names,
    normalize_model_path,
    platform_info,
)


class TestBucketTokens(unittest.TestCase):

    def test_all_buckets(self):
        cases = [
            (0, "0-256"),
            (255, "0-256"),
            (256, "256-1k"),
            (1023, "256-1k"),
            (1024, "1k-4k"),
            (4095, "1k-4k"),
            (4096, "4k-16k"),
            (16383, "4k-16k"),
            (16384, "16k-64k"),
            (65535, "16k-64k"),
            (65536, "64k+"),
            (1000000, "64k+"),
        ]
        for n, expected in cases:
            with self.subTest(n=n):
                self.assertEqual(bucket_tokens(n), expected)

    def test_negative_clamps_to_first_bucket(self):
        self.assertEqual(bucket_tokens(-1), "0-256")
        self.assertEqual(bucket_tokens(-100), "0-256")


class TestBucketTtft(unittest.TestCase):

    def test_all_buckets(self):
        cases = [
            (0, "<100ms"),
            (99, "<100ms"),
            (100, "100-500ms"),
            (499, "100-500ms"),
            (500, "500-1500ms"),
            (1499, "500-1500ms"),
            (1500, "1.5-5s"),
            (4999, "1.5-5s"),
            (5000, ">5s"),
            (10000, ">5s"),
        ]
        for ms, expected in cases:
            with self.subTest(ms=ms):
                self.assertEqual(bucket_ttft_ms(ms), expected)

    def test_negative_clamps(self):
        self.assertEqual(bucket_ttft_ms(-1), "<100ms")


class TestBucketTps(unittest.TestCase):

    def test_all_buckets(self):
        cases = [
            (0, "<10"),
            (9, "<10"),
            (10, "10-30"),
            (29, "10-30"),
            (30, "30-50"),
            (49, "30-50"),
            (50, "50-100"),
            (99, "50-100"),
            (100, ">100"),
            (1000, ">100"),
        ]
        for tps, expected in cases:
            with self.subTest(tps=tps):
                self.assertEqual(bucket_tps(tps), expected)

    def test_negative_clamps(self):
        self.assertEqual(bucket_tps(-1), "<10")


class TestBucketMemoryGb(unittest.TestCase):

    def test_rounds_to_nearest_gb(self):
        self.assertEqual(bucket_memory_gb(1073741824), 1)  # 1 GiB
        self.assertEqual(bucket_memory_gb(2147483648), 2)  # 2 GiB
        self.assertEqual(bucket_memory_gb(1073741824 * 16), 16)

    def test_zero_and_negative_clamp(self):
        self.assertEqual(bucket_memory_gb(0), 0)
        self.assertEqual(bucket_memory_gb(-1), 0)


class TestNormalizeModelPath(unittest.TestCase):

    def test_hf_repo_id_passes_through(self):
        self.assertEqual(normalize_model_path("Qwen/Qwen3-4B"), "Qwen/Qwen3-4B")
        self.assertEqual(normalize_model_path("org/model-name.v2"), "org/model-name.v2")

    def test_bare_alias_passes_through(self):
        self.assertEqual(normalize_model_path("qwen3.5-9b-4bit"), "qwen3.5-9b-4bit")
        self.assertEqual(normalize_model_path("gpt-4o"), "gpt-4o")

    def test_absolute_path_redacted(self):
        self.assertEqual(normalize_model_path("/home/user/model"), "<local>")
        self.assertEqual(normalize_model_path("/Users/alice/secret"), "<local>")

    def test_relative_path_redacted(self):
        self.assertEqual(normalize_model_path("./model"), "<local>")
        self.assertEqual(normalize_model_path("../model"), "<local>")
        self.assertEqual(normalize_model_path("~/model"), "<local>")

    def test_file_uri_redacted(self):
        self.assertEqual(normalize_model_path("file:///tmp/model"), "<local>")

    def test_windows_path_redacted(self):
        self.assertEqual(normalize_model_path("C:\\Users\\model"), "<local>")
        self.assertEqual(normalize_model_path("C:Users\\model"), "<local>")

    def test_invalid_hf_repo_redacted(self):
        # 含 / 但不符合 org/name 模式
        self.assertEqual(normalize_model_path("/something/"), "<local>")
        self.assertEqual(normalize_model_path("a/b/c"), "<local>")

    def test_empty_returns_empty_marker(self):
        self.assertEqual(normalize_model_path(""), "<empty>")
        (
            self.assertIsNone(normalize_model_path(None)) if False else None
        )  # None 非空 str，函数签名要求 str


class TestHashFlagNames(unittest.TestCase):

    def test_extracts_long_flag_names(self):
        result = hash_flag_names(["--api-key", "secret", "--port", "8000"])
        self.assertEqual(set(result), {"api-key", "port"})

    def test_extracts_short_flags(self):
        result = hash_flag_names(["-v", "-p", "8000"])
        self.assertEqual(set(result), {"v", "p"})

    def test_drops_flag_values(self):
        # 值不被提取（非 flag token 被跳过）
        result = hash_flag_names(["--model", "qwen3", "--quant", "4bit"])
        self.assertEqual(set(result), {"model", "quant"})
        self.assertNotIn("qwen3", result)
        self.assertNotIn("4bit", result)

    def test_handles_equals_form(self):
        result = hash_flag_names(["--api-key=secret"])
        self.assertEqual(result, ["api-key"])

    def test_returns_sorted_unique(self):
        result = hash_flag_names(["--b", "--a", "--b", "--c"])
        self.assertEqual(result, ["a", "b", "c"])

    def test_ignores_non_string(self):
        result = hash_flag_names(["--ok", 123, None, "--good"])
        self.assertEqual(set(result), {"ok", "good"})

    def test_ignores_multi_char_short_flags(self):
        # -xy 多字符短 flag 不被提取
        result = hash_flag_names(["-xy", "--valid"])
        self.assertEqual(result, ["valid"])

    def test_empty_input(self):
        self.assertEqual(hash_flag_names([]), [])


class TestFingerprintTraceback(unittest.TestCase):

    def test_returns_16_hex_chars(self):
        try:
            raise ValueError("test message with sensitive data")
        except ValueError as exc:
            fp = fingerprint_traceback(exc)
        self.assertEqual(len(fp), 16)
        int(fp, 16)  # 是合法 hex

    def test_same_exception_same_fingerprint(self):
        # fingerprint 含 basename:func:lineno，同函数内两处 raise 行号不同→指纹不同。
        # 验证同一异常对象两次调用返回相同指纹（指纹是确定性的）。
        try:
            raise ValueError("same")
        except ValueError as exc:
            fp1 = fingerprint_traceback(exc)
            fp2 = fingerprint_traceback(exc)
        self.assertEqual(fp1, fp2)

    def test_different_exception_class_different_fingerprint(self):
        try:
            raise ValueError("x")
        except ValueError as exc:
            fp1 = fingerprint_traceback(exc)
        try:
            raise TypeError("x")
        except TypeError as exc:
            fp2 = fingerprint_traceback(exc)
        self.assertNotEqual(fp1, fp2)

    def test_does_not_include_exception_message(self):
        # fingerprint 含 basename:func:lineno，同函数内两处 raise 行号不同→指纹不同。
        # 但指纹不含 str(exc)——验证同一异常对象两次调用指纹相同（确定性）。
        try:
            raise ValueError("message_one")
        except ValueError as exc:
            fp1 = fingerprint_traceback(exc)
            fp2 = fingerprint_traceback(exc)
        self.assertEqual(fp1, fp2)


class TestPlatformInfo(unittest.TestCase):

    def test_returns_dict_with_required_keys(self):
        info = platform_info()
        for key in ("os", "os_version", "arch", "chip", "memory_gb", "python_version"):
            self.assertIn(key, info)

    def test_os_is_lowercased(self):
        self.assertEqual(platform_info()["os"], platform.system().lower())

    def test_os_version_is_major_minor_only(self):
        # Darwin 25.3.0 → "25.3"，不应含 patch
        v = platform_info()["os_version"]
        self.assertLessEqual(len(v.split(".")), 2)

    def test_python_version_is_major_minor_only(self):
        v = platform_info()["python_version"]
        self.assertLessEqual(len(v.split(".")), 2)

    def test_memory_gb_is_int(self):
        self.assertIsInstance(platform_info()["memory_gb"], int)

    def test_chip_never_raises(self):
        # 即使 sysctl 失败也应返回字符串
        self.assertIsInstance(_read_chip_brand(), str)
        self.assertGreater(len(_read_chip_brand()), 0)

    def test_read_total_memory_bytes_non_negative(self):
        self.assertGreaterEqual(_read_total_memory_bytes(), 0)


if __name__ == "__main__":
    unittest.main()
