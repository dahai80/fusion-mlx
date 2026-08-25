from fusion_mlx.api import _url_safety as _us
from fusion_mlx.api._url_safety import is_safe_local_path, is_safe_url


class TestIsSafeUrl:
    def test_public_url_allowed(self):
        assert is_safe_url("https://example.com/image.png")

    def test_localhost_blocked(self):
        assert not is_safe_url("http://localhost:8080/secret")

    def test_127_0_0_1_blocked(self):
        assert not is_safe_url("http://127.0.0.1/admin")

    def test_private_ip_blocked(self):
        assert not is_safe_url("http://192.168.1.1/router")

    def test_metadata_endpoint_blocked(self):
        assert not is_safe_url("http://metadata.google.internal/computeMetadata/v1/")

    def test_empty_url_blocked(self):
        assert not is_safe_url("")

    def test_no_hostname_blocked(self):
        assert not is_safe_url("http:///path")


class TestIsSafeLocalPath:
    def test_allowed_model_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            _us,
            "_ALLOWED_READ_DIRS",
            [str(tmp_path)],
        )
        test_file = tmp_path / "model.safetensors"
        test_file.write_text("test")
        assert is_safe_local_path(str(test_file))

    def test_traversal_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            _us,
            "_ALLOWED_READ_DIRS",
            [str(tmp_path)],
        )
        assert not is_safe_local_path("/etc/passwd")

    def test_path_traversal_dotdot_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            _us,
            "_ALLOWED_READ_DIRS",
            [str(tmp_path / "safe")],
        )
        (tmp_path / "safe").mkdir()
        traversal = str(tmp_path / "safe" / ".." / ".." / "etc" / "passwd")
        assert not is_safe_local_path(traversal)

    def test_null_byte_blocked(self):
        assert not is_safe_local_path("/tmp/test\0/etc/passwd")

    def test_empty_path_blocked(self):
        assert not is_safe_local_path("")

    def test_file_uri_scheme(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            _us,
            "_ALLOWED_READ_DIRS",
            [str(tmp_path)],
        )
        test_file = tmp_path / "model.bin"
        test_file.write_text("data")
        assert is_safe_local_path(f"file://{test_file}")

    def test_nonexistent_path_in_allowed_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            _us,
            "_ALLOWED_READ_DIRS",
            [str(tmp_path)],
        )
        assert is_safe_local_path(str(tmp_path / "does_not_exist.bin"))


class TestAllowedReadDirsEnv:
    # Issue #633: FUSION_MLX_ALLOWED_READ_DIRS extends the allow-list.

    def test_env_extends_allowed_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_us, "_ALLOWED_READ_DIRS", [])
        monkeypatch.setenv("FUSION_MLX_ALLOWED_READ_DIRS", str(tmp_path))
        target = tmp_path / "scene_1_last.png"
        target.write_text("frame")
        assert is_safe_local_path(str(target))

    def test_env_colon_separated_multiple(self, tmp_path, monkeypatch):
        d1 = tmp_path / "out1"
        d2 = tmp_path / "out2"
        d1.mkdir()
        d2.mkdir()
        monkeypatch.setattr(_us, "_ALLOWED_READ_DIRS", [])
        monkeypatch.setenv("FUSION_MLX_ALLOWED_READ_DIRS", f"{d1}:{d2}")
        assert is_safe_local_path(str(d1 / "a.png"))
        assert is_safe_local_path(str(d2 / "b.png"))

    def test_env_empty_keeps_base_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_us, "_ALLOWED_READ_DIRS", [str(tmp_path)])
        monkeypatch.delenv("FUSION_MLX_ALLOWED_READ_DIRS", raising=False)
        assert is_safe_local_path(str(tmp_path / "x.bin"))

    def test_env_path_outside_blocked(self, tmp_path, monkeypatch):
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        monkeypatch.setattr(_us, "_ALLOWED_READ_DIRS", [])
        monkeypatch.setenv("FUSION_MLX_ALLOWED_READ_DIRS", str(allowed))
        assert not is_safe_local_path("/etc/passwd")

    def test_get_allowed_read_dirs_no_dup(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_us, "_ALLOWED_READ_DIRS", [str(tmp_path)])
        monkeypatch.setenv("FUSION_MLX_ALLOWED_READ_DIRS", f"{tmp_path}:/other")
        dirs = _us.get_allowed_read_dirs()
        assert dirs.count(str(tmp_path)) == 1
        assert "/other" in dirs
