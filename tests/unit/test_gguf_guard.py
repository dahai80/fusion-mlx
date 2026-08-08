import pytest

from fusion_mlx.engine.gguf_guard import (
    GGUFLoadError,
    assert_not_gguf,
    is_gguf_model,
)


def test_direct_gguf_file_detected(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\x00")
    assert is_gguf_model(str(f)) is True


def test_gguf_file_guard_raises(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"\x00")
    with pytest.raises(GGUFLoadError) as exc:
        assert_not_gguf(str(f), engine_kind="LLM")
    assert "GGUF" in str(exc.value)
    assert "mlx-community" in str(exc.value)


def test_mlx_dir_not_flagged(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"\x00")
    assert is_gguf_model(str(tmp_path)) is False
    assert_not_gguf(str(tmp_path), engine_kind="LLM")


def test_gguf_only_dir_flagged(tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"\x00")
    assert is_gguf_model(str(tmp_path)) is True


def test_gguf_dir_with_config_not_flagged(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.gguf").write_bytes(b"\x00")
    assert is_gguf_model(str(tmp_path)) is False


def test_empty_name_not_flagged():
    assert is_gguf_model("") is False
    assert_not_gguf("", engine_kind="LLM")


def test_nonexistent_path_not_flagged(tmp_path):
    assert is_gguf_model(str(tmp_path / "nope.gguf")) is False
    assert_not_gguf(str(tmp_path / "nope.gguf"), engine_kind="LLM")


def test_hf_repo_id_not_flagged():
    assert is_gguf_model("mlx-community/Qwen2.5-7B-Instruct-4bit") is False
    assert is_gguf_model("Qwen2.5-7B-Instruct") is False


def test_error_message_mentions_convert_endpoint(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"\x00")
    with pytest.raises(GGUFLoadError) as exc:
        assert_not_gguf(str(f))
    assert "/v1/convert" in str(exc.value)


def test_guard_no_op_for_normal_path(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "w.safetensors").write_bytes(b"\x00")
    assert_not_gguf(str(tmp_path), engine_kind="VLM")
