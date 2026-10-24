# Issue #1377: the voice fine-tune trainer imports soundfile and librosa, but
# neither was declared in the [audio] extra, so a fresh `pip install -e
# ".[audio]"` failed at job time instead of at install time.

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _audio_extra():
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]["audio"]


def test_audio_extra_declares_voice_training_deps():
    names = [req.split(">")[0].split("=")[0].strip() for req in _audio_extra()]
    assert "soundfile" in names
    assert "librosa" in names
