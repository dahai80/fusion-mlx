# SPDX-License-Identifier: Apache-2.0
# Server-side reference-voice bundles for cloning-capable TTS models (#1383).
#
# A bundle is a directory under the voice library (``~/.fusion-mlx/voices``,
# overridable with FUSION_VOICE_DIR) holding a reference clip plus its
# transcript. The speech route resolves ``voice`` to the pair so clients do
# not have to host and base64-encode their own reference audio.
#
# The same tree also holds fine-tune packs (#1336), which use
# ``voice.safetensors`` + ``manifest.json`` instead; the two layouts do not
# collide because a bundle requires a reference clip.

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

REF_AUDIO_STEMS = ("ref", "reference", "sample")
AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
REF_TEXT_NAME = "ref.txt"
REF_TEXT_ALIASES = ("ref.txt", "reference.txt")
MAX_REF_TEXT_CHARS = 4000
MIN_REF_AUDIO_BYTES = 1024


@dataclass(frozen=True)
class VoiceBundle:
    name: str
    directory: Path
    audio_path: Path
    ref_text: str


def _base_dir() -> Path:
    from fusion_mlx.training.voice import VOICE_BASE_DIR

    return Path(VOICE_BASE_DIR)


def voice_bundle_dir() -> Path:
    return _base_dir()


def bundle_to_dict(bundle: VoiceBundle) -> dict:
    try:
        created_at = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(bundle.audio_path.stat().st_mtime)
        )
    except OSError:
        created_at = None
    return {
        "name": bundle.name,
        "ref_audio": bundle.audio_path.name,
        "ref_text_chars": len(bundle.ref_text),
        "created_at": created_at,
    }


def _valid_name(name: str) -> bool:
    if not name or not isinstance(name, str):
        return False
    if len(name) > 64:
        return False
    if "/" in name or "\\" in name or "\x00" in name:
        return False
    if name.startswith("."):
        return False
    return name not in {".", ".."}


def _find_audio(directory: Path) -> Path | None:
    for stem in REF_AUDIO_STEMS:
        for suffix in AUDIO_SUFFIXES:
            candidate = directory / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def _find_ref_text(directory: Path) -> Path | None:
    for filename in REF_TEXT_ALIASES:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def resolve_voice_bundle(name: str) -> VoiceBundle | None:
    if not _valid_name(name):
        return None
    directory = _base_dir() / name
    if not directory.is_dir():
        return None
    audio_path = _find_audio(directory)
    if audio_path is None:
        return None
    text_path = _find_ref_text(directory)
    if text_path is None:
        logger.warning(
            "voice bundle %r has %s but no %s; ignored (the transcript is "
            "required for in-context-cloning models)",
            name,
            audio_path.name,
            REF_TEXT_NAME,
        )
        return None
    try:
        ref_text = text_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning("voice bundle %r: cannot read %s: %s", name, text_path.name, exc)
        return None
    if not ref_text:
        logger.warning("voice bundle %r: %s is empty; ignored", name, text_path.name)
        return None
    try:
        size = audio_path.stat().st_size
    except OSError:
        return None
    if size < MIN_REF_AUDIO_BYTES:
        logger.warning(
            "voice bundle %r: %s is %d bytes (minimum %d); ignored",
            name,
            audio_path.name,
            size,
            MIN_REF_AUDIO_BYTES,
        )
        return None
    return VoiceBundle(
        name=name,
        directory=directory,
        audio_path=audio_path,
        ref_text=ref_text[:MAX_REF_TEXT_CHARS],
    )


def list_voice_bundles() -> list[dict]:
    base = _base_dir()
    if not base.is_dir():
        return []
    bundles: list[dict] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        bundle = resolve_voice_bundle(entry.name)
        if bundle is None:
            continue
        bundles.append(bundle_to_dict(bundle))
    return bundles


def detect_audio_suffix(audio_bytes: bytes) -> str:
    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        return ".mp3"
    if audio_bytes[:4] == b"fLaC":
        return ".flac"
    if audio_bytes[:4] == b"OggS":
        return ".ogg"
    if len(audio_bytes) >= 12 and audio_bytes[4:8] == b"ftyp":
        return ".m4a"
    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return ".wav"
    return ""


def write_voice_bundle(
    name: str, audio_bytes: bytes, ref_text: str, *, overwrite: bool = False
) -> VoiceBundle:
    if not _valid_name(name):
        raise ValueError(
            f"invalid voice bundle name {name!r}: use 1-64 characters, no "
            "path separators, no leading dot"
        )
    text = (ref_text or "").strip()
    if not text:
        raise ValueError("ref_text is required (transcript of the reference clip)")
    if len(text) > MAX_REF_TEXT_CHARS:
        raise ValueError(
            f"ref_text is {len(text)} characters (maximum {MAX_REF_TEXT_CHARS})"
        )
    if len(audio_bytes) < MIN_REF_AUDIO_BYTES:
        raise ValueError(
            f"reference clip is {len(audio_bytes)} bytes (minimum "
            f"{MIN_REF_AUDIO_BYTES})"
        )
    suffix = detect_audio_suffix(audio_bytes)
    if not suffix:
        raise ValueError(
            "reference clip is not a recognised audio format "
            f"(accepted: {', '.join(AUDIO_SUFFIXES)})"
        )
    directory = _base_dir() / name
    if directory.is_dir() and not overwrite and _find_audio(directory) is not None:
        raise FileExistsError(f"voice bundle {name!r} already exists")
    _clean_audio_files(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"ref{suffix}").write_bytes(audio_bytes)
    (directory / REF_TEXT_NAME).write_text(text, encoding="utf-8")
    logger.info(
        "voice bundle %r written: %s (%d bytes), ref_text=%d chars",
        name,
        directory / f"ref{suffix}",
        len(audio_bytes),
        len(text),
    )
    bundle = resolve_voice_bundle(name)
    if bundle is None:
        raise ValueError(f"voice bundle {name!r} could not be read back after writing")
    return bundle


def _clean_audio_files(directory: Path) -> None:
    if not directory.is_dir():
        return
    for stem in REF_AUDIO_STEMS:
        for suffix in AUDIO_SUFFIXES:
            stale = directory / f"{stem}{suffix}"
            if stale.is_file():
                try:
                    os.unlink(stale)
                except OSError as exc:
                    logger.warning("cannot remove stale %s: %s", stale, exc)


def delete_voice_bundle(name: str) -> bool:
    if not _valid_name(name):
        return False
    directory = _base_dir() / name
    if not directory.is_dir():
        return False
    if resolve_voice_bundle(name) is None:
        # Not a bundle: never delete a fine-tune pack or an unrelated dir.
        return False
    removed = []
    for stem in REF_AUDIO_STEMS:
        for suffix in AUDIO_SUFFIXES:
            candidate = directory / f"{stem}{suffix}"
            if candidate.is_file():
                candidate.unlink()
                removed.append(candidate.name)
    for filename in REF_TEXT_ALIASES:
        candidate = directory / filename
        if candidate.is_file():
            candidate.unlink()
            removed.append(candidate.name)
    try:
        directory.rmdir()
    except OSError:
        pass
    logger.info("voice bundle %r deleted (%s)", name, ", ".join(removed) or "no files")
    return True
