# SPDX-License-Identifier: Apache-2.0
import io
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TTS_MODEL = "mlx-community/Kokoro-82M-bf16"

KOKORO_VOICES = [
    "af_heart",
    "af_bella",
    "af_nicole",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
]

CHATTERBOX_VOICES = ["default"]


class UnsupportedAudioFormatError(Exception):

    def __init__(
        self,
        requested: str,
        supported: list[str],
        hint: str | None = None,
    ):
        self.requested = requested
        self.supported = supported
        self.hint = hint
        msg = (
            f"response_format={requested!r} is not supported by this "
            f"build. Supported formats: {', '.join(supported)}."
        )
        if hint:
            msg = f"{msg} {hint}"
        super().__init__(msg)


def _list_snapshot_voices(model_name: str) -> list[str]:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return []

    try:
        from .registry import resolve_audio_alias
    except ImportError:
        resolve_audio_alias = None

    hf_id = model_name
    if resolve_audio_alias is not None:
        entry = resolve_audio_alias(model_name)
        if entry is not None:
            hf_id = entry.hf_id

    if "/" not in hf_id:
        return []

    cached = try_to_load_from_cache(repo_id=hf_id, filename="config.json")
    if not cached or not isinstance(cached, str):
        return []

    voices_dir = Path(cached).parent / "voices"
    if not voices_dir.is_dir():
        return []

    return sorted(p.stem for p in voices_dir.glob("*.safetensors"))


@dataclass
class AudioOutput:
    audio: np.ndarray
    sample_rate: int
    duration: float


class TTSEngine:

    def __init__(
        self,
        model_name: str = DEFAULT_TTS_MODEL,
    ):
        self.model_name = model_name
        self.model = None
        self._loaded = False
        self._model_family = self._detect_family(model_name)

    def _detect_family(self, model_name: str) -> str:
        name_lower = model_name.lower()
        if "kokoro" in name_lower:
            return "kokoro"
        elif "chatterbox" in name_lower:
            return "chatterbox"
        elif "vibevoice" in name_lower:
            return "vibevoice"
        elif "voxcpm" in name_lower:
            return "voxcpm"
        elif "csm" in name_lower:
            return "csm"
        elif "cosyvoice" in name_lower:
            return "cosyvoice"
        else:
            return "kokoro"

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from mlx_audio.tts.generate import load_model

            self.model = load_model(self.model_name)
            self._loaded = True
            logger.info(
                "TTS model loaded: %s (family: %s)",
                self.model_name,
                self._model_family,
            )
        except ImportError as e:
            logger.error("mlx-audio not installed: %s", e)
            raise ImportError(
                "mlx-audio is required for TTS. Install with: pip install mlx-audio"
            ) from e

    def generate(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = "a",
    ) -> AudioOutput:
        if not self._loaded:
            self.load()

        try:
            import mlx.core as mx

            audio_chunks = []
            sample_rate = 24000

            for result in self.model.generate(
                text=text,
                voice=voice,
                speed=speed,
                lang_code=lang_code,
            ):
                audio_data = result.audio
                if hasattr(result, "sample_rate"):
                    sample_rate = result.sample_rate

                if isinstance(audio_data, mx.array) or hasattr(audio_data, "tolist"):
                    audio_np = np.array(audio_data.tolist(), dtype=np.float32)
                else:
                    audio_np = np.array(audio_data, dtype=np.float32)

                audio_chunks.append(audio_np)

            if not audio_chunks:
                raise RuntimeError("No audio generated")

            full_audio = (
                np.concatenate(audio_chunks)
                if len(audio_chunks) > 1
                else audio_chunks[0]
            )
            duration = len(full_audio) / sample_rate

            return AudioOutput(
                audio=full_audio,
                sample_rate=sample_rate,
                duration=duration,
            )
        except Exception as e:
            logger.error("TTS generation failed: %s", e)
            raise

    def stream_generate(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
    ) -> Iterator[AudioOutput]:
        if not self._loaded:
            self.load()

        sample_rate = 24000

        for result in self.model.generate(
            text=text,
            voice=voice,
            speed=speed,
        ):
            audio_data = result.audio
            if hasattr(result, "sample_rate"):
                sample_rate = result.sample_rate

            if hasattr(audio_data, "tolist"):
                audio_np = np.array(audio_data.tolist(), dtype=np.float32)
            else:
                audio_np = np.array(audio_data, dtype=np.float32)

            yield AudioOutput(
                audio=audio_np,
                sample_rate=sample_rate,
                duration=len(audio_np) / sample_rate,
            )

    def save(
        self,
        audio: AudioOutput,
        path: str | Path,
        format: str = "wav",
    ) -> None:
        try:
            from mlx_audio.tts import save_audio

            save_audio(audio.audio, str(path), sample_rate=audio.sample_rate)
            logger.info("Audio saved to %s", path)
        except ImportError:
            import scipy.io.wavfile as wav

            audio_int16 = (audio.audio * 32767).astype(np.int16)
            wav.write(str(path), audio.sample_rate, audio_int16)
            logger.info("Audio saved to %s (scipy fallback)", path)

    def to_bytes(
        self,
        audio: AudioOutput,
        format: str = "wav",
    ) -> bytes:
        fmt = (format or "wav").lower()
        audio_int16 = (np.clip(audio.audio, -1.0, 1.0) * 32767).astype(np.int16)

        if fmt == "wav":
            import scipy.io.wavfile as wav

            buffer = io.BytesIO()
            wav.write(buffer, audio.sample_rate, audio_int16)
            return buffer.getvalue()

        if fmt == "pcm":
            return audio_int16.tobytes()

        try:
            import soundfile as sf
        except ImportError as e:
            raise UnsupportedAudioFormatError(
                requested=fmt,
                supported=["wav", "pcm"],
                hint="Install with: pip install 'fusion-mlx[audio]'",
            ) from e

        soundfile_targets: dict[str, tuple[str, str | None]] = {
            "flac": ("FLAC", None),
            "ogg": ("OGG", "VORBIS"),
            "opus": ("OGG", "OPUS"),
            "mp3": ("MP3", None),
        }
        target = soundfile_targets.get(fmt)
        if target is None:
            raise UnsupportedAudioFormatError(
                requested=fmt,
                supported=sorted(["wav", "pcm", *soundfile_targets.keys()]),
            )

        container, subtype = target
        buffer = io.BytesIO()
        try:
            sf.write(
                buffer,
                audio_int16,
                audio.sample_rate,
                format=container,
                subtype=subtype,
            )
        except Exception as e:
            raise UnsupportedAudioFormatError(
                requested=fmt,
                supported=sorted(["wav", "pcm", *soundfile_targets.keys()]),
                hint=(
                    f"Encoder for {fmt!r} is not available in this "
                    f"libsndfile build ({e}). Upgrade libsndfile to "
                    "the latest release, or request a supported format."
                ),
            ) from e
        return buffer.getvalue()

    def get_voices(self) -> list:
        snapshot = _list_snapshot_voices(self.model_name)
        if snapshot:
            return snapshot
        if self._model_family == "kokoro":
            return KOKORO_VOICES
        elif self._model_family == "chatterbox":
            return CHATTERBOX_VOICES
        else:
            return ["default"]

    def unload(self) -> None:
        self.model = None
        self._loaded = False
        logger.info("TTS model unloaded")


def generate_speech(
    text: str,
    model_name: str = DEFAULT_TTS_MODEL,
    voice: str = "af_heart",
    speed: float = 1.0,
) -> AudioOutput:
    engine = TTSEngine(model_name)
    engine.load()
    return engine.generate(text, voice=voice, speed=speed)
