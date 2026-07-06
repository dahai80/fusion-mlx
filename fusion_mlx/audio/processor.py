# SPDX-License-Identifier: Apache-2.0
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SAM_MODEL = "mlx-community/sam-audio-large-fp16"


@dataclass
class SeparationResult:
    target: np.ndarray
    residual: np.ndarray
    sample_rate: int
    peak_memory: float


class AudioProcessor:

    def __init__(
        self,
        model_name: str = DEFAULT_SAM_MODEL,
    ):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._loaded = False
        self.sample_rate = 44100

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from mlx_audio.sts import SAMAudio, SAMAudioProcessor

            self.model = SAMAudio.from_pretrained(self.model_name)
            self.processor = SAMAudioProcessor.from_pretrained(self.model_name)

            if hasattr(self.model, "sample_rate"):
                self.sample_rate = self.model.sample_rate

            self._loaded = True
            logger.info("Audio processor loaded: %s", self.model_name)
        except ImportError as e:
            logger.error("mlx-audio not installed: %s", e)
            raise ImportError(
                "mlx-audio is required. Install with: pip install mlx-audio"
            ) from e

    def separate(
        self,
        audio_path: str | Path,
        description: str = "speech",
        chunk_seconds: float | None = None,
    ) -> SeparationResult:
        if not self._loaded:
            self.load()

        audio_path = str(audio_path)

        try:
            batch = self.processor(
                descriptions=[description],
                audios=[audio_path],
            )

            if chunk_seconds:
                result = self.model.separate_long(
                    audios=batch.audios,
                    descriptions=batch.descriptions,
                    chunk_seconds=chunk_seconds,
                    overlap_seconds=chunk_seconds / 3,
                    anchor_ids=getattr(batch, "anchor_ids", None),
                    anchor_alignment=getattr(batch, "anchor_alignment", None),
                )
            else:
                result = self.model.separate(
                    audios=batch.audios,
                    descriptions=batch.descriptions,
                    sizes=getattr(batch, "sizes", None),
                    anchor_ids=getattr(batch, "anchor_ids", None),
                    anchor_alignment=getattr(batch, "anchor_alignment", None),
                )

            target = self._to_numpy(result.target[0])
            residual = self._to_numpy(result.residual[0])

            return SeparationResult(
                target=target,
                residual=residual,
                sample_rate=self.sample_rate,
                peak_memory=getattr(result, "peak_memory", 0.0),
            )
        except Exception as e:
            logger.error("Audio separation failed: %s", e)
            raise

    def _to_numpy(self, audio) -> np.ndarray:
        if hasattr(audio, "tolist"):
            return np.array(audio.tolist(), dtype=np.float32)
        return np.array(audio, dtype=np.float32)

    def save(
        self,
        audio: np.ndarray,
        path: str | Path,
        sample_rate: int | None = None,
    ) -> None:
        sr = sample_rate or self.sample_rate

        try:
            from mlx_audio.sts import save_audio

            save_audio(audio, str(path), sample_rate=sr)
        except ImportError:
            import scipy.io.wavfile as wav

            audio_int16 = (audio * 32767).astype(np.int16)
            wav.write(str(path), sr, audio_int16)

        logger.info("Audio saved to %s", path)

    def unload(self) -> None:
        self.model = None
        self.processor = None
        self._loaded = False
        logger.info("Audio processor unloaded")


def separate_voice(
    audio_path: str | Path,
    model_name: str = DEFAULT_SAM_MODEL,
    description: str = "speech",
) -> tuple[np.ndarray, np.ndarray]:
    processor = AudioProcessor(model_name)
    processor.load()
    result = processor.separate(audio_path, description=description)
    return result.target, result.residual
