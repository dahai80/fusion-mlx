# SPDX-License-Identifier: Apache-2.0
"""Audio support for fusion-mlx using mlx-audio.

Provides:
- STT (Speech-to-Text): Whisper, Parakeet
- TTS (Text-to-Speech): Kokoro, Chatterbox, VibeVoice, VoxCPM
- Audio Processing: SAM-Audio (voice separation)
"""

from .processor import AudioProcessor, separate_voice
from .stt import STTEngine, transcribe_audio
from .tts import TTSEngine, generate_speech

__all__ = [
    "STTEngine",
    "transcribe_audio",
    "TTSEngine",
    "generate_speech",
    "AudioProcessor",
    "separate_voice",
]
