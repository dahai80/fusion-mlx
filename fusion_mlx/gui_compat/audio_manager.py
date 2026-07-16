"""Audio transcription using MLX Whisper integrated with existing model manager."""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import mlx_whisper

logger = logging.getLogger(__name__)


class WhisperError(Exception):
    """Exception raised when Whisper transcription fails."""
    pass


class AudioManager:
    """Manages audio transcription using MLX Whisper."""
    
    def __init__(self):
        self.default_model = "mlx-community/whisper-tiny"
        self.supported_formats = {'.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg', '.webm'}
        logger.info("AudioManager initialized")
    
    def transcribe_audio(
        self,
        audio_path: str,
        model: str = None,
        language: Optional[str] = None,
        word_timestamps: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using MLX Whisper.
        
        Args:
            audio_path: Path to audio file
            model: Model name or HuggingFace ID
            language: Language code (e.g., 'en', 'es') or None for auto-detect
            word_timestamps: Whether to include word-level timestamps
            
        Returns:
            Dictionary with transcription results
            
        Raises:
            WhisperError: If transcription fails
        """
        try:
            model = model or self.default_model
            
            logger.info(f"Transcribing audio with model {model}")
            
            # Validate audio file exists
            if not os.path.exists(audio_path):
                raise WhisperError(f"Audio file not found: {audio_path}")
            
            # Validate file format
            if not self.validate_audio_file(audio_path):
                suffix = Path(audio_path).suffix
                raise WhisperError(f"Unsupported audio format: {suffix}")
            
            # Perform transcription
            result = mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=model,
                language=language,
                word_timestamps=word_timestamps,
            )
            
            logger.info(f"Transcription completed successfully")
            
            return {
                "text": result["text"],
                "language": result.get("language"),
                "segments": result.get("segments", []) if word_timestamps else [],
                "model": model
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise WhisperError(f"Transcription failed: {str(e)}") from e
    
    def validate_audio_file(self, file_path: str) -> bool:
        """Validate audio file format."""
        return Path(file_path).suffix.lower() in self.supported_formats
    
    def get_available_models(self) -> list[str]:
        """Get list of available Whisper models."""
        return [
            "mlx-community/whisper-tiny",
            "mlx-community/whisper-base", 
            "mlx-community/whisper-small",
            "mlx-community/whisper-medium",
            "mlx-community/whisper-large-v2",
            "mlx-community/whisper-large-v3"
        ]
    
    def get_supported_formats(self) -> list[str]:
        """Get list of supported audio formats."""
        return list(self.supported_formats)