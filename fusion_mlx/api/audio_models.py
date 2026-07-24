# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for OpenAI-compatible audio API.

These models define the request and response schemas for:
- Audio transcription (speech-to-text)
- Audio speech synthesis (text-to-speech)
"""

from pydantic import BaseModel, Field


class AudioTranscriptionRequest(BaseModel):
    """OpenAI-compatible audio transcription request."""

    model: str
    language: str | None = None
    prompt: str | None = None
    response_format: str | None = "json"
    temperature: float | None = Field(0.0, ge=0.0, le=2.0)


class AudioTranscriptionResponse(BaseModel):
    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[dict] | None = None


class AudioSpeechRequest(BaseModel):
    model: str
    input: str
    voice: str | None = None
    instructions: str | None = None
    speed: float | None = Field(1.0, ge=0.25, le=4.0)
    response_format: str | None = "wav"
    ref_audio: str | None = None
    ref_text: str | None = None
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_k: int | None = Field(None, ge=0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(None, ge=0.0)
    max_tokens: int | None = Field(None, ge=1)
    stream: bool | None = False
    streaming_interval: float | None = Field(None, ge=0.1, le=10.0)


class AudioProcessRequest(BaseModel):
    """Request model for audio processing (speech enhancement / STS).

    Used by POST /v1/audio/process — the audio file is submitted as a
    multipart upload alongside this model field.
    """

    model: str
