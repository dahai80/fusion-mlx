# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for OpenAI-compatible audio API.

These models define the request and response schemas for:
- Audio transcription (speech-to-text)
- Audio speech synthesis (text-to-speech)
"""

from pydantic import BaseModel, Field, field_validator, model_validator

_ALLOWED_AUDIO_FORMATS = ("wav", "pcm", "flac", "ogg", "opus", "mp3")


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
    streaming_interval: float | None = Field(None, gt=0.0, le=10.0)

    @model_validator(mode="before")
    @classmethod
    def _fold_format_alias(cls, values):
        # R11-B-F2 (#724): legacy ``format`` field folds into
        # ``response_format``. Explicit ``response_format`` wins;
        # ``format=None`` is treated as unset so the Pydantic default
        # still applies. Non-string values fall through to the
        # response_format type validator which surfaces the 400 on the
        # canonical field name (loc=response_format, not format).
        if not isinstance(values, dict):
            return values
        fmt = values.get("format")
        rf_present = (
            "response_format" in values and values["response_format"] is not None
        )
        if fmt is not None and not rf_present:
            values["response_format"] = fmt
        values.pop("format", None)
        return values

    @field_validator("response_format", mode="before")
    @classmethod
    def _validate_audio_response_format(cls, v):
        if v is None:
            return v
        if not isinstance(v, str):
            raise ValueError("response_format must be a string")
        if v not in _ALLOWED_AUDIO_FORMATS:
            raise ValueError(
                f"response_format must be one of "
                f"{', '.join(_ALLOWED_AUDIO_FORMATS)} (got {v!r})"
            )
        return v


class AudioProcessRequest(BaseModel):
    """Request model for audio processing (speech enhancement / STS).

    Used by POST /v1/audio/process — the audio file is submitted as a
    multipart upload alongside this model field.
    """

    model: str
