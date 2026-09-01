import logging
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .convert_models import _get_allowed_output_prefixes

logger = logging.getLogger(__name__)


def _validate_output_path(v: str | None) -> str | None:
    if v is None:
        return v
    resolved = Path(v).resolve()
    for prefix in _get_allowed_output_prefixes():
        try:
            if resolved.is_relative_to(prefix.resolve()):
                return str(resolved)
        except Exception:
            pass
    logger.warning("watermark output_path rejected (outside allowed dirs): %s", v)
    raise ValueError(
        "output_path must be within allowed model directories "
        "(~/.fusion-mlx/models, CWD, or HF cache)"
    )


class WatermarkEmbedRequest(BaseModel):
    model: str = Field(
        ..., description="HF repo (org/name), model alias, or local model path"
    )
    payload: dict = Field(..., description="JSON-serializable dict to embed")
    secret: str = Field(..., description="Signing/seed secret (FMH_WATERMARK_SECRET)")
    layers: list[str] | None = Field(
        None, description="Glob patterns restricting which weight tensors to watermark"
    )
    bits_per_weight: int = Field(1, ge=1, le=3, description="LSB bits per carrier weight")
    in_place: bool = Field(False, description="Mutate safetensors in place (else copy)")
    output_path: str | None = Field(
        None, description="Copy destination (required when in_place is false)"
    )

    @field_validator("output_path")
    @classmethod
    def _check_output_path(cls, v):
        return _validate_output_path(v)


class WatermarkVerifyRequest(BaseModel):
    model: str = Field(..., description="Model to verify (path/alias/repo)")
    secret: str = Field(..., description="Same secret used at embed")
    layers: list[str] | None = Field(
        None, description="Same restriction used at embed"
    )
    bits_per_weight: int = Field(1, ge=1, le=3, description="Same value used at embed")
