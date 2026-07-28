# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for NER API endpoint."""

from typing import Any

from pydantic import BaseModel, Field


class NERRequest(BaseModel):
    """Request for named entity recognition."""

    text: str | list[str]
    """Input text or list of texts for NER extraction."""

    labels: list[str]
    """Entity types to extract (e.g., ['person', 'organization', 'location'])."""

    model: str
    """ID of the NER model to use."""

    threshold: float = Field(0.5, ge=0.0, le=1.0)
    """Minimum confidence threshold for entity extraction."""

    flat_ner: bool = True
    """Use flat NER (no nested entities)."""

    multi_label: bool = False
    """Allow multiple labels per entity."""

    user: str | None = None
    """OpenAI abuse-tracking field. Accepted but not validated."""


class NEREntity(BaseModel):
    """A single extracted entity."""

    start: int
    """Start character offset in the source text."""

    end: int
    """End character offset in the source text."""

    text: str
    """Extracted entity text."""

    label: str
    """Entity type label."""

    score: float
    """Confidence score."""


class NERUsage(BaseModel):
    """Token usage statistics for NER request."""

    prompt_tokens: int
    total_tokens: int


class NERResponse(BaseModel):
    """Response from NER extraction."""

    object: str = "list"
    data: list[list[NEREntity]]
    model: str
    usage: NERUsage
