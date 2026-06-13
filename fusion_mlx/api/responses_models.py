# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the OpenAI Responses API (/v1/responses)."""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .shared_models import IDPrefix, generate_id, get_unix_timestamp

# =============================================================================
# Request Models
# =============================================================================


class InputItem(BaseModel):
    """A single item in the Responses API input array.

    Supports EasyInputMessage (no type field), message, function_call,
    function_call_output, and many other types from the Responses API.
    """

    # type is optional — EasyInputMessage omits it
    type: str | None = None
    # message fields
    role: str | None = None
    content: str | list[Any] | None = None
    # function_call fields
    id: str | None = None
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    # function_call_output fields
    output: str | list[Any] | dict[str, Any] | None = None
    # status field (present on many item types)
    status: str | None = None

    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    @classmethod
    def _serialize_complex_output(cls, data: Any) -> Any:
        """Serialize list/dict output to JSON string for compatibility.

        Agent frameworks may send multimodal tool outputs (e.g. images) as
        lists or dicts. Convert them to JSON strings so downstream code that
        expects ``str`` keeps working.
        """
        if isinstance(data, dict):
            output = data.get("output")
            if isinstance(output, (list, dict)):
                data = {**data, "output": json.dumps(output)}
        return data


class ResponsesTool(BaseModel):
    """Tool definition in Responses API format.

    Supports function, local_shell, mcp, web_search, and other tool types.
    """

    type: str = "function"
    # function tool fields
    name: str | None = None
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None

    model_config = {"extra": "allow"}


class TextFormatConfig(BaseModel):
    """Text format configuration."""

    type: str = "text"  # "text", "json_object", "json_schema"
    name: str | None = None
    description: str | None = None
    schema_: dict[str, Any] | None = Field(None, alias="schema")
    strict: bool | None = None

    model_config = {"extra": "allow", "populate_by_name": True}


class TextConfig(BaseModel):
    """Text configuration wrapper."""

    format: TextFormatConfig | None = None
    verbosity: str | None = None  # "low", "medium", "high"

    model_config = {"extra": "allow"}


class ResponsesRequest(BaseModel):
    """Request body for POST /v1/responses."""

    model: str
    input: str | list[InputItem] | None = None
    instructions: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    stream: bool = False
    tools: list[ResponsesTool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    text: TextConfig | None = None
    previous_response_id: str | None = None
    store: bool | None = None
    truncation: str | None = None  # "auto" or "disabled"
    metadata: dict[str, str] | None = None
    reasoning: dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    # Fields that Codex CLI sends
    include: list[str] | None = None
    service_tier: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: str | None = None
    user: str | None = None
    top_logprobs: int | None = None
    background: bool | None = None
    conversation: Any | None = None
    max_tool_calls: int | None = None
    stream_options: dict[str, Any] | None = None
    # Seed for reproducible generation (best-effort)
    seed: int | None = None

    model_config = {"extra": "allow"}


# =============================================================================
# Response Models
# =============================================================================


class OutputContent(BaseModel):
    """Content block within an output message item."""

    type: str = "output_text"
    text: str = ""
    annotations: list[Any] = Field(default_factory=list)


class ReasoningSummaryPart(BaseModel):
    """A single part of a reasoning summary."""

    type: str = "summary_text"
    text: str = ""


class OutputItem(BaseModel):
    """A single item in the response output array.

    Can be a message, function_call, or reasoning.
    """

    type: str  # "message" or "function_call" or "reasoning"
    id: str
    status: str = "completed"
    # message fields
    role: str | None = None
    content: list[OutputContent] | None = None
    # function_call fields
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    # reasoning fields
    summary: list[ReasoningSummaryPart] | None = None


class InputTokensDetails(BaseModel):
    """Details about input token usage."""

    cached_tokens: int = 0


class OutputTokensDetails(BaseModel):
    """Details about output token usage."""

    reasoning_tokens: int = 0


class ResponseUsage(BaseModel):
    """Token usage for Responses API."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: InputTokensDetails = Field(
        default_factory=InputTokensDetails
    )
    output_tokens_details: OutputTokensDetails = Field(
        default_factory=OutputTokensDetails
    )

    def model_post_init(self, __context) -> None:
        if self.total_tokens == 0 and (self.input_tokens > 0 or self.output_tokens > 0):
            object.__setattr__(
                self,
                "total_tokens",
                self.input_tokens + self.output_tokens,
            )


class ResponseObject(BaseModel):
    """Full response object for the Responses API."""

    id: str = Field(default_factory=lambda: generate_id(IDPrefix.RESPONSE))
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=get_unix_timestamp)
    model: str
    status: str = "completed"  # "completed", "in_progress", "failed", "incomplete"
    output: list[OutputItem] = Field(default_factory=list)
    usage: ResponseUsage | None = None
    text: TextConfig | None = None
    tool_choice: str | dict[str, Any] | None = "auto"
    tools: list[ResponsesTool] = Field(default_factory=list)
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    previous_response_id: str | None = None
    metadata: dict[str, str] | None = Field(default_factory=dict)
    truncation: str | None = None
    error: dict[str, Any] | None = None
