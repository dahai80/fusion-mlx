# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the OpenAI Responses API (/v1/responses)."""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .models import (
    StreamOptions,
    _validate_reasoning_effort_value,
    _validate_response_format_raw,
    _validate_token_budget,
)
from .shared_models import (
    IDPrefix,
    generate_id,
    get_unix_timestamp,
    validate_seed,
    validate_top_k,
)

# =============================================================================
# Request Models
# =============================================================================

# #524: documented Responses-API message roles. InputItem.role must be one
# of these (or None for non-message item types). Module-level so Pydantic
# does not capture it as a ModelPrivateAttr.
_ALLOWED_RESPONSES_ROLES = frozenset(
    {"user", "assistant", "system", "tool", "developer"}
)


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

    # #524: reject unknown message roles at the Pydantic layer so the
    # RequestValidationError handler returns 400 with a clean field path,
    # instead of an opaque 500 inside the Jinja chat template. Only
    # message-type items carry a role; function_call / reasoning / output
    # items have role=None on the wire, so a non-None role must be one of
    # the documented Responses-API roles.
    @field_validator("role")
    @classmethod
    def _validate_role(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if v not in _ALLOWED_RESPONSES_ROLES:
            allowed = ", ".join(sorted(_ALLOWED_RESPONSES_ROLES))
            raise ValueError(f"Input 'role' must be one of: {allowed} (got {v!r})")
        return v

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


class ResponsesContentItem(BaseModel):
    type: str = "input_text"
    text: str | None = None
    image_url: str | dict[str, Any] | None = None
    input_audio: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


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
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    max_output_tokens: int | None = Field(None, ge=1)
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
    stream_options: StreamOptions | None = None
    # Seed for reproducible generation (best-effort)
    seed: int | None = None
    # Fields forwarded to ChatCompletionRequest
    reasoning_effort: str | None = None
    reasoning_max_tokens: int | None = Field(None, ge=1)
    enable_thinking: bool | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    top_k: int | None = Field(None, ge=0)

    model_config = {"extra": "allow"}

    @field_validator("response_format", mode="before")
    @classmethod
    def _validate_response_format_field(cls, v):
        return _validate_response_format_raw(v)

    @field_validator("reasoning_effort")
    @classmethod
    def _validate_reasoning_effort_field(cls, v):
        # R10-H5: top-level shorthand surface. /v1/responses accepts the
        # same reasoning_effort set as /v1/chat/completions — "banana"
        # must 422, not silently flow to the model. Mirrors the chat
        # lane via the shared _validate_reasoning_effort_value helper.
        return _validate_reasoning_effort_value(v)

    @field_validator("max_output_tokens", mode="before")
    @classmethod
    def _validate_token_budget_field(cls, v, info):
        # Reject bool / non-int / non-positive before lax coercion -
        # pinned by TestPositiveIntGenerationBudget (cross-route parity).
        return _validate_token_budget(v, info.field_name)

    @field_validator("top_k", mode="before")
    @classmethod
    def _validate_top_k_field(cls, v):
        # R6-H8 parity: /v1/responses must honour the same top_k cap +
        # bool/negative rejection as the chat surface. Pre-fix the
        # Responses surface silently dropped pathological top_k values.
        return validate_top_k(v, "top_k")

    @field_validator("seed", mode="before")
    @classmethod
    def _validate_seed_field(cls, v):
        # H-11 / r5-E B-8 parity: reject negative + bool seed before lax
        # coercion. seed=0 is a legitimate PRNG key and stays valid.
        return validate_seed(v, "seed")

    @model_validator(mode="after")
    def _validate_input_not_empty(self):
        # D-ANTHRO-VALIDATION F11 Responses parity: empty ``input``
        # ("" or []) must 400 with a clear envelope, not silently run
        # inference on nothing. ``input`` defaults to None (allowed —
        # some flows rely on ``instructions`` + ``previous_response_id``);
        # only reject the explicitly-empty string/list shapes.
        if self.input is None:
            return self
        if isinstance(self.input, str) and self.input == "":
            raise ValueError("input must not be empty")
        if isinstance(self.input, list) and len(self.input) == 0:
            raise ValueError("input must not be empty")
        return self

    @model_validator(mode="after")
    def _validate_reasoning_nested_effort(self):
        # R10-H5: canonical OpenAI Responses spec nests effort under
        # ``reasoning.effort`` (not the top-level shorthand). The
        # ``reasoning`` field is a free-form dict (other keys like
        # ``summary`` / ``encrypted_content`` must pass through), so we
        # only gate the ``effort`` key when present. Error surfaces the
        # dotted path ``reasoning.effort`` so the 400 envelope names the
        # right param. Mirrors the chat lane allowed-set.
        if not isinstance(self.reasoning, dict):
            return self
        if "effort" in self.reasoning:
            self.reasoning["effort"] = _validate_reasoning_effort_value(
                self.reasoning["effort"], field_path="reasoning.effort"
            )
        return self


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
    # computer_call fields
    action: dict[str, Any] | None = None
    pending_safety_checks: list[Any] = Field(default_factory=list)


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
    input_tokens_details: InputTokensDetails = Field(default_factory=InputTokensDetails)
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
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    max_output_tokens: int | None = Field(None, ge=1)
    previous_response_id: str | None = None
    metadata: dict[str, str] | None = Field(default_factory=dict)
    truncation: str | None = None
    error: dict[str, Any] | None = None
    instructions: str | None = None
    service_tier: str | None = None
    incomplete_details: dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None


# Rapid-MLX compatibility aliases
ResponsesInputItem = InputItem
ResponsesOutputContent = OutputContent
ResponsesOutputItem = OutputItem
ResponsesResponse = ResponseObject
ResponsesUsage = ResponseUsage
