# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for Anthropic Messages API.

These models define the request and response schemas for:
- Anthropic Messages API (/v1/messages)
- Streaming events
- Tool calling in Anthropic format
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fusion_mlx.api.shared_models import (
    IDPrefix,
    _validate_finite_in_range,
    _validate_nonnegative_int,
    generate_id,
)

from .models import StreamOptions, _validate_token_budget

logger = logging.getLogger(__name__)


ANTHROPIC_EFFORT_TO_REASONING_MAX_TOKENS: dict[str, int] = {
    "low": 512,
    "medium": 2048,
    "high": 8192,
}

# =============================================================================
# Content Blocks
# =============================================================================


class ContentBlockText(BaseModel):
    """Text content block."""

    type: Literal["text"] = "text"
    text: str

    @field_validator("text", mode="before")
    @classmethod
    def _validate_text_type(cls, value):
        # H-15: reject non-string ``text`` with a clean, field-named
        # message. ``text: str`` already 422s on a non-string value, but
        # Pydantic's default ``string_type`` message buries ``text`` under
        # the ``str | list[ContentBlock]`` union loc trail on the parent
        # ``AnthropicMessage.content``. Run a ``mode="before"`` validator
        # so the error surfaces a single actionable line naming
        # ``content[].text`` (parity with the OpenAI ``ContentPart`` side).
        # D-ANTHRO-VALIDATION F4: a text block with null/missing ``text``
        # carries no usable content — the spec rejects ``{type:'text'}``.
        # ``ContentBlockText`` is text-only (unlike the OpenAI ``ContentPart``
        # union where ``text=None`` is legal for a media part), so reject
        # ``None`` here with the same field-named message.
        if isinstance(value, str):
            return value
        logger.debug(
            "ContentBlockText rejecting non-string text type=%s",
            type(value).__name__,
        )
        raise ValueError(
            f"content[].text must be a string (got {type(value).__name__})"
        )


class ContentBlockImage(BaseModel):
    """Image content block with source data."""

    type: Literal["image"] = "image"
    source: dict[
        str, Any
    ]  # {"type": "base64"|"url", "media_type": "...", "data"|"url": "..."}

    @model_validator(mode="after")
    def _validate_source_string_fields(self):
        # H-15 sibling: ``source`` is declared ``dict[str, Any]`` (no inner
        # schema), so a non-string ``data`` / ``url`` value (e.g. a nested
        # list or int) falls through the schema layer and surfaces as an
        # uninformative downstream error when the adapter decodes the
        # base64/url. Pin a string-typed check here for parity with the
        # OpenAI-side ``image_url.url`` rule (F-066) so the failure names
        # the field cleanly at the schema layer.
        if isinstance(self.source, dict):
            for key in ("data", "url"):
                if key in self.source:
                    val = self.source[key]
                    if val is not None and not isinstance(val, str):
                        logger.debug(
                            "ContentBlockImage rejecting non-string source.%s"
                            " type=%s",
                            key,
                            type(val).__name__,
                        )
                        raise ValueError(
                            f"image source.{key} must be a string "
                            f"(got {type(val).__name__})"
                        )
        return self


class ContentBlockToolUse(BaseModel):
    """Tool use content block (model requesting a tool call)."""

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ContentBlockToolResult(BaseModel):
    """Tool result content block (user providing tool output)."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[dict[str, Any]] | dict[str, Any] | list[Any] | Any
    is_error: bool | None = None


class ContentBlockThinking(BaseModel):
    """Thinking content block for reasoning models (e.g., Claude Opus 4.6)."""

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None


class ContentBlockDocument(BaseModel):
    """Document content block (PDF, plain text)."""

    type: Literal["document"] = "document"
    source: dict[
        str, Any
    ]  # {"type": "base64", "media_type": "application/pdf", "data": "..."}
    title: str | None = None
    context: str | None = None
    citations: dict[str, Any] | None = None
    cache_control: dict[str, str] | None = None


class ContentBlockInputAudio(BaseModel):
    """Audio input content block."""

    type: Literal["input_audio"] = "input_audio"
    source: dict[
        str, Any
    ]  # {"type": "base64", "media_type": "audio/wav", "data": "..."}


# Union type for all content blocks
ContentBlock = (
    ContentBlockText
    | ContentBlockImage
    | ContentBlockToolUse
    | ContentBlockToolResult
    | ContentBlockThinking
    | ContentBlockDocument
    | ContentBlockInputAudio
)


# =============================================================================
# System Content
# =============================================================================


class SystemContent(BaseModel):
    """System message content block."""

    type: Literal["text"] = "text"
    text: str
    cache_control: dict[str, str] | None = None


# =============================================================================
# Messages
# =============================================================================


_ALLOWED_ANTHROPIC_BLOCK_TYPES = frozenset(
    {
        "text",
        "image",
        "tool_use",
        "tool_result",
        "thinking",
        "document",
        "input_audio",
    }
)

_ANTHROPIC_BLOCK_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "text": ("text",),
    "image": ("source",),
    "tool_use": ("id", "name", "input"),
    "tool_result": ("tool_use_id", "content"),
    "thinking": ("thinking",),
    "document": ("source",),
    "input_audio": ("source",),
}

# D-ANTHRO-VALIDATION F10: role-block compatibility matrix. user-role
# messages carry tool results / images / documents; assistant-role
# messages carry thinking / tool_use. Cross-role blocks have undefined
# semantics and must 400 at the schema layer.
_ROLE_DISALLOWED_BLOCKS: dict[str, frozenset[str]] = {
    "user": frozenset({"thinking", "tool_use"}),
    "assistant": frozenset({"tool_result", "image"}),
}


class AnthropicMessage(BaseModel):
    """A message in an Anthropic conversation."""

    role: Literal["user", "assistant", "system"]
    content: str | list[ContentBlock]

    @model_validator(mode="before")
    @classmethod
    def _validate_block_shape_and_role(cls, data):
        # D-ANTHRO-VALIDATION F4 + F10: validate content-block shape
        # (recognized type + per-type required fields) and role
        # recognition at the raw-input layer so the error surfaces a
        # single actionable line rather than the union-arm error salad
        # Pydantic emits when the discriminated union fails.
        if not isinstance(data, dict):
            return data
        role = data.get("role")
        if role is not None and role not in ("user", "assistant", "system"):
            logger.debug("AnthropicMessage rejecting unrecognized role=%r", role)
            raise ValueError(
                f"{role!r} is not recognized as a role; allowed roles "
                f"are: user, assistant, system"
            )
        content = data.get("content")
        if not isinstance(content, list):
            return data
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype not in _ALLOWED_ANTHROPIC_BLOCK_TYPES:
                logger.debug(
                    "AnthropicMessage rejecting unrecognized block type=%r",
                    btype,
                )
                raise ValueError(
                    f"{btype!r} is not a recognized Anthropic content block "
                    f"type; allowed types are: "
                    f"{', '.join(sorted(_ALLOWED_ANTHROPIC_BLOCK_TYPES))}"
                )
            required = _ANTHROPIC_BLOCK_REQUIRED_FIELDS.get(btype, ())
            missing = [f for f in required if f not in block or block[f] is None]
            if missing:
                logger.debug(
                    "AnthropicMessage rejecting %s block missing=%s",
                    btype,
                    missing,
                )
                raise ValueError(
                    f"{btype} block is missing required field(s): "
                    f"{', '.join(missing)}"
                )
        return data

    @model_validator(mode="after")
    def _validate_role_block_compat(self):
        # D-ANTHRO-VALIDATION F10: cross-role block-type violations
        # (user-role thinking/tool_use, assistant-role tool_result/image)
        # have undefined semantics. Reject at the schema layer.
        disallowed = _ROLE_DISALLOWED_BLOCKS.get(self.role, frozenset())
        if not disallowed or not isinstance(self.content, list):
            return self
        for block in self.content:
            btype = getattr(block, "type", None)
            if btype in disallowed:
                logger.debug(
                    "AnthropicMessage rejecting role=%s with block=%s",
                    self.role,
                    btype,
                )
                raise ValueError(
                    f"role {self.role!r} is not allowed to use block type " f"{btype!r}"
                )
        return self


# =============================================================================
# Tools
# =============================================================================


class AnthropicTool(BaseModel):
    """Tool definition in Anthropic format.

    Supports two shapes:
        1. User-defined tool: requires ``input_schema``.
        2. Anthropic server-side tool (web_search, code_execution, bash,
        text_editor, computer): carries a versioned ``type`` like
        ``web_search_20250305`` and no ``input_schema``. FusionMLX cannot execute
        these locally; they are accepted for compatibility with clients such
        as Claude for Excel/PowerPoint/Word and dropped before inference.
    """

    name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    type: str | None = None
    cache_control: dict[str, str] | None = None

    # Forward-compat with extra fields Anthropic may attach to server-side
    # tools (e.g. max_uses, allowed_domains, user_location for web_search).
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _require_schema_or_type(self) -> "AnthropicTool":
        if self.input_schema is None and self.type is None:
            raise ValueError(
                "AnthropicTool requires either 'input_schema' (user-defined "
                "tool) or 'type' (Anthropic server-side tool)."
            )
        return self


class ToolChoice(BaseModel):
    """Tool choice specification."""

    type: Literal["auto", "any", "tool"]
    name: str | None = None  # Required when type="tool"


# =============================================================================
# Thinking Configuration
# =============================================================================


class ThinkingConfig(BaseModel):
    """Configuration for extended thinking/reasoning."""

    type: Literal["enabled", "disabled", "adaptive"] = "enabled"
    budget_tokens: int | None = None


# =============================================================================
# Request
# =============================================================================


class MessagesRequest(BaseModel):
    """Request for Anthropic Messages API."""

    model: str
    # Optional LoRA adapter path (mlx-lm server-compatible). When set, the
    # request is routed to a derived engine entry keyed by (model, adapter)
    # so each adapter gets its own loaded model instance.
    adapters: str | None = None
    max_tokens: int = Field(ge=1, le=131072)
    # D-ANTHRO-VALIDATION F11: messages=[] must 400 (not 500). Pydantic
    # ``min_length=1`` surfaces a clear "at least 1 item" message.
    messages: list[AnthropicMessage] = Field(min_length=1)
    system: str | list[SystemContent] | None = None
    stop_sequences: list[str] | None = None
    stream: bool = False
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    top_k: int | None = Field(None, ge=0)
    metadata: dict[str, Any] | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: dict[str, Any] | ToolChoice | None = None
    thinking: ThinkingConfig | None = None

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v):
        # Anthropic spec: temperature in [0.0, 1.0] (narrower than OpenAI's
        # [0, 2]). The Field allows up to 2.0 for forward-compat with the
        # OpenAI surface, but /v1/messages rejects > 1.0 per the Anthropic
        # contract (H-10 sweep).
        return _validate_finite_in_range(
            v, min_value=0.0, max_value=1.0, field_name="temperature"
        )

    @field_validator("top_p")
    @classmethod
    def _check_top_p(cls, v):
        # Anthropic spec: top_p in (0.0, 1.0] — 0.0 is illegal (exclusive min).
        return _validate_finite_in_range(
            v,
            min_value=0.0,
            max_value=1.0,
            field_name="top_p",
            min_inclusive=False,
        )

    @field_validator("top_k")
    @classmethod
    def _check_top_k(cls, v):
        return _validate_nonnegative_int(v, field_name="top_k")

    @field_validator("tool_choice", mode="before")
    @classmethod
    def _validate_tool_choice_type(cls, v):
        # D-ANTHRO-VALIDATION F1 / M-03 (#742 follow-up): parse-time
        # validation of ``tool_choice.type`` on /v1/messages. Before
        # this, ``tool_choice={"type":"banana"}`` silently degraded to
        # "auto" (the adapter's _convert_tool_choice fell through).
        # The field is ``dict | ToolChoice | None`` with ``dict`` first
        # so a spec-legal dict is preserved verbatim (the adapter and
        # chat route read the raw dict; coercing to ``ToolChoice`` would
        # drop extra keys like ``disable_parallel_tool_use``).
        if v is None:
            return v
        # Non-dict (e.g. a bare string "auto" — the OpenAI shape) is
        # rejected by the union arm; return as-is so the union surfaces
        # a "tool_choice" field-named error rather than a misleading
        # gate message.
        if not isinstance(v, dict):
            return v
        ctype = v.get("type")
        # Empty dict {} is spec-back-compat: the adapter defaults to
        # "auto" (pinned by TestConvertToolChoice.test_missing_type).
        if ctype is None:
            return v
        if ctype not in ("auto", "any", "tool", "none"):
            logger.debug("MessagesRequest rejecting tool_choice.type=%r", ctype)
            raise ValueError(
                f"tool_choice.type {ctype!r} is not recognized; "
                f"allowed types are: auto, any, tool, none"
            )
        if ctype == "tool":
            name = v.get("name")
            if not isinstance(name, str) or name.strip() == "":
                logger.debug(
                    "MessagesRequest rejecting tool_choice type=tool " "name=%r",
                    name,
                )
                raise ValueError(
                    "tool_choice.type 'tool' requires a non-empty string " "'name'"
                )
        return v

    # Chat template kwargs (e.g. enable_thinking, reasoning_effort)
    chat_template_kwargs: dict[str, Any] | None = None
    # OpenAI-compat surface: stream_options.include_usage gates the trailing
    # usage SSE chunk. Declared on the Anthropic model so the cross-route
    # StreamOptions contract is uniform (pinned by TestStreamOptionsIncludeUsageCrossRoute).
    stream_options: StreamOptions | None = None

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _validate_token_budget_field(cls, v, info):
        # Reject bool / non-int / non-positive before lax coercion -
        # pinned by TestPositiveIntGenerationBudget (cross-route parity).
        return _validate_token_budget(v, info.field_name)


AnthropicRequest = MessagesRequest


# =============================================================================
# Token Counting
# =============================================================================


class TokenCountRequest(BaseModel):
    """Request for token counting (Anthropic format)."""

    model: str
    messages: list[AnthropicMessage] = Field(min_length=1)
    system: str | list[SystemContent] | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: ToolChoice | dict[str, Any] | None = None
    thinking: ThinkingConfig | None = None


class TokenCountResponse(BaseModel):
    """Response for token counting."""

    input_tokens: int


# =============================================================================
# Response
# =============================================================================


class AnthropicUsage(BaseModel):
    """Token usage statistics for Anthropic API."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    """Response for Anthropic Messages API."""

    id: str = Field(default_factory=lambda: generate_id(IDPrefix.MESSAGE))
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: list[ContentBlockText | ContentBlockToolUse | ContentBlockThinking]
    stop_reason: (
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"] | None
    ) = None
    stop_sequence: str | None = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


# =============================================================================
# Streaming Events
# =============================================================================


class MessageStartEvent(BaseModel):
    """Event sent at the start of a message."""

    type: Literal["message_start"] = "message_start"
    message: dict[str, Any]  # Partial MessagesResponse


class ContentBlockStartEvent(BaseModel):
    """Event sent at the start of a content block."""

    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: dict[str, Any]  # Partial content block


class TextDelta(BaseModel):
    """Text delta for streaming."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class InputJsonDelta(BaseModel):
    """JSON input delta for tool use streaming."""

    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class ContentBlockDeltaEvent(BaseModel):
    """Event sent for content block updates."""

    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: TextDelta | InputJsonDelta | dict[str, Any]


class ContentBlockStopEvent(BaseModel):
    """Event sent when a content block ends."""

    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaEvent(BaseModel):
    """Event sent for message-level updates (stop_reason, usage)."""

    type: Literal["message_delta"] = "message_delta"
    delta: dict[str, Any]  # {"stop_reason": "...", "stop_sequence": ...}
    usage: dict[str, int]  # {"output_tokens": N}


class MessageStopEvent(BaseModel):
    """Event sent when the message ends."""

    type: Literal["message_stop"] = "message_stop"


class PingEvent(BaseModel):
    """Ping event for keeping connection alive."""

    type: Literal["ping"] = "ping"


class ErrorEvent(BaseModel):
    """Error event for streaming errors."""

    type: Literal["error"] = "error"
    error: dict[str, Any]  # {"type": "...", "message": "..."}


# Union type for all streaming events
StreamingEvent = (
    MessageStartEvent
    | ContentBlockStartEvent
    | ContentBlockDeltaEvent
    | ContentBlockStopEvent
    | MessageDeltaEvent
    | MessageStopEvent
    | PingEvent
    | ErrorEvent
)


# =============================================================================
# Error Response
# =============================================================================


class AnthropicErrorDetail(BaseModel):
    """Error detail in Anthropic format."""

    type: str  # "invalid_request_error", "authentication_error", "api_error", etc.
    message: str


class AnthropicErrorResponse(BaseModel):
    """Error response in Anthropic format."""

    type: Literal["error"] = "error"
    error: AnthropicErrorDetail
