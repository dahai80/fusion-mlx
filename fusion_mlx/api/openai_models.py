# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-mlx (https://github.com/vllm-project/vllm-mlx).
"""
Pydantic models for OpenAI-compatible API.

These models define the request and response schemas for:
- Chat completions
- Text completions
- Tool calling
- MCP (Model Context Protocol) integration
"""

import json
import logging
import math
from typing import Any

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from fusion_mlx.api.models import _validate_response_format_raw
from fusion_mlx.api.shared_models import (
    BaseUsage,
    IDPrefix,
    _validate_finite_in_range,
    _validate_logit_bias_finite,
    _validate_nonnegative_int,
    generate_id,
    get_unix_timestamp,
)

logger = logging.getLogger(__name__)


def _reject_nonfinite_float(v):
    if v is None:
        return v
    if not math.isfinite(v):
        raise ValueError("sampling parameter must be finite")
    return v


# =============================================================================
# Content Types
# =============================================================================


class ImageURL(BaseModel):
    """Image URL or base64 data URI for vision model input."""

    url: str  # "https://..." or "data:image/jpeg;base64,..."
    detail: str | None = "auto"  # "low", "high", "auto"


class VideoURL(BaseModel):
    """Video URL or base64 data URI for video model input."""

    url: str  # "https://..." or "data:video/mp4;base64,..."


class AudioURL(BaseModel):
    """Audio URL or base64 data URI for audio model input."""

    url: str  # "https://..." or "data:audio/wav;base64,..."


class ContentPart(BaseModel):
    """
    A part of a message content array.

    Supports:
    - text: Plain text content
    - image_url: Image input for vision models
    - video: Local video path for video models
    - video_url: Video input for video models
    - audio_url: Audio input for audio models
    - file: File attachment (PDF, etc.)
    """

    # Mirrors fusion_mlx.api.models.ContentPart so the OpenAI route does not
    # silently drop video/audio parts via pydantic field filtering. (#77)
    type: str  # "text", "image_url", "video", "video_url", "audio_url", or "file"
    text: str | None = None
    image_url: ImageURL | None = None
    video: str | None = None
    video_url: VideoURL | dict | str | None = None
    audio_url: AudioURL | dict | str | None = None
    file: dict | None = None
    # OpenAI Chat Completions audio input part: ``{data, format}``. Declared
    # so ``model_dump()`` preserves it for the VLM multimodal path — without
    # this field pydantic silently drops the sub-dict and audio content sent
    # through the chat endpoint is lost (parity with responses_models and
    # the image_url/file fields above).
    input_audio: dict[str, Any] | None = None
    # Anthropic-compatible cache_control hint (extension for OpenAI endpoint).
    # When set on a system message content part, marks the prefix boundary
    # for KV cache reuse across requests sharing the same system prompt.
    cache_control: dict[str, str] | None = None


# =============================================================================
# Messages
# =============================================================================


class Message(BaseModel):
    """
    A message in a chat conversation.

    Supports:
    - Simple text messages (role + content string)
    - Content array messages (role + content list with text parts)
    - Tool call messages (assistant with tool_calls)
    - Tool response messages (role="tool" with tool_call_id)
    """

    role: str
    content: str | list[ContentPart] | list[dict] | None = None
    # Reasoning/thinking content from <think> blocks (OpenAI reasoning_content field)
    reasoning_content: str | None = None
    # For assistant messages with tool calls
    tool_calls: list[dict] | None = None
    # For tool response messages (role="tool")
    tool_call_id: str | None = None
    # Participant name, rendered into chat template (e.g. Kimi K2/K2.5 named assistants)
    name: str | None = None
    # Continue from this message instead of starting a new turn (prefill / partial mode)
    partial: bool = False
    # Anthropic-compatible cache_control on the message itself (for string content)
    cache_control: dict[str, str] | None = None

    @field_validator("tool_calls", mode="before")
    @classmethod
    def _validate_tool_call_arguments(cls, v: Any) -> Any:
        """Validate arguments on each tool_call before the raw dict is stored.

        tool_calls is typed as List[dict] for flexibility, which bypasses
        FunctionCall's own validator. Re-run the same coercion here so
        malformed arguments surface as 422 instead of crashing the chat
        template on the next turn.
        """
        if not isinstance(v, list):
            return v
        for tc in v:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function")
            if not isinstance(func, dict) or "arguments" not in func:
                continue
            func["arguments"] = _coerce_tool_call_arguments(func["arguments"])
        return v


# =============================================================================
# Tool Calling
# =============================================================================


def _coerce_tool_call_arguments(v: Any) -> str:
    """Normalize a tool_call.arguments value to a JSON-object string.

    Native tool-calling chat templates (Qwen3.5/3.6, GLM-4.x, MiniMax)
    iterate `arguments.items()`, which requires the echoed value to parse
    back into a dict. Rejecting malformed inputs here turns the silent 500
    in downstream template rendering into a clear 422 that tells the client
    what to fix. Dict inputs (non-spec but common) are coerced to JSON
    strings, empty/whitespace strings normalize to ``"{}"``, and any value
    that can't round-trip into a JSON object raises ValueError.
    """
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    if not isinstance(v, str):
        raise ValueError(
            f"arguments must be a JSON-encoded string, got {type(v).__name__}. "
            "Per the OpenAI spec tool_call.arguments is a string containing JSON, "
            'not a dict/list/number. Example: \'{"location": "Tokyo"}\'.'
        )
    stripped = v.strip()
    if not stripped:
        return "{}"
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError) as e:
        snippet = stripped if len(stripped) <= 120 else stripped[:117] + "..."
        raise ValueError(
            f"arguments must be valid JSON, got parse error: {e}. "
            "This usually means the client echoed a previous tool call "
            "with a malformed arguments value. Send arguments as a "
            'JSON-encoded object string like \'{"location": "Tokyo"}\'. '
            f"Received: {snippet!r}"
        ) from e
    if not isinstance(parsed, dict):
        raise ValueError(
            f"arguments must be a JSON object, got {type(parsed).__name__}. "
            "Tool-call arguments cannot be a list, number, or bare string. "
            'Example: \'{"location": "Tokyo"}\'.'
        )
    return v


class FunctionCall(BaseModel):
    """A function call with name and arguments."""

    name: str
    arguments: str  # JSON string

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name_whitespace(cls, v: Any) -> str:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("arguments", mode="before")
    @classmethod
    def _validate_arguments_json(cls, v: Any) -> str:
        return _coerce_tool_call_arguments(v)


class ToolCall(BaseModel):
    """A tool call from the model."""

    id: str
    type: str = "function"
    function: FunctionCall


class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by the model."""

    type: str = "function"
    function: dict


# =============================================================================
# Structured Output (JSON Schema)
# =============================================================================


class ResponseFormatJsonSchema(BaseModel):
    """JSON Schema definition for structured output."""

    name: str
    description: str | None = None
    schema_: dict = Field(alias="schema")  # JSON Schema specification
    strict: bool | None = False

    class Config:
        populate_by_name = True


class ResponseFormat(BaseModel):
    """
    Response format specification for structured output.

    Supports:
    - "text": Default text output (no structure enforcement)
    - "json_object": Forces valid JSON output
    - "json_schema": Forces JSON matching a specific schema
    """

    type: str = "text"  # "text", "json_object", "json_schema"
    json_schema: ResponseFormatJsonSchema | None = None


class StructuredOutputOptions(BaseModel):
    """vLLM-compatible structured output options.

    Exactly one field should be set. When passed via ``extra_body`` in the
    OpenAI client, the key is ``structured_outputs``.

    Supports:
    - json: JSON schema (dict or string) for logit-level enforcement
    - regex: Regular expression the output must match
    - choice: List of allowed string values (output will be exactly one)
    - grammar: EBNF/GBNF context-free grammar string
    """

    model_config = {"populate_by_name": True}

    json_schema: str | dict | None = Field(None, alias="json")
    regex: str | None = None
    choice: list[str] | None = None
    grammar: str | None = None


# =============================================================================
# Chat Completion
# =============================================================================


class StreamOptions(BaseModel):
    """Options for streaming responses."""

    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""

    model: str
    # Optional LoRA adapter path (mlx-lm server-compatible). When set, the
    # request is routed to a derived engine entry keyed by (model, adapter)
    # so each adapter gets its own loaded model instance.
    adapters: str | None = None
    # D-ANTHRO-VALIDATION F11 parity: messages=[] must 400, not 500.
    messages: list[Message] = Field(min_length=1)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    top_k: int | None = None
    repetition_penalty: float | None = Field(None, ge=0.0)
    max_tokens: int | None = Field(None, ge=1, le=131072)
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: list[str] | None = None
    min_p: float | None = Field(None, ge=0.0, le=1.0)
    xtc_probability: float | None = None
    xtc_threshold: float | None = None
    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    # Tool calling
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict | None = None  # "auto", "none", or specific tool
    # Structured output
    response_format: ResponseFormat | dict | None = None
    # vLLM-compatible structured output (grammar, regex, choice, json)
    structured_outputs: StructuredOutputOptions | dict | None = None
    # Grammar backend selection: "auto" | "llguidance" | "xgrammar"
    grammar_backend: str | None = None
    # Chat template kwargs (e.g. enable_thinking, reasoning_effort)
    chat_template_kwargs: dict[str, Any] | None = None
    # Thinking budget (max thinking tokens, None = unlimited)
    thinking_budget: int | None = None
    # SpecPrefill: per-request enable/disable (None = use model setting)
    specprefill: bool | None = None
    # SpecPrefill: per-request keep percentage (0.1-0.5, None = use model setting)
    specprefill_keep_pct: float | None = None
    # SpecPrefill: per-request threshold override (min tokens to trigger, None = use model setting)
    specprefill_threshold: int | None = None
    # Seed for reproducible generation (best-effort)
    seed: int | None = None
    # Logprobs: return log probabilities of output tokens (OpenAI-compatible).
    logprobs: bool | None = None
    top_logprobs: int | None = None
    # OpenAI logit_bias — declared so Pydantic stops silently dropping it;
    # values validated finite by _check_logit_bias (H-10). Currently not
    # forwarded to the mlx-lm sampler (tracked separately); the field gate
    # still rejects nan/inf/bool defensively before the route runs.
    logit_bias: dict[str, float] | None = None
    # Issue #226: optional client-supplied session id for per-session usage stats.
    session_id: str | None = None

    @field_validator("top_logprobs")
    @classmethod
    def _validate_top_logprobs(cls, v):
        if v is not None and (v < 0 or v > 20):
            raise ValueError("top_logprobs must be between 0 and 20")
        return v

    @field_validator("stop", mode="before")
    @classmethod
    def coerce_stop(cls, v):
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("tool_choice", mode="before")
    @classmethod
    def _validate_tool_choice_shape(cls, v):
        # H-16 OpenAI-surface parity: parse-time validation of
        # ``tool_choice`` on /v1/chat/completions. Before this, malformed
        # values (``{"type":"banana"}``, ``"any"``, ``{"foo":"bar"}``)
        # silent-degraded to free-form generation because the typed
        # ``str | dict`` union swallowed the shape and the chat-route
        # ``type=='function'`` guard did not match. Mirrors M-03's
        # ``_validate_tool_choice_type`` on the Anthropic surface
        # (anthropic_models.py), adapted to the OpenAI vocabulary.
        if v is None:
            return v
        if isinstance(v, str):
            allowed_strings = ("none", "auto", "required", "function")
            if v not in allowed_strings:
                logger.debug("ChatCompletionRequest rejecting tool_choice string=%r", v)
                raise ValueError(
                    f"tool_choice {v!r} is not recognized; "
                    f"allowed string values are: none, auto, required, function"
                )
            return v
        if isinstance(v, dict):
            ctype = v.get("type")
            if ctype is None:
                logger.debug(
                    "ChatCompletionRequest rejecting tool_choice object "
                    "without type field=%r",
                    v,
                )
                raise ValueError(
                    "tool_choice object must include a 'type' field; "
                    "allowed types are: auto, none, function, required"
                )
            allowed_types = ("auto", "none", "function", "required")
            if ctype not in allowed_types:
                logger.debug(
                    "ChatCompletionRequest rejecting tool_choice.type=%r",
                    ctype,
                )
                raise ValueError(
                    f"tool_choice.type {ctype!r} is not recognized; "
                    f"allowed types are: auto, none, function, required"
                )
            return v
        # Non-str/non-dict (numbers, lists, booleans) is rejected by the
        # ``str | dict`` union arm with a field-named error.
        return v

    @model_validator(mode="before")
    @classmethod
    def _alias_max_completion_tokens(cls, values: Any) -> Any:
        if isinstance(values, dict):
            if "max_completion_tokens" in values and "max_tokens" not in values:
                values["max_tokens"] = values.pop("max_completion_tokens")
            elif "max_completion_tokens" in values:
                values.pop("max_completion_tokens")
        return values

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v):
        return _validate_finite_in_range(
            v, min_value=0.0, max_value=2.0, field_name="temperature"
        )

    @field_validator("top_p")
    @classmethod
    def _check_top_p(cls, v):
        return _validate_finite_in_range(
            v,
            min_value=0.0,
            max_value=1.0,
            field_name="top_p",
            min_inclusive=True,
        )

    @field_validator("min_p")
    @classmethod
    def _check_min_p(cls, v):
        return _validate_finite_in_range(
            v, min_value=0.0, max_value=1.0, field_name="min_p"
        )

    @field_validator("repetition_penalty")
    @classmethod
    def _check_repetition_penalty(cls, v):
        return _validate_finite_in_range(
            v, min_value=0.0, max_value=2.0, field_name="repetition_penalty"
        )

    @field_validator("presence_penalty")
    @classmethod
    def _check_presence_penalty(cls, v):
        return _validate_finite_in_range(
            v, min_value=-2.0, max_value=2.0, field_name="presence_penalty"
        )

    @field_validator("frequency_penalty")
    @classmethod
    def _check_frequency_penalty(cls, v):
        return _validate_finite_in_range(
            v, min_value=-2.0, max_value=2.0, field_name="frequency_penalty"
        )

    @field_validator("top_k")
    @classmethod
    def _check_top_k(cls, v):
        return _validate_nonnegative_int(v, field_name="top_k")

    @field_validator("logit_bias")
    @classmethod
    def _check_logit_bias(cls, v):
        return _validate_logit_bias_finite(v)


class AssistantMessage(BaseModel):
    """Response message from the assistant."""

    role: str = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None

    @field_validator("content", mode="before")
    @classmethod
    def _sanitize_content_field(cls, v):
        if v is None or not isinstance(v, str):
            return v
        from .utils import sanitize_output

        return sanitize_output(v)

    @field_validator("reasoning_content", mode="before")
    @classmethod
    def _sanitize_reasoning_content_field(cls, v):
        if v is None or not isinstance(v, str):
            return v
        from .utils import sanitize_reasoning_content

        return sanitize_reasoning_content(v)


class ChatCompletionChoice(BaseModel):
    """A single choice in chat completion response."""

    index: int = 0
    message: AssistantMessage
    finish_reason: str | None = "stop"
    logprobs: Any = None


class PromptTokensDetails(BaseModel):
    """Breakdown of prompt tokens used."""

    cached_tokens: int | None = None
    audio_tokens: int | None = None


class Usage(BaseUsage):
    """Token usage statistics for OpenAI API.

    Extends BaseUsage with optional timing metrics (FusionMLX extension).
    When present, timing values are in seconds.
    """

    prompt_tokens_details: PromptTokensDetails | None = None
    # Timing metrics (FusionMLX extension, seconds)
    model_load_duration: float | None = None
    time_to_first_token: float | None = None
    total_time: float | None = None
    prompt_eval_duration: float | None = None
    generation_duration: float | None = None
    prompt_tokens_per_second: float | None = None
    generation_tokens_per_second: float | None = None


class ChatCompletionResponse(BaseModel):
    """Response for chat completion."""

    id: str = Field(default_factory=lambda: generate_id(IDPrefix.CHAT_COMPLETION))
    object: str = "chat.completion"
    created: int = Field(default_factory=get_unix_timestamp)
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# =============================================================================
# Text Completion
# =============================================================================


class CompletionRequest(BaseModel):
    """Request for text completion."""

    model: str
    # Optional LoRA adapter path (mlx-lm server-compatible). See
    # ChatCompletionRequest.adapters for routing semantics.
    adapters: str | None = None
    prompt: str | list[str]
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    top_k: int | None = None
    repetition_penalty: float | None = Field(None, ge=0.0)
    max_tokens: int | None = Field(None, ge=1, le=131072)
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: list[str] | None = None
    min_p: float | None = Field(None, ge=0.0, le=1.0)
    xtc_probability: float | None = None
    xtc_threshold: float | None = None
    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    # Seed for reproducible generation (best-effort)
    seed: int | None = None
    # OpenAI FIM (fill-in-the-middle) suffix. Declared so Pydantic stops
    # silently dropping it; rejected with 400 in the /v1/completions route
    # when non-empty since no MLX engine implements FIM yet (silently
    # ignoring it produces wrong completions on code-completion clients).
    suffix: str | None = None
    # R10-H4: /v1/completions response_format parity with the chat lane.
    # Pre-fix the field was undeclared -> Pydantic dropped it (extra=ignore)
    # -> {"type":"json_object"} silently vanished on both sync + stream.
    # Declared + validated so the route can apply the same JSON peel.
    response_format: ResponseFormat | dict | None = None

    @field_validator("response_format", mode="before")
    @classmethod
    def _validate_response_format_field(cls, v):
        # Same closed-set check the chat lane runs (text/json_object/
        # json_schema). Rejects unknown type + missing type + malformed
        # json_schema shape before Pydantic's Union arm coerces them.
        return _validate_response_format_raw(v)

    @field_validator("stop", mode="before")
    @classmethod
    def coerce_stop(cls, v):
        """Accept stop as a single string (OpenAI compat) and wrap in a list."""
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v):
        return _validate_finite_in_range(
            v, min_value=0.0, max_value=2.0, field_name="temperature"
        )

    @field_validator("top_p")
    @classmethod
    def _check_top_p(cls, v):
        return _validate_finite_in_range(
            v,
            min_value=0.0,
            max_value=1.0,
            field_name="top_p",
            min_inclusive=True,
        )

    @field_validator("min_p")
    @classmethod
    def _check_min_p(cls, v):
        return _validate_finite_in_range(
            v, min_value=0.0, max_value=1.0, field_name="min_p"
        )

    @field_validator("repetition_penalty")
    @classmethod
    def _check_repetition_penalty(cls, v):
        return _validate_finite_in_range(
            v, min_value=0.0, max_value=2.0, field_name="repetition_penalty"
        )

    @field_validator("presence_penalty")
    @classmethod
    def _check_presence_penalty(cls, v):
        return _validate_finite_in_range(
            v, min_value=-2.0, max_value=2.0, field_name="presence_penalty"
        )

    @field_validator("frequency_penalty")
    @classmethod
    def _check_frequency_penalty(cls, v):
        return _validate_finite_in_range(
            v, min_value=-2.0, max_value=2.0, field_name="frequency_penalty"
        )

    @field_validator("top_k")
    @classmethod
    def _check_top_k(cls, v):
        return _validate_nonnegative_int(v, field_name="top_k")


class CompletionChoice(BaseModel):
    """A single choice in text completion response."""

    index: int = 0
    text: str
    finish_reason: str | None = "stop"


class CompletionResponse(BaseModel):
    """Response for text completion."""

    id: str = Field(default_factory=lambda: generate_id(IDPrefix.COMPLETION))
    object: str = "text_completion"
    created: int = Field(default_factory=get_unix_timestamp)
    model: str
    choices: list[CompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# =============================================================================
# Models List
# =============================================================================


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=get_unix_timestamp)
    owned_by: str = "fusion-mlx"
    modality: str = "text"
    capabilities: dict | None = None


class ModelsResponse(BaseModel):
    """Response for listing models."""

    object: str = "list"
    data: list[ModelInfo]


# =============================================================================
# MCP (Model Context Protocol)
# =============================================================================


class MCPToolInfo(BaseModel):
    """Information about an MCP tool."""

    name: str
    description: str
    server: str
    parameters: dict = Field(default_factory=dict)


class MCPToolsResponse(BaseModel):
    """Response for listing MCP tools."""

    tools: list[MCPToolInfo]
    count: int


class MCPServerInfo(BaseModel):
    """Information about an MCP server."""

    name: str
    state: str
    transport: str
    tools_count: int
    error: str | None = None


class MCPServersResponse(BaseModel):
    """Response for listing MCP servers."""

    servers: list[MCPServerInfo]


class MCPExecuteRequest(BaseModel):
    """Request to execute an MCP tool."""

    model_config = {"populate_by_name": True}

    tool_name: str = Field(validation_alias=AliasChoices("tool_name", "tool"))
    arguments: dict = Field(default_factory=dict)


class MCPExecuteResponse(BaseModel):
    """Response from executing an MCP tool."""

    tool_name: str
    content: str | list | dict | None = None
    is_error: bool = False
    error_message: str | None = None


# =============================================================================
# Streaming (for SSE responses)
# =============================================================================


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in a streaming chunk."""

    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict] | None = None

    @field_validator("content", mode="before")
    @classmethod
    def _sanitize_content_delta(cls, v):
        if v is None or not isinstance(v, str):
            return v
        from .utils import sanitize_reasoning_for_stream

        out = sanitize_reasoning_for_stream(v)
        return out or None

    @field_validator("reasoning_content", mode="before")
    @classmethod
    def _sanitize_reasoning_delta(cls, v):
        if v is None or not isinstance(v, str):
            return v
        from .utils import sanitize_reasoning_for_stream

        out = sanitize_reasoning_for_stream(v)
        return out or None


class ChatCompletionChunkChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None
    logprobs: Any = None


class ChatCompletionChunk(BaseModel):
    """A streaming chunk for chat completion."""

    id: str = Field(default_factory=lambda: generate_id(IDPrefix.CHAT_COMPLETION))
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=get_unix_timestamp)
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: Usage | None = None  # Present on last chunk when include_usage=true
