# SPDX-License-Identifier: Apache-2.0
"""
Pydantic models for OpenAI-compatible API.

These models define the request and response schemas for:
- Chat completions
- Text completions
- Tool calling
- MCP (Model Context Protocol) integration
"""

import logging
import math
import re
import time
import uuid
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_serializer,
    model_validator,
)

from .shared_models import (
    _validate_finite_in_range,
    _validate_logit_bias_finite,
    _validate_nonnegative_int,
)

logger = logging.getLogger(__name__)


def _reject_nonfinite_float(v):
    if v is None:
        return v
    if not math.isfinite(v):
        raise ValueError("sampling parameter must be finite")
    return v


# =============================================================================
# Content Types (for multimodal messages)
# =============================================================================


class ImageUrl(BaseModel):
    """Image URL with optional detail level."""

    url: str
    detail: str | None = None


class VideoUrl(BaseModel):
    """Video URL."""

    url: str


class AudioUrl(BaseModel):
    """Audio URL for audio content."""

    url: str


class ContentPart(BaseModel):
    """
    A part of a multimodal message content.

    Supports:
    - text: Plain text content
    - image_url: Image from URL or base64
    - video: Video from local path
    - video_url: Video from URL or base64
    - audio_url: Audio from URL or base64
    """

    type: str  # "text", "image_url", "video", "video_url", "audio_url"
    text: str | None = None
    image_url: ImageUrl | dict | str | None = None
    video: str | None = None
    video_url: VideoUrl | dict | str | None = None
    audio_url: AudioUrl | dict | str | None = None

    @field_validator("text", mode="before")
    @classmethod
    def _validate_text_type(cls, value):
        # H-15: reject non-string ``text`` with a clean, field-named
        # message. ``text: str | None`` already 422s on a non-string
        # value, but Pydantic's default ``string_type`` message buries
        # ``text`` under a nested loc trail. Run a ``mode="before"``
        # validator so the error surfaces a single actionable line
        # naming ``content[].text`` (parity with the Anthropic side).
        if value is None or isinstance(value, str):
            return value
        logger.debug(
            "ContentPart rejecting non-string text type=%s",
            type(value).__name__,
        )
        raise ValueError(
            f"content[].text must be a string (got {type(value).__name__})"
        )

    @model_validator(mode="after")
    def _reject_bare_string_media(self):
        # F-065: a typed media part (type=="image_url"|"video_url"|
        # "audio_url") that carries a bare string in its media slot
        # (image_url="data:..." instead of {"url":"data:..."}) used to
        # slip past the schema layer — the union ``ImageUrl | dict | str``
        # accepts the string, the multimodal preprocessor unwraps
        # ``image["url"]`` only on the dict shape, and the bare-string
        # form was silently dropped (model received text only and
        # hallucinated, or 400'd on modality with a misleading message).
        # Reject at the ContentPart level so direct construction (not
        # just the Message dict-fallback arm) is guarded. Gate on
        # ``type`` so a type:"text" part carrying an unrelated
        # image_url string slot (legacy clients) is NOT collaterally
        # broken.
        media_slots = (
            ("image_url", "image_url"),
            ("video_url", "video_url"),
            ("audio_url", "audio_url"),
        )
        for type_value, field_name in media_slots:
            if self.type != type_value:
                continue
            slot = getattr(self, field_name, None)
            if isinstance(slot, str):
                logger.debug(
                    "ContentPart rejecting bare-string %s (type=%s)",
                    field_name,
                    type_value,
                )
                raise ValueError(
                    f"{field_name} must be an object with a 'url' field "
                    f"(got a bare string)"
                )
        return self


# =============================================================================
# Messages
# =============================================================================


class Message(BaseModel):
    """
    A message in a chat conversation.

    Supports:
    - Simple text messages (role + content string)
    - Multimodal messages (role + content list with text/images/videos)
    - Tool call messages (assistant with tool_calls)
    - Tool response messages (role="tool" with tool_call_id)
    """

    role: str
    content: str | list[ContentPart] | list[dict] | None = None
    # For assistant messages with tool calls
    tool_calls: list[dict] | None = None
    # For tool response messages (role="tool")
    tool_call_id: str | None = None

    @field_validator("content", mode="before")
    @classmethod
    def _validate_multimodal_content(cls, v):
        # F-066: the ``content`` union selects the ``list[dict]`` arm for a
        # multimodal part whose ``image_url``/``video_url``/``audio_url``
        # carries a non-string ``url`` (ImageUrl rejects int url, the union
        # falls back to dict, ``{"url": 123}`` is a valid dict), so
        # ContentPart's own validators never run. That non-string url then
        # crashed ``process_image_input`` -> ``is_base64_image(123)`` with a
        # raw ``AttributeError: 'int' object has no attribute 'startswith'``
        # that leaked into the 400 body. Scan dict parts here and reject
        # non-string url / bare-string shorthand at the schema layer (422).
        # H-15 sibling: the same dict-fallback escapes a non-string ``text``
        # (ContentPart's ``text`` validator only fires on the
        # ``list[ContentPart]`` arm). A non-string ``text`` in a dict part
        # used to surface as a buried union error or a downstream 500 in
        # ``_join_text_parts``; reject it here with the same clean message.
        if not isinstance(v, list):
            return v
        media_fields = ("image_url", "video_url", "audio_url")
        for part in v:
            if not isinstance(part, dict):
                continue
            text_value = part.get("text")
            if text_value is not None and not isinstance(text_value, str):
                logger.debug(
                    "Message rejecting non-string text in dict part type=%s",
                    type(text_value).__name__,
                )
                raise ValueError(
                    f"content[].text must be a string "
                    f"(got {type(text_value).__name__})"
                )
            for field_name in media_fields:
                slot = part.get(field_name)
                if slot is None:
                    continue
                if isinstance(slot, str):
                    raise ValueError(
                        f"{field_name} must be an object with a 'url' field "
                        f"(got a bare string)"
                    )
                if isinstance(slot, dict):
                    url = slot.get("url")
                    if not isinstance(url, str):
                        raise ValueError(
                            f"{field_name}.url must be a string "
                            f"(got {type(url).__name__})"
                        )
        return v

    @model_validator(mode="after")
    def _reject_null_content_for_input_roles(self):
        # R15 #175 item 3 (adversarial fuzz a9c828): a request body carrying
        # ``{"role":"user","content":null}`` (or "system" / "developer")
        # used to return HTTP 200 with garbled output (the chat-template
        # flattened ``None`` to the literal string ``"None"`` on some
        # templates, dropped the turn on others). Reject at the schema layer
        # so the route surfaces 400 via the RequestValidationError envelope.
        # Assistant + tool roles keep ``content=None`` allowed (OpenAI spec:
        # assistant content=null when tool_calls present; tool turn content
        # is the tool-result payload and may legitimately be null in
        # follow-up roundtrips).
        if self.role in ("user", "system", "developer") and self.content is None:
            raise ValueError(
                f"content is required for role '{self.role}' "
                f"(content must not be null)"
            )
        return self


# =============================================================================
# Tool Calling
# =============================================================================


class FunctionCall(BaseModel):
    """A function call with name and arguments."""

    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    """A tool call from the model."""

    id: str
    type: str = "function"
    function: FunctionCall


# R10-H6: chat-lane computer_use shorthand parity with the Responses
# lane (responses_adapter._convert_tools). A bare {"type": alias}
# (no ``function`` field) is rewritten to a synthetic ``computer``
# function tool so the UI-TARS tool parser sees a matching entry.
# Geometry hints (display_width / display_height / environment) ride
# inside ``function.parameters._computer_use`` — same plumbing the
# Responses lane uses. Aliases mirror _RESPONSES_TOOL_TYPE_ALIASES
# plus the bare ``computer_use`` shorthand. Module-level (NOT class
# attrs — pydantic treats a leading underscore as ModelPrivateAttr,
# breaking ``in`` lookups inside validators).
_COMPUTER_USE_ALIASES: frozenset[str] = frozenset(
    {"computer_use", "computer_use_preview", "computer_20251022"}
)
_COMPUTER_USE_HINT_KEYS: tuple[str, ...] = (
    "display_width",
    "display_height",
    "environment",
)


class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by the model."""

    type: str = "function"
    function: dict

    @model_validator(mode="before")
    @classmethod
    def _normalize_computer_use_shorthand(cls, data):
        # mode="before" so we see the raw dict before Pydantic enforces
        # ``function: dict`` (which would 400 on the shorthand). Only
        # rewrite the alias shape; the classic {"type":"function",
        # "function":{...}} shape passes through untouched.
        if not isinstance(data, dict):
            return data
        ttype = data.get("type")
        if ttype not in _COMPUTER_USE_ALIASES:
            return data
        if "function" in data and isinstance(data["function"], dict):
            # Caller already supplied a function block for the alias;
            # don't clobber — only stamp the canonical name if missing.
            fn = dict(data["function"])
            fn.setdefault("name", "computer")
            fn.setdefault("parameters", {})
            data["type"] = "function"
            data["function"] = fn
            return data
        hints = {k: data[k] for k in _COMPUTER_USE_HINT_KEYS if k in data}
        params = {"_computer_use": hints} if hints else {}
        data["type"] = "function"
        data["function"] = {"name": "computer", "parameters": params}
        # Drop the hint keys from the top level so extra=ignore doesn't
        # surface them (clean round-trip; the hints live under params).
        for k in _COMPUTER_USE_HINT_KEYS:
            data.pop(k, None)
        return data

    @field_validator("function")
    @classmethod
    def _validate_function_name(cls, v):
        # F-035 / F-146: ``function.name`` must match the OpenAI spec regex
        # ``^[a-zA-Z0-9_-]{1,64}$``. Pre-fix the schema accepted any dict,
        # and empty / emoji / 10k-char / shell-metachar / newline / space /
        # dot / slash names all 200'd — on hermes-parser models the
        # empty-name case leaked literal ``<tool_call>{"name":"",...}`` into
        # ``content`` because the tool-call detector keyed off a non-empty
        # name. A single regex constraint at the schema layer covers the
        # whole class. Surface the field path as ``function.name`` so the
        # OpenAI-shaped 400 envelope can populate ``error.param``.
        if not isinstance(v, dict):
            raise ValueError("function must be an object")
        name = v.get("name")
        if not isinstance(name, str) or not re.match(r"^[a-zA-Z0-9_-]{1,64}$", name):
            raise ValueError(
                "function.name must match ^[a-zA-Z0-9_-]{1,64}$ "
                "(1-64 chars: letters, digits, underscore, hyphen)"
            )
        return v


# =============================================================================
# Structured Output (JSON Schema)
# =============================================================================


class ResponseFormatJsonSchema(BaseModel):
    """JSON Schema definition for structured output."""

    name: str
    description: str | None = None
    schema_: dict = Field(alias="schema")  # JSON Schema specification
    strict: StrictBool | None = False

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
    strict: StrictBool | None = None


_VALID_RESPONSE_FORMAT_TYPES = ("text", "json_object", "json_schema")

# OpenAI / Anthropic compatible reasoning_effort closed set. Shared by
# ChatCompletionRequest (top-level field) and ResponsesRequest (top-level
# + nested reasoning.effort) so the two surfaces can't drift. "none" is
# the explicit-disable value (distinct from field-absent None). "xhigh"
# is a non-canonical extension some launchers (e.g. Codex-like clients)
# and models (e.g. Qwen3 family) emit; accepted as pass-through.
_REASONING_EFFORT_ALLOWED: frozenset[str] = frozenset(
    {"minimal", "low", "medium", "high", "xhigh", "none"}
)


def _validate_reasoning_effort_value(v, *, field_path: str = "reasoning_effort"):
    # Shared ``mode="before"``-style guard for reasoning_effort. Rejects
    # non-spec strings (e.g. "banana") before they reach the model. bool
    # is an int subclass but is not a str, so Pydantic's type coercion
    # rejects True/42/[]/{}/None-with-value at the field layer; this
    # guard closes the str-valued hole. Error surfaces the field path so
    # the nested reasoning.effort case reports "reasoning.effort". The
    # nested ``reasoning`` field is a free-form dict, so effort can
    # arrive as any type (list/int/bool) — reject non-str without
    # crashing on ``v not in frozenset`` (unhashable []/{} TypeError).
    if v is None:
        return v
    if not isinstance(v, str):
        raise ValueError(
            f"{field_path} must be one of "
            f"{sorted(_REASONING_EFFORT_ALLOWED)}, got {v!r}"
        )
    if v not in _REASONING_EFFORT_ALLOWED:
        raise ValueError(
            f"{field_path} must be one of "
            f"{sorted(_REASONING_EFFORT_ALLOWED)}, got {v!r}"
        )
    return v


def _validate_response_format_raw(v):
    # Shared ``mode="before"`` guard for the ``response_format`` field.
    # Validates the bare-dict union arm that Pydantic would otherwise
    # silently coerce; raises ValueError (-> 422 ValidationError) for
    # every shape that used to 200-through unconstrained. Returns the
    # value untouched on success so Pydantic can still coerce dicts.
    if v is None or isinstance(v, ResponseFormat):
        return v
    if not isinstance(v, dict):
        raise ValueError("response_format must be an object")
    if "type" not in v:
        raise ValueError("response_format.type is required")
    rf_type = v.get("type")
    if not isinstance(rf_type, str) or rf_type not in _VALID_RESPONSE_FORMAT_TYPES:
        raise ValueError(
            "response_format.type must be 'text', 'json_object' or 'json_schema'"
        )
    strict = v.get("strict")
    if strict is not None and not isinstance(strict, bool):
        raise ValueError(
            f"response_format.strict must be a boolean (got {type(strict).__name__})"
        )
    if rf_type == "json_schema":
        js = v.get("json_schema")
        if not isinstance(js, dict) or not js:
            raise ValueError(
                "response_format.json_schema must be a non-empty 'json_schema' field"
            )
        name = js.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("response_format.json_schema.name is required")
        schema = js.get("schema")
        if not isinstance(schema, dict):
            raise ValueError(
                "response_format.json_schema.schema must be an object, "
                f"got {type(schema).__name__}"
            )
        if not schema:
            raise ValueError(
                "response_format.json_schema.schema must be a non-empty object"
            )
        inner_strict = js.get("strict")
        if inner_strict is not None and not isinstance(inner_strict, bool):
            raise ValueError(
                "response_format.json_schema.strict must be a boolean "
                f"(got {type(inner_strict).__name__})"
            )
    return v


# =============================================================================
# Logprobs
# =============================================================================


class TopLogProb(BaseModel):
    """A top log probability for a token."""

    token: str
    logprob: float
    bytes: list[int] | None = None


class TokenLogProb(BaseModel):
    """Log probability information for a single token."""

    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogProb] = []


class ChoiceLogProbs(BaseModel):
    """Log probability information for a choice."""

    content: list[TokenLogProb] | None = None


# =============================================================================
# Chat Completion
# =============================================================================


def _validate_token_budget(v, field_name: str):
    # R7-M3 cross-route guard: every token-budget field (max_tokens on
    # chat/completions/messages, max_completion_tokens on chat,
    # max_output_tokens on /v1/responses) rejects bool (Python bool is an
    # int subclass -> would coerce True->1 silently), non-int wire shapes
    # (JSON string "100"), and non-positive values (0 / negative). None is
    # the server-default sentinel on the OpenAI-compat surfaces and stays
    # valid; the message names the field so cross-route tests can assert it.
    if v is None:
        return v
    if isinstance(v, bool):
        raise ValueError(f"{field_name} must be an integer, got boolean")
    if not isinstance(v, int):
        raise ValueError(f"{field_name} must be an integer")
    if v < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return v


def _reject_non_one_n(v):
    # F-155: pin the wire schema - ``n`` must equal 1 on both
    # /v1/chat/completions and /v1/completions. Server generates one
    # completion per request (no server-side rerank); n=0 is an
    # off-by-one typo, n=-1 is the SDK "use default" sentinel, n>1
    # requests rerank we don't implement. All three were silently
    # accepted as HTTP 200 with one choice, hiding the client bug.
    # bool is a Python int subclass - reject it BEFORE the == 1 check
    # so True->1 doesn't sneak through and False->0 doesn't re-intro n=0.
    if isinstance(v, bool):
        raise ValueError("n must be an integer, not bool")
    if v is None or v == 1:
        return v
    raise ValueError(
        "n must equal 1 (server generates one completion per "
        "request; no server-side rerank)"
    )


class StreamOptions(BaseModel):
    """Options for streaming responses."""

    # StrictBool rejects string "true"/"yes"/1 coercion at parse time so a
    # wire bug on include_usage surfaces as a 4xx, not a silent-200 (the
    # lax bool arm coerced "yes"->True and emitted usage unconditionally).
    include_usage: StrictBool = False  # Include usage stats in final chunk


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""

    model: str = "default"
    messages: list[Message]
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(None, ge=1, le=131072)
    # OpenAI-canonical token cap since Sept 2024 (preferred over max_tokens for
    # reasoning models; newer SDKs >=1.45 send only this field). Normalized to
    # max_tokens by a model_validator so all downstream code keeps reading the
    # single max_tokens field.
    max_completion_tokens: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = (
        None  # Streaming options (include_usage, etc.)
    )
    stop: list[str] | None = None
    # Extended OpenAI-compatible sampling parameters. Without these declared,
    # Pydantic drops them on parse (#355). top_k / min_p flow through to the
    # mlx-lm sampler; repetition_penalty / presence_penalty / frequency_penalty
    # flow through to mlx-lm's make_logits_processors().
    top_k: int | None = Field(None, ge=0)
    min_p: float | None = Field(None, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(None, ge=0.0)
    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    # Tool calling
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict | None = None  # "auto", "none", or specific tool
    # OpenAI extended spec — declared so Pydantic stops silently dropping it.
    # When set to False, the route caps the parsed ``tool_calls`` list at
    # length 1 in the response. Default None == True (model may emit
    # multiple). Cannot rely on decoder-level enforcement; this is a
    # post-generation truncation (the only reliable lever absent FSM
    # constraints — see PR #132 / #442 for the decoder-level path).
    parallel_tool_calls: bool | None = None
    # Legacy OpenAI tool-calling shape (pre-1.0 SDK + LangChain compat layers).
    # When set and the modern ``tools``/``tool_choice`` slots are empty, the
    # post-init validator below normalizes them to the modern equivalent so
    # downstream code keeps reading a single shape. Declared so Pydantic
    # stops silently dropping them (same blind-spot family as #355 /
    # #459 / #464). If a client supplies BOTH shapes, modern wins —
    # OpenAI's documented deprecation behavior — and the legacy slots are
    # ignored.
    functions: list[dict] | None = None
    function_call: str | dict | None = None
    # Structured output
    response_format: ResponseFormat | dict | None = None
    # Logprobs
    logprobs: bool | None = None
    top_logprobs: int | None = None  # 0-20, per OpenAI spec
    # OpenAI extended spec — declared so Pydantic stops silently dropping
    # it. Currently rejected with 400 in routes/chat.py if non-empty;
    # mapping to mlx-lm's logits processor is tracked separately.
    logit_bias: dict[str, float] | None = None
    # MLLM-specific parameters
    video_fps: float | None = None
    video_max_frames: int | None = None
    # Request timeout in seconds (None = use server default)
    timeout: float | None = None
    # Thinking/reasoning control (Qwen3 style).  None = server default.
    enable_thinking: bool | None = None
    # Reasoning effort (OpenAI / Anthropic compatible).  None = unset.
    # Valid values: "minimal", "low", "medium", "high", "none".
    reasoning_effort: str | None = None

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v):
        return _validate_finite_in_range(
            v, min_value=0.0, max_value=2.0, field_name="temperature"
        )

    @field_validator("top_p")
    @classmethod
    def _check_top_p(cls, v):
        # OpenAI surface: top_p in [0.0, 1.0] — inclusive min to match the
        # Field(ge=0.0) bound and the committed contract pinned by
        # test_sampling_validation (top_p=0.0 = greedy, legal).
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
        # H-10 root bug: pre-fix no upper cap → penalty > 2 degenerates the
        # distribution. Range matches mlx-lm's non-negative contract + cap.
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

    @field_validator("reasoning_effort")
    @classmethod
    def _validate_reasoning_effort(cls, v: str | None) -> str | None:
        return _validate_reasoning_effort_value(v)

    @field_validator("response_format", mode="before")
    @classmethod
    def _validate_response_format_field(cls, v):
        # Reject silent-200 dict shapes (unknown type, missing schema,
        # non-dict schema, missing name) before Pydantic's Union arm
        # coerces them - pinned by test_response_format_strict_types.py.
        return _validate_response_format_raw(v)

    @field_validator("max_tokens", "max_completion_tokens", mode="before")
    @classmethod
    def _validate_token_budget_field(cls, v, info):
        # Reject bool / non-int / non-positive before Pydantic lax-coerces
        # "100"->100 or True->1 - pinned by TestPositiveIntGenerationBudget.
        return _validate_token_budget(v, info.field_name)

    @field_validator("n", mode="before")
    @classmethod
    def _validate_chat_n(cls, v):
        # F-155: n must equal 1 (no server-side rerank). Mirrors the
        # CompletionRequest guard so both routes reject n!=1 at parse
        # time (422) instead of silently returning one choice (200).
        # mode="before" so bool True is caught before Pydantic coerces
        # it to int 1 (bool is an int subclass).
        return _reject_non_one_n(v)

    # Hard cap on reasoning token budget (set by reasoning_effort tier or
    # explicitly by the client).  None = no cap / server default.
    reasoning_max_tokens: int | None = None
    # OpenAI extended spec: arbitrary kwargs forwarded to the chat template.
    # We currently honor the ``enable_thinking`` key here; other keys are
    # accepted (no Pydantic drop) but not yet forwarded — see
    # ``_resolve_enable_thinking`` in service/helpers.py for precedence.
    chat_template_kwargs: dict | None = None
    # Prefix cache boundary token count — hint from Anthropic cache_control blocks.
    # When set, the engine can reuse cached KV blocks for tokens 0..boundary-1.
    prefix_cache_boundary: int | None = None
    # Number of completions (only n=1 supported)
    n: int | None = None
    # Web search augmentation — when True, the server performs a web search
    # on the user's last message and injects results into the context before
    # inference. Uses DuckDuckGo HTML (no API key required).
    web_search: bool = False

    @model_validator(mode="after")
    def _normalize_max_completion_tokens(self) -> "ChatCompletionRequest":
        if self.max_completion_tokens is not None:
            if (
                self.max_tokens is not None
                and self.max_tokens != self.max_completion_tokens
            ):
                raise ValueError(
                    "Cannot specify both max_tokens and max_completion_tokens with "
                    "different values; use max_completion_tokens only."
                )
            self.max_tokens = self.max_completion_tokens
        return self

    @model_validator(mode="after")
    def _normalize_legacy_functions(self) -> "ChatCompletionRequest":
        """Translate the pre-1.0 ``functions``/``function_call`` shape into
        the modern ``tools``/``tool_choice`` slots so the route never has
        to know about the legacy form. Modern fields take precedence when
        a client supplies both — matches OpenAI's deprecation behavior."""
        if self.functions and self.tools is None:
            self.tools = [
                ToolDefinition(type="function", function=fn) for fn in self.functions
            ]
        if self.function_call is not None and self.tool_choice is None:
            fc = self.function_call
            if isinstance(fc, str):
                # "auto" / "none" map 1:1; anything else passes through and
                # the existing tool_choice handler will 400 on it.
                self.tool_choice = fc
            elif isinstance(fc, dict) and "name" in fc:
                self.tool_choice = {
                    "type": "function",
                    "function": {"name": fc["name"]},
                }
        return self

    @model_validator(mode="after")
    def _validate_tool_schema_depth(self) -> "ChatCompletionRequest":
        if not self.tools:
            return self
        from ..utils.json_depth import (
            json_nesting_depth_exceeds,
            resolve_max_tool_schema_depth,
        )

        max_depth = resolve_max_tool_schema_depth()
        if max_depth <= 0:
            return self
        for tool in self.tools:
            tool_dict = (
                tool.model_dump(exclude_none=True)
                if hasattr(tool, "model_dump")
                else tool
            )
            params = (
                tool_dict.get("function", {}).get("parameters")
                if isinstance(tool_dict, dict)
                else None
            )
            if params and json_nesting_depth_exceeds(params, max_depth):
                raise ValueError(
                    f"Tool schema nesting depth exceeds the {max_depth}-level "
                    f"server cap (set via FUSION_MLX_MAX_TOOL_SCHEMA_DEPTH)."
                )
        return self


class AssistantMessage(BaseModel):
    """Response message from the assistant."""

    role: str = "assistant"
    content: str | None = None
    reasoning_content: str | None = (
        None  # Reasoning/thinking content (when --reasoning-parser is used)
    )
    tool_calls: list[ToolCall] | None = None

    @field_validator("content", mode="before")
    @classmethod
    def _sanitize_content_field(cls, v):
        # R12-MED-2: strip leaked special-token markers (e.g. <|im_start|>)
        # at the type boundary so every AssistantMessage call site — chat
        # route, Responses adapter, Anthropic adapter — funnels through one
        # sanitizer. Pure-markup content collapses to None (drops under
        # exclude_none); mixed text keeps the prose, loses only the marker.
        if v is None or not isinstance(v, str):
            return v
        from .utils import sanitize_output

        return sanitize_output(v)

    @field_validator("reasoning_content", mode="before")
    @classmethod
    def _sanitize_reasoning_content_field(cls, v):
        # R12-MED-2: reasoning_content bypassed the content sanitizer on the
        # tool_choice="required" branch (qwen3 forced-prefix replay dropped a
        # residual <|im_start|> into reasoning_text). Sanitize here so the
        # leak cannot survive regardless of which route constructed the msg.
        # sanitize_reasoning_content None-collapses pure markup (parity with
        # sanitize_output) + identity-preserves plain text.
        if v is None or not isinstance(v, str):
            return v
        from .utils import sanitize_reasoning_content

        return sanitize_reasoning_content(v)

    def model_post_init(self, __context) -> None:
        pass

    @model_serializer(mode="wrap")
    def _serialize(self, handler, info):
        # F-040 / D-MISSING-CONTENT-KEY: ``content`` is a REQUIRED key on
        # OpenAI's chat.completion assistant message wire schema. When the
        # server emits ``content=None`` (reasoning-only / tool-call-only /
        # empty-stop turns), ``exclude_none=True`` drops the key and strict
        # clients (Swift Codable, Rust serde, pydantic-strict) crash with a
        # missing-required-key / KeyError. Serialize ``None`` as ``""`` so
        # the ``string`` type discriminator survives exclude_none.
        # R9-CRIT3: the deprecation-window ``reasoning`` alias of
        # ``reasoning_content`` (R7-H2) was the byte-for-byte root cause of
        # openai-agents text_delta doubling — do NOT re-add it.
        d = handler(self)
        if d.get("content") is None and not (info.mode_is_json() and info.exclude_none):
            # dict path (model_dump): fill before exclude_none stripping
            d["content"] = ""
        elif d.get("content") is None and info.mode_is_json() and info.exclude_none:
            # json path: exclude_none already stripped content — re-add ""
            d["content"] = ""
        return d


class ChatCompletionChoice(BaseModel):
    """A single choice in chat completion response."""

    index: int = 0
    message: AssistantMessage
    finish_reason: str | None = "stop"
    logprobs: ChoiceLogProbs | None = None


class PromptTokensDetails(BaseModel):
    """Breakdown of prompt token usage (OpenAI-compatible)."""

    cached_tokens: int = 0


class CompletionTokensDetails(BaseModel):
    """Breakdown of completion token usage (OpenAI-compatible)."""

    reasoning_tokens: int = 0


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    completion_tokens_details: CompletionTokensDetails | None = None
    prompt_tokens_details: PromptTokensDetails | None = None


OPENAI_REASONING_EFFORT_TO_MAX_TOKENS: dict[str, int] = {
    "minimal": 256,
    "low": 512,
    "medium": 2048,
    "high": 8192,
}


class ChatCompletionResponse(BaseModel):
    """Response for chat completion."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# =============================================================================
# Text Completion
# =============================================================================


class CompletionRequest(BaseModel):
    """Request for text completion."""

    model: str = "default"
    prompt: str | list[str]
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(None, ge=1, le=131072)
    stream: bool = False
    stop: list[str] | None = None
    # Extended OpenAI-compatible sampling parameters — see #355 + the
    # matching block on ChatCompletionRequest for wiring + caveats.
    top_k: int | None = Field(None, ge=0)
    min_p: float | None = Field(None, ge=0.0, le=1.0)
    repetition_penalty: float | None = Field(None, ge=0.0)
    presence_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(None, ge=-2.0, le=2.0)
    # OpenAI ``n``/``best_of`` - declared so Pydantic stops silently
    # dropping them; rejected with 400 in routes/completions.py when >1
    # since we generate one completion per request (no server-side rerank).
    n: int | None = Field(None, ge=1, le=128)
    best_of: int | None = Field(None, ge=1, le=128)
    # OpenAI ``echo``/``response_format``/``stream_options`` - declared
    # so Pydantic stops silently dropping them; consumed in
    # routes/completions.py (echo gates the fence-strip + prompt logprobs,
    # response_format selects JSON cleanup, stream_options gates usage SSE).
    echo: bool | None = None
    response_format: ResponseFormat | dict | None = None
    stream_options: StreamOptions | None = None
    # Logprobs
    logprobs: int | None = None  # 0-5, per OpenAI legacy completions spec
    top_logprobs: int | None = None  # 0-20, per OpenAI spec
    # OpenAI FIM (fill-in-the-middle) suffix. Declared so Pydantic stops
    # silently dropping it; rejected with 400 in routes/completions.py
    # when non-empty since no MLX engine implements FIM yet (and silently
    # ignoring it produces wrong completions on code-completion clients).
    suffix: str | None = None
    # Request timeout in seconds (None = use server default)
    timeout: float | None = None

    @field_validator("response_format", mode="before")
    @classmethod
    def _validate_response_format_field(cls, v):
        # R10-H4: /v1/completions response_format parity with the chat
        # lane. Pre-fix the field was declared but UNVALIDATED -> a dict
        # like {"type":"xml"} or {} parsed silently into ResponseFormat
        # (type is a bare str with no enum). Closed-set check rejects
        # unknown type + missing type + malformed json_schema before the
        # route applies the JSON peel. Pinned by test_r10_h4.
        return _validate_response_format_raw(v)

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

    @field_validator("n", mode="before")
    @classmethod
    def _validate_completion_n(cls, v):
        # F-152: pin the wire schema - ``n`` must equal 1. Pydantic raises
        # here (422 at the schema layer) before the route's own 400 envelope
        # runs; the production server still rewrites 422->400, but the raw
        # contract is "one completion per request, no server-side rerank".
        # mode="before" so this fires ahead of the Field(ge=1, le=128)
        # constraint and every illegal n surfaces the "must equal 1" message
        # (0/-1 fail ge=1, 1000 fails le=128 - both would mask the real rule).
        return _reject_non_one_n(v)

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _validate_token_budget_field(cls, v, info):
        # Reject bool / non-int / non-positive before lax coercion -
        # pinned by TestPositiveIntGenerationBudget (cross-route parity).
        return _validate_token_budget(v, info.field_name)

    @field_validator("logprobs", mode="before")
    @classmethod
    def _validate_completion_logprobs_type(cls, v):
        # ``logprobs`` is an integer 0-5 on the legacy completions spec.
        # bool is a subclass of int, so without a before-mode guard Pydantic
        # silently coerces True->1 - reject it explicitly with a message that
        # names the integer expectation (not the opaque bool_parsing error).
        if v is None:
            return v
        if isinstance(v, bool):
            raise ValueError("logprobs must be an integer (0-5), got boolean")
        if not isinstance(v, int):
            raise ValueError("logprobs must be an integer (0-5)")
        return v


class CompletionChoice(BaseModel):
    """A single choice in text completion response."""

    index: int = 0
    text: str
    finish_reason: str | None = "stop"
    # Legacy /v1/completions logprobs are the OpenAI 4-array shape
    # (tokens/token_logprobs/top_logprobs/text_offset), NOT the chat
    # ``ChoiceLogProbs.content`` shape. Forward-ref as a string because
    # ``LegacyCompletionLogProbs`` is defined below this class and the
    # module does not use ``from __future__ import annotations``.
    logprobs: "LegacyCompletionLogProbs | None" = None


class LegacyCompletionLogProbs(BaseModel):
    tokens: list[str]
    token_logprobs: list[float]
    top_logprobs: list[dict[str, float]]
    text_offset: list[int]


class CompletionResponse(BaseModel):
    """Response for text completion."""

    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:8]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# =============================================================================
# Models List
# =============================================================================


class ModelInfo(BaseModel):
    """Information about an available model."""

    model_config = ConfigDict(extra="ignore")

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "rapid-mlx"
    modality: str | None = None
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

    tool_name: str
    arguments: dict = Field(default_factory=dict)


class MCPExecuteResponse(BaseModel):
    """Response from executing an MCP tool."""

    tool_name: str
    content: str | list | dict | None = None
    is_error: bool = False
    error_message: str | None = None


# =============================================================================
# Audio (STT/TTS)
# =============================================================================


class AudioTranscriptionRequest(BaseModel):
    """Request for audio transcription (STT)."""

    model: str = "whisper-large-v3"
    language: str | None = None
    response_format: str = "json"
    temperature: float = 0.0
    timestamp_granularities: list[str] | None = None


class AudioTranscriptionResponse(BaseModel):
    """Response from audio transcription."""

    text: str
    language: str | None = None
    duration: float | None = None
    segments: list[dict] | None = None


_ALLOWED_AUDIO_FORMATS = ("wav", "pcm", "flac", "ogg", "opus", "mp3")


class AudioSpeechRequest(BaseModel):
    """Request for text-to-speech."""

    model: str = "kokoro"
    input: str
    voice: str = "af_heart"
    speed: float = 1.0
    response_format: str = "wav"

    @model_validator(mode="before")
    @classmethod
    def _fold_format_alias(cls, values):
        # R11-B-F2 (#505): legacy ``format`` field folds into
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


class AudioSeparationRequest(BaseModel):
    """Request for audio source separation."""

    model: str = "htdemucs"
    stems: list[str] = Field(default_factory=lambda: ["vocals", "accompaniment"])


# =============================================================================
# Embeddings
# =============================================================================


class EmbeddingRequest(BaseModel):
    """Request for text embeddings (OpenAI compatible)."""

    # extra="forbid" turns silent-drop into a 422 with a clear field name.
    # Without it, fields like `dimensions` or `encoding_format` typos pass
    # through and the user only notices when the response shape is wrong.
    # protected_namespaces=() suppresses the Pydantic v2 warning about
    # the `model` field colliding with the reserved `model_` prefix; a
    # future Pydantic point release could otherwise promote that warning
    # to an error and 500 every embeddings request.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # OpenAI spec lists 4 input shapes: ``str``, ``list[str]``,
    # ``list[int]`` (single pre-tokenized input), and
    # ``list[list[int]]`` (batch of pre-tokenized inputs). Production
    # pipelines that pre-tokenize with a shared HF tokenizer send the
    # latter two forms — refusing them broke LangChain / LlamaIndex
    # integrations that hard-code the spec shape (R10 sweep H6).
    #
    # ``StrictInt`` / ``StrictStr`` so Pydantic does NOT silently
    # coerce ``"123"`` → 123 (would be treated as token id 123, a
    # different embedding from the word "123") or ``True`` → 1
    # (Python ``bool`` is an ``int`` subclass; without ``StrictInt``
    # a JSON ``true`` would pass as token id 1).
    input: StrictStr | list[StrictStr] | list[StrictInt] | list[list[StrictInt]]
    model: str
    # Literal so an unknown value (typo like "base65" or "BASE64") 422s
    # at parse time rather than silently falling back to float — that
    # silent fallback is the same class of bug this PR exists to close.
    encoding_format: Literal["float", "base64"] | None = "float"
    # OpenAI spec: per-vector truncation. Common for MRL-style models
    # (text-embedding-3-large, nomic-embed-text-v1.5). Implemented in
    # the route as a post-embed slice + L2 renormalization (required
    # for the truncated vector to remain a valid embedding for cosine
    # similarity per the OpenAI cookbook).
    dimensions: int | None = None
    # OpenAI abuse-tracking field. Accepted (not validated) so clients
    # using the upstream SDK don't see a 422 on unknown field.
    user: str | None = None


class EmbeddingData(BaseModel):
    """A single embedding result."""

    object: str = "embedding"
    index: int
    # `list[float]` for encoding_format="float"; base64-encoded float32
    # little-endian bytes (as ASCII string) for encoding_format="base64".
    embedding: list[float] | str


class EmbeddingUsage(BaseModel):
    """Token usage for embedding requests."""

    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    """Response for embeddings endpoint (OpenAI compatible)."""

    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage = Field(default_factory=EmbeddingUsage)


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
        # R12-MED-2 streaming sibling: deltas are concatenated verbatim by
        # clients, so the sanitizer MUST preserve surrounding whitespace
        # (sanitize_reasoning_for_stream removes ONLY the marker bytes, no
        # .strip()). Pure-marker delta collapses to "" then None so the
        # field drops out cleanly; "" -> None keeps the None-vs-empty
        # contract stable across the content/reasoning channels.
        if v is None or not isinstance(v, str):
            return v
        from .utils import sanitize_reasoning_for_stream

        out = sanitize_reasoning_for_stream(v)
        return out or None

    @field_validator("reasoning_content", mode="before")
    @classmethod
    def _sanitize_reasoning_delta(cls, v):
        # R12-MED-2: the streaming hot path (_fast_sse_chunk) and the
        # logprobs pydantic path must agree byte-for-byte. Whitespace-
        # preserving sanitizer so "foo" + " bar <|im_start|>" concatenates
        # to "foo bar " not "foobar". Pure markup -> "" -> None.
        if v is None or not isinstance(v, str):
            return v
        from .utils import sanitize_reasoning_for_stream

        out = sanitize_reasoning_for_stream(v)
        return out or None

    @model_serializer(mode="wrap")
    def _serialize(self, handler, info):
        # F-040 / D-MISSING-CONTENT-KEY (streaming sibling): a terminal
        # chunk delta carrying reasoning_content / tool_calls but no content
        # must expose ``content: ""`` so clients reading
        # ``chunk.choices[0].delta.content`` on the last chunk do not crash
        # with a missing-required-key under exclude_none. A truly empty
        # finish-marker delta (no payload at all) stays ``{}`` — do not
        # pollute pure finish chunks with a spurious content key.
        d = handler(self)
        if d.get("content") is None and (
            self.reasoning_content is not None or self.tool_calls is not None
        ):
            d["content"] = ""
        return d


class ChatCompletionChunkChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None
    logprobs: ChoiceLogProbs | None = None


class ChatCompletionChunk(BaseModel):
    """A streaming chunk for chat completion."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: Usage | None = None  # Included when stream_options.include_usage=true
