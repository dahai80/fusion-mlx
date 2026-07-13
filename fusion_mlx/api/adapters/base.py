# SPDX-License-Identifier: Apache-2.0
"""
Base adapter interface for API format conversion.

This module defines the abstract interface that all API adapters must implement,
plus internal data structures for request/response handling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class InternalMessage:
    """Internal representation of a chat message."""

    role: str
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


@dataclass
class InternalRequest:
    """
    Internal request format used by the inference engine.

    This provides a unified format that all adapters convert to/from.
    """

    # Required fields
    messages: list[InternalMessage]

    # Generation parameters
    max_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stream: bool = False

    # Stop conditions
    stop: list[str] | None = None
    stop_token_ids: list[int] | None = None

    # Tool calling
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None

    # Response format
    response_format: dict[str, Any] | None = None

    # Model
    model: str | None = None

    # Metadata
    request_id: str | None = None


@dataclass
class InternalResponse:
    """
    Internal response format from the inference engine.

    This provides a unified format that all adapters convert from.
    """

    # Generated content
    text: str
    finish_reason: str | None = None
    reasoning_content: str | None = None

    # Token counts
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    # Tool calls (parsed)
    tool_calls: list[dict[str, Any]] | None = None

    # Metadata
    request_id: str | None = None
    model: str | None = None

    # Logprobs (populated only when requested; None until engine plumbing lands)
    logprobs: Any = None


@dataclass
class StreamChunk:
    """A single chunk in a streaming response."""

    text: str = ""
    reasoning_content: str | None = None
    finish_reason: str | None = None
    tool_call_delta: dict[str, Any] | None = None
    is_first: bool = False
    is_last: bool = False

    # Token counts (usually only on last chunk)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    # Logprobs (populated only when requested; None until engine plumbing lands)
    logprobs: Any = None


class BaseAdapter(ABC):
    """
    Abstract base class for API adapters.

    Adapters handle conversion between external API formats (OpenAI, Anthropic)
    and the internal request/response format used by the inference engine.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the adapter name (e.g., 'openai', 'anthropic')."""
        pass

    @abstractmethod
    def parse_request(self, request: Any) -> InternalRequest:
        """
        Convert an external API request to internal format.

        Args:
            request: The external API request object.

        Returns:
            InternalRequest in unified format.
        """
        pass

    @abstractmethod
    def format_response(
        self,
        response: InternalResponse,
        request: Any,
    ) -> Any:
        """
        Convert an internal response to external API format.

        Args:
            response: The internal response object.
            request: The original external request (for context).

        Returns:
            Response in the external API format.
        """
        pass

    @abstractmethod
    def format_stream_chunk(
        self,
        chunk: StreamChunk,
        request: Any,
    ) -> str:
        """
        Format a streaming chunk for SSE output.

        Args:
            chunk: The stream chunk to format.
            request: The original external request (for context).

        Returns:
            SSE-formatted string.
        """
        pass

    @abstractmethod
    def format_stream_end(self, request: Any) -> str:
        """
        Format the stream end marker.

        Args:
            request: The original external request (for context).

        Returns:
            SSE-formatted end marker.
        """
        pass

    @abstractmethod
    def create_error_response(
        self,
        error: str,
        error_type: str = "server_error",
        status_code: int = 500,
    ) -> dict:
        """
        Create an error response in the adapter's format.

        Args:
            error: Error message.
            error_type: Type of error (e.g., "invalid_request_error").
            status_code: HTTP status code.

        Returns:
            Error response dict in the adapter's format.
        """
        pass
