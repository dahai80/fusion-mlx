# SPDX-License-Identifier: Apache-2.0
"""
API models, utilities, and tool calling support for oMLX.

This module provides shared components used by the server:
- Pydantic models for OpenAI-compatible API
- Pydantic models for Anthropic Messages API
- Utility functions for text processing
- Tool calling parsing and conversion
"""

from .anthropic_models import (
    AnthropicErrorDetail,
    # Error response
    AnthropicErrorResponse,
    AnthropicMessage,
    AnthropicTool,
    AnthropicUsage,
    ContentBlockDeltaEvent,
    ContentBlockImage,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    # Content blocks
    ContentBlockText,
    ContentBlockToolResult,
    ContentBlockToolUse,
    ErrorEvent,
    InputJsonDelta,
    MessageDeltaEvent,
    # Streaming events
    MessageStartEvent,
    MessageStopEvent,
    PingEvent,
    # Messages
    SystemContent,
    TextDelta,
    ThinkingConfig,
    # Token counting
    TokenCountRequest,
    TokenCountResponse,
    ToolChoice,
)
from .anthropic_models import (
    # Request/Response
    MessagesRequest as AnthropicMessagesRequest,
)
from .anthropic_models import (
    MessagesResponse as AnthropicMessagesResponse,
)
from .anthropic_utils import (
    convert_anthropic_to_internal,
    convert_anthropic_tools_to_internal,
    convert_internal_to_anthropic_response,
    create_content_block_start_event,
    create_content_block_stop_event,
    create_error_event,
    create_input_json_delta_event,
    create_message_delta_event,
    create_message_start_event,
    create_message_stop_event,
    create_ping_event,
    create_text_delta_event,
    format_sse_event,
    map_finish_reason_to_stop_reason,
)
from .embedding_models import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from .embedding_utils import (
    count_tokens,
    encode_embedding_base64,
    normalize_input,
    truncate_embedding,
)

# MCP routes
from .mcp_routes import router as mcp_router
from .mcp_routes import set_mcp_manager_getter
from .openai_models import (
    AssistantMessage,
    ChatCompletionChoice,
    # Chat requests/responses
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    # Completion requests/responses
    CompletionRequest,
    CompletionResponse,
    # Content types
    ContentPart,
    # Tool calling
    FunctionCall,
    MCPExecuteRequest,
    MCPExecuteResponse,
    MCPServerInfo,
    MCPServersResponse,
    # MCP
    MCPToolInfo,
    MCPToolsResponse,
    Message,
    ModelInfo,
    ModelsResponse,
    # Structured output
    ResponseFormat,
    ResponseFormatJsonSchema,
    ToolCall,
    ToolDefinition,
    # Common
    Usage,
)
from .thinking import (
    ThinkingParser,
    extract_thinking,
)
from .tool_calling import (
    build_json_system_prompt,
    convert_tools_for_template,
    extract_json_from_text,
    # Structured output
    parse_json_output,
    parse_tool_calls,
    validate_json_schema,
)
from .utils import (
    SPECIAL_TOKENS_PATTERN,
    clean_output_text,
    clean_special_tokens,
    extract_text_content,
)

__all__ = [
    # Models
    "ContentPart",
    "Message",
    "FunctionCall",
    "ToolCall",
    "ToolDefinition",
    "ResponseFormat",
    "ResponseFormatJsonSchema",
    "ChatCompletionRequest",
    "ChatCompletionChoice",
    "ChatCompletionResponse",
    "AssistantMessage",
    "CompletionRequest",
    "CompletionChoice",
    "CompletionResponse",
    "Usage",
    "ModelInfo",
    "ModelsResponse",
    "MCPToolInfo",
    "MCPToolsResponse",
    "MCPServerInfo",
    "MCPServersResponse",
    "MCPExecuteRequest",
    "MCPExecuteResponse",
    # Utils
    "clean_output_text",
    "clean_special_tokens",
    "extract_text_content",
    "SPECIAL_TOKENS_PATTERN",
    # Thinking
    "ThinkingParser",
    "extract_thinking",
    # Tool calling
    "parse_tool_calls",
    "convert_tools_for_template",
    # Structured output
    "parse_json_output",
    "validate_json_schema",
    "extract_json_from_text",
    "build_json_system_prompt",
    # Anthropic models
    "ContentBlockText",
    "ContentBlockImage",
    "ContentBlockToolUse",
    "ContentBlockToolResult",
    "SystemContent",
    "AnthropicMessage",
    "AnthropicTool",
    "ToolChoice",
    "ThinkingConfig",
    "AnthropicMessagesRequest",
    "AnthropicMessagesResponse",
    "AnthropicUsage",
    "TokenCountRequest",
    "TokenCountResponse",
    "MessageStartEvent",
    "ContentBlockStartEvent",
    "ContentBlockDeltaEvent",
    "ContentBlockStopEvent",
    "MessageDeltaEvent",
    "MessageStopEvent",
    "PingEvent",
    "ErrorEvent",
    "TextDelta",
    "InputJsonDelta",
    "AnthropicErrorResponse",
    "AnthropicErrorDetail",
    # Anthropic utils
    "convert_anthropic_to_internal",
    "convert_anthropic_tools_to_internal",
    "convert_internal_to_anthropic_response",
    "map_finish_reason_to_stop_reason",
    "format_sse_event",
    "create_message_start_event",
    "create_content_block_start_event",
    "create_text_delta_event",
    "create_input_json_delta_event",
    "create_content_block_stop_event",
    "create_message_delta_event",
    "create_message_stop_event",
    "create_ping_event",
    "create_error_event",
    # Embedding models
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingData",
    "EmbeddingUsage",
    # Embedding utils
    "encode_embedding_base64",
    "truncate_embedding",
    "count_tokens",
    "normalize_input",
    # MCP routes
    "mcp_router",
    "set_mcp_manager_getter",
]
