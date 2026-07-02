from typing import Any, Literal

from pydantic import BaseModel, Field

# =============================================================================
# Pydantic Models
# =============================================================================


class LoginRequest(BaseModel):
    """Request model for admin login."""

    api_key: str
    remember: bool = False


class SetupApiKeyRequest(BaseModel):
    """Request model for initial API key setup."""

    api_key: str
    api_key_confirm: str


class CreateSubKeyRequest(BaseModel):
    """Request model for creating a sub API key."""

    key: str
    name: str = ""


class DeleteSubKeyRequest(BaseModel):
    """Request model for deleting a sub API key."""

    key: str


class CacheProbeRequest(BaseModel):
    """Request model for probing per-prompt cache state.

    Tokenizes a chat message list with the target model's tokenizer, then
    classifies each block's location in the cache hierarchy:
    - Hot SSD (in-RAM copy of SSD cache, ready to mount without disk read)
    - Disk SSD (persisted only, needs disk read to reuse)
    - Cold (fully uncached — would require full prefill)
    """

    model_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    chat_template_kwargs: dict[str, Any] | None = None


class ModelSettingsRequest(BaseModel):
    """Request model for updating per-model settings."""

    model_alias: str | None = None
    model_type_override: str | None = None
    max_context_window: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    force_sampling: bool | None = None
    max_tool_result_tokens: int | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    forced_ct_kwargs: list[str] | None = None
    ttl_seconds: int | None = None
    index_cache_freq: int | None = None
    enable_thinking: bool | None = None
    thinking_budget_enabled: bool | None = None
    thinking_budget_tokens: int | None = None
    # TurboQuant KV cache (mlx-vlm backend)
    turboquant_kv_enabled: bool | None = None
    turboquant_kv_bits: float | None = None
    # SpecPrefill (experimental)
    specprefill_enabled: bool | None = None
    specprefill_draft_model: str | None = None
    specprefill_keep_pct: float | None = None
    specprefill_threshold: int | None = None
    # DFlash (block diffusion speculative decoding)
    dflash_enabled: bool | None = None
    dflash_draft_model: str | None = None
    dflash_draft_quant_enabled: bool | None = None
    dflash_draft_quant_weight_bits: int | None = None
    dflash_draft_quant_activation_bits: int | None = None
    dflash_draft_quant_group_size: int | None = None
    dflash_max_ctx: int | None = None
    dflash_in_memory_cache: bool | None = None
    dflash_in_memory_cache_max_entries: int | None = None
    dflash_in_memory_cache_max_bytes: int | None = None
    dflash_ssd_cache: bool | None = None
    dflash_ssd_cache_max_bytes: int | None = None
    dflash_draft_window_size: int | None = None
    dflash_draft_sink_size: int | None = None
    dflash_verify_mode: str | None = None
    # Native MTP (mlx-lm PR 990 / PR 15 monkey-patch)
    mtp_enabled: bool | None = None
    # VLM MTP speculative decoding via external assistant drafter (mlx-vlm 191d7c8+)
    vlm_mtp_enabled: bool | None = None
    vlm_mtp_draft_model: str | None = None
    vlm_mtp_draft_block_size: int | None = None
    reasoning_parser: str | None = None
    is_pinned: bool | None = None
    is_default: bool | None = None
    # Security: per-model opt-in for trust_remote_code (issue #926)
    trust_remote_code: bool | None = None


class CreateProfileRequest(BaseModel):
    """Request body for creating a per-model profile."""
    name: str
    display_name: str
    description: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    also_save_as_template: bool = False
    source_template: str | None = None


class UpdateProfileRequest(BaseModel):
    """Request body for updating/renaming a per-model profile."""
    new_name: str | None = None
    display_name: str | None = None
    description: str | None = None
    settings: dict[str, Any] | None = None
    source_template: str | None = None
    also_save_as_template: bool = False


class CreateTemplateRequest(BaseModel):
    """Request body for creating a global template."""
    name: str
    display_name: str
    description: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class UpdateTemplateRequest(BaseModel):
    """Request body for updating/renaming a global template."""
    new_name: str | None = None
    display_name: str | None = None
    description: str | None = None
    settings: dict[str, Any] | None = None


class GlobalSettingsRequest(BaseModel):
    """Request model for updating global server settings."""

    # Server settings
    host: str | None = None
    port: int | None = None
    log_level: str | None = None
    server_aliases: list[str] | None = None
    sse_keepalive_mode: str | None = None

    # Model settings
    model_dirs: list[str] | None = None
    model_dir: str | None = None  # Deprecated: kept for backward compatibility
    model_fallback: bool | None = None

    # Memory enforcement
    memory_prefill_memory_guard: bool | None = None
    memory_guard_tier: str | None = None  # "safe" / "balanced" / "aggressive" / "custom"
    memory_guard_custom_ceiling_gb: float | None = None  # only used when tier == "custom"

    # Scheduler settings
    max_concurrent_requests: int | None = None
    embedding_batch_size: int | None = None
    chunked_prefill: bool | None = None

    # Cache settings
    cache_enabled: bool | None = None
    ssd_cache_dir: str | None = None
    ssd_cache_max_size: str | None = None
    hot_cache_only: bool | None = None
    hot_cache_max_size: str | None = None  # "0" = disabled, "8GB", etc.
    initial_cache_blocks: int | None = None  # Starting blocks (requires restart)

    # MCP settings
    mcp_config: str | None = None

    # HuggingFace settings
    hf_endpoint: str | None = None

    # ModelScope settings
    ms_endpoint: str | None = None

    # Network settings
    network_http_proxy: str | None = None
    network_https_proxy: str | None = None
    network_no_proxy: str | None = None
    network_ca_bundle: str | None = None

    # Sampling defaults
    sampling_max_context_window: int | None = None
    sampling_max_tokens: int | None = None
    sampling_temperature: float | None = None
    sampling_top_p: float | None = None
    sampling_top_k: int | None = None
    sampling_repetition_penalty: float | None = None

    # Claude Code settings
    claude_code_context_scaling_enabled: bool | None = None
    claude_code_target_context_size: int | None = None
    claude_code_mode: str | None = None
    claude_code_opus_model: str | None = None
    claude_code_sonnet_model: str | None = None
    claude_code_haiku_model: str | None = None

    # Other integrations settings
    integrations_copilot_model: str | None = None
    integrations_codex_model: str | None = None
    integrations_opencode_model: str | None = None
    integrations_openclaw_model: str | None = None
    integrations_hermes_model: str | None = None
    integrations_pi_model: str | None = None
    integrations_openclaw_tools_profile: Literal["minimal", "coding", "messaging", "full"] | None = None

    # UI settings
    ui_language: str | None = None

    # Idle timeout settings. null disables the global fallback.
    idle_timeout_seconds: int | None = Field(default=None, ge=60)

    # Auth settings
    api_key: str | None = None
    skip_api_key_verification: bool | None = None


class HFDownloadRequest(BaseModel):
    """Request model for starting a HuggingFace model download."""

    repo_id: str
    hf_token: str = ""


class HFRetryRequest(BaseModel):
    """Request model for retrying a HuggingFace model download."""

    hf_token: str = ""


class MSDownloadRequest(BaseModel):
    """Request model for starting a ModelScope model download."""

    model_id: str
    ms_token: str = ""


class MSRetryRequest(BaseModel):
    """Request model for retrying a ModelScope model download."""

    ms_token: str = ""


class OQStartRequest(BaseModel):
    """Request model for starting an oQ quantization task."""

    model_path: str
    oq_level: float = 0
    group_size: int = 64
    sensitivity_model_path: str = ""
    text_only: bool = False
    dtype: str = "bfloat16"
    preserve_mtp: bool = False
    auto_proxy_sensitivity: bool = True
    recipe: str = ""


class HFUploadRequest(BaseModel):
    """Request model for starting a HuggingFace upload task."""

    model_path: str
    repo_id: str
    hf_token: str
    readme_source_path: str = ""
    auto_readme: bool = True
    redownload_notice: bool = False
    private: bool = False


class HFValidateTokenRequest(BaseModel):
    """Request model for validating a HuggingFace token."""

    hf_token: str

