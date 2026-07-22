# Vendored from stefanopineda/dspark-metal (MIT) into fusion-mlx and evolved
# independently. Upstream has been dormant (no commit/issue activity for 20+
# days), so fusion-mlx carries its own copy under
# fusion_mlx/speculative/dspark/engine/ rather than depending on a pip package.
# See LICENSE and NOTICE alongside this file for upstream attribution.
# Includes PR#2 (Qwen3VLTargetAdapter, Direction B native multimodal) which
# extends DSpark speculative decoding to mlx-vlm targets on text positions.
"""MLX runtime for DSpark speculative decoding on Apple Silicon."""

from .adapters import LoadedTargetModel, adapter_for_model_type, load_target_model
from .api import DSparkGenerator, DSparkResult, DSparkStreamEvent
from .draft import DSparkDraftModel, load_draft_model
from .heads import ConfidenceHead, MarkovHead
from .runtime import (
    dspark_generate,
    dspark_generate_stream,
    longest_prefix_match,
    sample_tokens,
)
from .sampling import (
    KeySequence,
    logits_to_probs,
    sample_from_logits,
    sample_from_probs,
    sample_residual,
    speculative_accept,
)
from .sts import apply_sts, fit_sts, load_sts_temperatures, save_sts_temperatures

__all__ = [
    "DSparkGenerator",
    "apply_sts",
    "fit_sts",
    "load_sts_temperatures",
    "save_sts_temperatures",
    "DSparkResult",
    "DSparkStreamEvent",
    "DSparkDraftModel",
    "ConfidenceHead",
    "KeySequence",
    "MarkovHead",
    "LoadedTargetModel",
    "adapter_for_model_type",
    "dspark_generate",
    "dspark_generate_stream",
    "load_draft_model",
    "load_target_model",
    "logits_to_probs",
    "longest_prefix_match",
    "sample_from_logits",
    "sample_from_probs",
    "sample_residual",
    "sample_tokens",
    "speculative_accept",
]
