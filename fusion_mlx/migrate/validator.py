# SPDX-License-Identifier: Apache-2.0
"""Inference validation runner — loads converted MLX model and runs sanity check.

Callers: fusion_mlx.admin.migrate_route
API: validate_model(model_dir, prompt, max_tokens) -> ValidationResult
Schema: ValidationResult(dataclass) — success, output_text, tokens_per_sec, num_tokens, error
User instruction verbatim: "做一个端到端的功能，做模型迁移和量化的功能，以openpangu为例，把迁移的每个步骤展现在GUI上"
"""

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    success: bool = False
    output_text: str = ""
    tokens_per_sec: float = 0.0
    num_tokens: int = 0
    error: str | None = None


def validate_model(
    model_dir: str,
    prompt: str = "Hello, how are you?",
    max_tokens: int = 32,
) -> ValidationResult:
    result = ValidationResult()

    try:
        from mlx_lm import generate, load

        logger.info("Loading model from %s", model_dir)
        model, tokenizer = load(model_dir)

        logger.info("Generating with prompt: %s (max_tokens=%d)", prompt, max_tokens)
        t0 = time.time()
        output = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        elapsed = time.time() - t0

        result.output_text = output
        result.num_tokens = len(tokenizer.encode(output))
        result.tokens_per_sec = result.num_tokens / elapsed if elapsed > 0 else 0.0
        result.success = True
        logger.info(
            "Validation OK: %d tokens in %.2fs (%.1f tok/s)",
            result.num_tokens, elapsed, result.tokens_per_sec,
        )

    except ImportError:
        result.error = "mlx_lm not installed — cannot validate"
        logger.error(result.error)
    except Exception as e:
        logger.exception("Validation failed: %s", e)
        result.error = str(e)

    return result
