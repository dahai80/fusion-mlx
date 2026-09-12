# SPDX-License-Identifier: Apache-2.0
"""Centralized error-code → remediation suggestions registry.

Single source for the ``solutions`` field surfaced in API error responses.
All middleware handlers + drain paths reference this module instead of
assembling suggestion lists inline. R-6 (#0910 audit): ops auditability +
consistent client guidance across status codes.

Extends the original 413/503/507 set with 429 (rate limit), 504 (gateway
timeout), 500 (internal error — generic triage hints), plus 400/401/403/404/408/502
to cover the 12 most common client-facing error classes (enhance-0911 S1.5).
"""

import logging

logger = logging.getLogger(__name__)

ERROR_SOLUTIONS_MAP: dict[int, list[str]] = {
    400: [
        "Check the request body against the OpenAI/Anthropic API schema",
        "Verify required fields (model, messages) are present and well-formed",
        "Inspect the error.detail field for the specific validation failure",
    ],
    401: [
        "Set FUSION_MLX_API_KEY in settings.json or pass Authorization: Bearer <key>",
        "Verify the API key matches the server's configured key",
        "If using a proxy, ensure it forwards the Authorization header",
    ],
    403: [
        "Check if the requested engine/modality is disabled by the active profile",
        "Run `fusion-mlx doctor` to see the active profile and disabled modules",
        "Switch profile (lite/standard/full) in settings.json to enable the engine",
    ],
    404: [
        "Verify the model alias with `fusion-mlx models`",
        "Use `fusion-mlx pull <model>` to download the model first",
        "Check for typos in the model id; aliases are case-sensitive",
    ],
    408: [
        "Increase client-side timeout to exceed prefill duration for long prompts",
        "Reduce max_context or shorten the prompt",
        "Switch to streaming to receive the first token sooner",
    ],
    413: [
        "Reduce max_context or shorten the prompt",
        "Use a smaller quantization (e.g. 4bit instead of 8bit)",
        "Set profile=lite in settings.json to reduce mounted engines",
    ],
    429: [
        "Slow down request rate; honor Retry-After header",
        "Reduce concurrency (fewer parallel streams / agents)",
        "Raise FUSION_MAX_CONCURRENT_REQUESTS in settings.json if capacity allows",
    ],
    500: [
        "Check `fusion-mlx log` for the stack trace",
        "Retry once — transient engine faults (CUDA/Metal) often clear",
        "If reproducible, report with the request_id from the error body",
    ],
    503: [
        "Retry after a short backoff (Retry-After header)",
        "Reduce --max-concurrent-requests to lower queue depth",
        "Check if another model is consuming memory with `fusion-mlx ps`",
    ],
    504: [
        "Increase client-side timeout to exceed generation duration",
        "Reduce max_tokens or switch to streaming to receive first token sooner",
        "Check `fusion-mlx status` for slow-model / queue-depth buildup",
    ],
    502: [
        "Check if a cloud-fallback router is configured and reachable",
        "Retry — upstream gateway faults are often transient",
        "Inspect `fusion-mlx log` for the forwarded request's failure reason",
    ],
    507: [
        "Reduce max_tokens for the request",
        "Use a smaller quantization to free KV cache memory",
        "Unload other models via the /v1/models admin endpoint",
    ],
}


def get_solutions(status_code: int) -> list[str]:
    try:
        return ERROR_SOLUTIONS_MAP.get(status_code, [])
    except Exception:
        logger.debug("solutions lookup failed for status %s", status_code)
        return []
