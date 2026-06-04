# SPDX-License-Identifier: Apache-2.0
"""Tokenizer detection utilities."""


def is_gemma4_model(model_path: str) -> bool:
    """Check if the model is a Gemma 4 variant."""
    return "gemma" in model_path.lower() and "4" in model_path.lower()


def is_harmony_model(model_path: str) -> bool:
    """Check if the model is a Harmony variant."""
    return "harmony" in model_path.lower()
