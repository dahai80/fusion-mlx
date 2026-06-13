# SPDX-License-Identifier: Apache-2.0
"""Model alias definitions for fusion-mlx."""

from dataclasses import dataclass


@dataclass
class AliasProfile:
    name: str
    hf_path: str
    supports_dflash: bool = False
    is_moe: bool = False
    drafter_hf_path: str | None = None
    description: str = ""


def list_profiles() -> list[AliasProfile]:
    return []
