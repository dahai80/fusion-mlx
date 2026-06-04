# SPDX-License-Identifier: Apache-2.0
"""Model alias definitions for fusion-mlx."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AliasProfile:
    name: str
    hf_path: str
    supports_dflash: bool = False
    is_moe: bool = False
    drafter_hf_path: Optional[str] = None
    description: str = ""


def list_profiles() -> List[AliasProfile]:
    return []
