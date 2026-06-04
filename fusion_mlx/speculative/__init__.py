"""mlx-spec-decode — Speculative decoding drafters.

Provides SuffixDecoding, DFlash, MTP, PLD, and VLM-MTP drafters.
Merged from omlx/speculative + Rapid-MLX speculative implementations.
"""

from .vlm_mtp import VLMMTPDrafter, load_vlm_mtp_drafter, run_vlm_mtp_decode
from .suffix_decoding import SuffixDecodingDrafter, DraftStats
from .prompt_lookup import PromptLookupDecoder, prompt_lookup_generate_step
from .mtp_generate import MTPStats, MTPOutput, mtp_generate_step
from . import dflash

__all__ = [
    "VLMMTPDrafter",
    "load_vlm_mtp_drafter",
    "run_vlm_mtp_decode",
    "SuffixDecodingDrafter",
    "DraftStats",
    "PromptLookupDecoder",
    "prompt_lookup_generate_step",
    "MTPStats",
    "MTPOutput",
    "mtp_generate_step",
    "dflash",
]
