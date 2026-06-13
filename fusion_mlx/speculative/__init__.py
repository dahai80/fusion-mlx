"""mlx-spec-decode — Speculative decoding drafters.

Provides SuffixDecoding, DFlash, MTP, PLD, and VLM-MTP drafters.
Merged from omlx/speculative + Rapid-MLX speculative implementations.
"""

from . import dflash
from .mtp_generate import MTPOutput, MTPStats, mtp_generate_step
from .prompt_lookup import PromptLookupDecoder, prompt_lookup_generate_step
from .suffix_decoding import DraftStats, SuffixDecodingDrafter
from .vlm_mtp import VLMMTPDrafter, load_vlm_mtp_drafter, run_vlm_mtp_decode

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
