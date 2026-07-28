# SPDX-License-Identifier: Apache-2.0
# UniWorld-V1 feature merge utilities: _insert_img_to_vlm, find_true_blocks.
# Pure-MLX reimplementation of the PyTorch masked_scatter_ + find_true_blocks
# logic from modeling_univa_qwen2p5vl.py. For batch_size=1 inference we
# simplify with Python loops + mx.concatenate instead of masked_scatter_.

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def find_true_blocks(mask: mx.array) -> list[list[tuple[int, int]]]:
    """Find contiguous True blocks in a 1D boolean mask per batch element.

    Args:
        mask: (B, L) boolean array
    Returns:
        List of lists of (start, end) tuples (end exclusive) per batch element
    """
    if mask.ndim == 1:
        mask = mask.reshape(1, -1)
    results = []
    for b in range(mask.shape[0]):
        row = mask[b]
        blocks = []
        in_block = False
        start = 0
        for i in range(row.shape[0]):
            val = bool(row[i])
            if val and not in_block:
                start = i
                in_block = True
            elif not val and in_block:
                blocks.append((start, i))
                in_block = False
        if in_block:
            blocks.append((start, row.shape[0]))
        results.append(blocks)
    return results


def find_all_token_positions(input_ids: mx.array, token_id: int) -> list[list[int]]:
    """Find all positions of a specific token_id per batch element.

    Args:
        input_ids: (B, L) integer array
        token_id: token id to find
    Returns:
        List of lists of positions per batch element
    """
    if input_ids.ndim == 1:
        input_ids = input_ids.reshape(1, -1)
    results = []
    for b in range(input_ids.shape[0]):
        row = input_ids[b]
        positions = []
        for i in range(row.shape[0]):
            if int(row[i]) == token_id:
                positions.append(i)
        results.append(positions)
    return results


def insert_img_to_vlm(
    vlm_hidden_states: mx.array,
    siglip_hidden_states: mx.array,
    input_ids: mx.array,
    image_end_token_id: int = 151646,
) -> mx.array:
    """Merge SigLIP features into VLM hidden states at image_end_token positions.

    Replaces PyTorch masked_scatter_ with explicit Python loop for batch_size=1.
    For each image_end_token position, the corresponding SigLIP features are
    inserted into the VLM hidden states.

    Args:
        vlm_hidden_states: (1, L, D_vlm) VLM output hidden states
        siglip_hidden_states: (1, N_siglip, D_siglip) SigLIP encoded features
        input_ids: (1, L) input token ids
        image_end_token_id: token id marking image boundaries
    Returns:
        (1, L, D_vlm) hidden states with SigLIP features merged
    """
    if vlm_hidden_states.shape[0] != 1:
        logger.warning(
            "insert_img_to_vlm: batch_size=%d, only batch_size=1 tested",
            vlm_hidden_states.shape[0],
        )

    vlm_d = vlm_hidden_states.shape[-1]
    siglip_d = siglip_hidden_states.shape[-1]

    end_positions = find_all_token_positions(input_ids, image_end_token_id)
    if not end_positions or not end_positions[0]:
        logger.debug("No image_end_token found, returning vlm_hidden_states unchanged")
        return vlm_hidden_states

    num_images = len(end_positions[0])
    total_siglip_tokens = siglip_hidden_states.shape[1]
    tokens_per_image = total_siglip_tokens // num_images

    if tokens_per_image == 0:
        logger.warning(
            "Not enough SigLIP tokens (%d) for %d images",
            total_siglip_tokens,
            num_images,
        )
        return vlm_hidden_states

    if vlm_d == siglip_d:
        siglip_projected = siglip_hidden_states
    else:
        siglip_projected = _project_dim(siglip_hidden_states, vlm_d)

    result = vlm_hidden_states
    for img_idx, end_pos in enumerate(end_positions[0]):
        start_offset = img_idx * tokens_per_image
        end_offset = start_offset + tokens_per_image
        siglip_chunk = siglip_projected[0, start_offset:end_offset, :]
        chunk_len = min(siglip_chunk.shape[0], end_pos)
        result = _replace_range(
            result, 0, end_pos - chunk_len, end_pos, siglip_chunk[:chunk_len]
        )

    return result


def apply_shortcut_blend(
    vlm_hidden_states: mx.array,
    shortcut_embeds: mx.array,
    image_mask: mx.array,
    scale: float = 0.5,
) -> mx.array:
    """Blend shortcut embeddings with VLM hidden states at image token positions.

    At image token positions: output = scale * shortcut + (1-scale) * vlm_output

    Args:
        vlm_hidden_states: (B, L, D) VLM output
        shortcut_embeds: (B, L, D) shortcut (original image) embeddings
        image_mask: (B, L) boolean mask for image token positions
        scale: blending factor (0=only VLM, 1=only shortcut)
    Returns:
        (B, L, D) blended hidden states
    """
    if shortcut_embeds is None or scale == 0.0:
        return vlm_hidden_states

    if image_mask.ndim == 2:
        mask = image_mask[:, :, None]
    else:
        mask = image_mask

    blended = scale * shortcut_embeds + (1.0 - scale) * vlm_hidden_states
    result = mx.where(mask, blended, vlm_hidden_states)
    return result


def apply_residual_image_factor(
    vlm_hidden_states: mx.array,
    original_image_embeds: mx.array,
    image_mask: mx.array,
    factor: float = 0.3,
) -> mx.array:
    """Blend original image embeddings back into VLM outputs.

    output = vlm_output + factor * original_image_embeds at image positions.

    Args:
        vlm_hidden_states: (B, L, D) VLM output
        original_image_embeds: (B, L, D) original image embeddings
        image_mask: (B, L) boolean mask for image token positions
        factor: residual factor
    Returns:
        (B, L, D) hidden states with residual blend
    """
    if original_image_embeds is None or factor == 0.0:
        return vlm_hidden_states

    if image_mask.ndim == 2:
        mask = image_mask[:, :, None]
    else:
        mask = image_mask

    residual = vlm_hidden_states + factor * original_image_embeds
    result = mx.where(mask, residual, vlm_hidden_states)
    return result


def _project_dim(x: mx.array, target_dim: int) -> mx.array:
    if x.shape[-1] == target_dim:
        return x
    import mlx.nn as nn

    proj = nn.Linear(x.shape[-1], target_dim)
    mx.eval(proj.parameters())
    return proj(x)


def _replace_range(
    arr: mx.array, batch: int, start: int, end: int, replacement: mx.array
) -> mx.array:
    parts = []
    if start > 0:
        parts.append(arr[batch, :start, :])
    if replacement.shape[0] > 0:
        parts.append(replacement)
    if end < arr.shape[1]:
        parts.append(arr[batch, end:, :])
    new_row = mx.concatenate(parts, axis=0)
    return mx.concatenate(
        [arr[:batch], new_row[None, :, :], arr[batch + 1 :]],
        axis=0,
    )
