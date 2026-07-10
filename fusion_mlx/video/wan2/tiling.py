from collections.abc import Callable

import mlx.core as mx

from ..ltx2.video_vae.tiling import (
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
    map_spatial_slice,
    map_temporal_slice,
    split_in_spatial,
    split_in_temporal,
)

__all__ = [
    "SpatialTilingConfig",
    "TemporalTilingConfig",
    "TilingConfig",
    "decode_with_tiling",
    "map_spatial_slice",
    "map_temporal_slice",
    "split_in_spatial",
    "split_in_temporal",
]


def decode_with_tiling(
    decoder_fn,
    latents: mx.array,
    tiling_config: TilingConfig,
    spatial_scale: int = 32,
    temporal_scale: int = 8,
    causal: bool = False,
    causal_temporal: bool = True,
    timestep: mx.array | None = None,
    chunked_conv: bool = False,
    on_frames_ready: Callable[[mx.array, int], None] | None = None,
) -> mx.array:
    import gc

    b, c, f_latent, h_latent, w_latent = latents.shape

    # Compute output shape
    out_f = (
        (1 + (f_latent - 1) * temporal_scale)
        if causal_temporal
        else (f_latent * temporal_scale)
    )
    out_h = h_latent * spatial_scale
    out_w = w_latent * spatial_scale

    # Get tile size and overlap in latent space
    if tiling_config.spatial_config is not None:
        s_cfg = tiling_config.spatial_config
        spatial_tile_size = s_cfg.tile_size_in_pixels // spatial_scale
        spatial_overlap = s_cfg.tile_overlap_in_pixels // spatial_scale
    else:
        spatial_tile_size = max(h_latent, w_latent)
        spatial_overlap = 0

    if tiling_config.temporal_config is not None:
        t_cfg = tiling_config.temporal_config
        temporal_tile_size = t_cfg.tile_size_in_frames // temporal_scale
        temporal_overlap = t_cfg.tile_overlap_in_frames // temporal_scale
    else:
        temporal_tile_size = f_latent
        temporal_overlap = 0

    # Compute intervals for each dimension
    if causal_temporal:
        temporal_intervals = split_in_temporal(
            temporal_tile_size, temporal_overlap, f_latent
        )
    else:
        temporal_intervals = split_in_spatial(
            temporal_tile_size, temporal_overlap, f_latent
        )
    height_intervals = split_in_spatial(spatial_tile_size, spatial_overlap, h_latent)
    width_intervals = split_in_spatial(spatial_tile_size, spatial_overlap, w_latent)

    num_t_tiles = len(temporal_intervals.starts)
    num_h_tiles = len(height_intervals.starts)
    num_w_tiles = len(width_intervals.starts)
    total_tiles = num_t_tiles * num_h_tiles * num_w_tiles  # noqa: F841

    # Initialize output and weight accumulator
    # Use float32 for accumulation to avoid precision issues
    output = mx.zeros((b, 3, out_f, out_h, out_w), dtype=mx.float32)
    weights = mx.zeros((b, 1, out_f, out_h, out_w), dtype=mx.float32)
    mx.eval(output, weights)

    tile_idx = 0
    for t_idx in range(num_t_tiles):
        t_start = temporal_intervals.starts[t_idx]
        t_end = temporal_intervals.ends[t_idx]
        t_left = temporal_intervals.left_ramps[t_idx]
        t_right = temporal_intervals.right_ramps[t_idx]

        # Map temporal coordinates
        if causal_temporal:
            out_t_slice, t_mask = map_temporal_slice(
                t_start, t_end, t_left, t_right, temporal_scale
            )
        else:
            out_t_slice, t_mask = map_spatial_slice(
                t_start, t_end, t_left, t_right, temporal_scale
            )

        for h_idx in range(num_h_tiles):
            h_start = height_intervals.starts[h_idx]
            h_end = height_intervals.ends[h_idx]
            h_left = height_intervals.left_ramps[h_idx]
            h_right = height_intervals.right_ramps[h_idx]

            # Map height coordinates
            out_h_slice, h_mask = map_spatial_slice(
                h_start, h_end, h_left, h_right, spatial_scale
            )

            for w_idx in range(num_w_tiles):
                w_start = width_intervals.starts[w_idx]
                w_end = width_intervals.ends[w_idx]
                w_left = width_intervals.left_ramps[w_idx]
                w_right = width_intervals.right_ramps[w_idx]

                # Map width coordinates
                out_w_slice, w_mask = map_spatial_slice(
                    w_start, w_end, w_left, w_right, spatial_scale
                )

                # Extract tile latents (small slice)
                tile_latents = latents[
                    :, :, t_start:t_end, h_start:h_end, w_start:w_end
                ]

                # Decode tile
                tile_output = decoder_fn(
                    tile_latents,
                    causal=causal,
                    timestep=timestep,
                    debug=False,
                    chunked_conv=chunked_conv,
                )
                mx.eval(tile_output)

                # Clear tile_latents reference
                del tile_latents

                # Get actual decoded dimensions
                _, _, decoded_t, decoded_h, decoded_w = tile_output.shape
                expected_t = out_t_slice.stop - out_t_slice.start
                expected_h = out_h_slice.stop - out_h_slice.start
                expected_w = out_w_slice.stop - out_w_slice.start

                # Handle potential size mismatches (use minimum)
                actual_t = min(decoded_t, expected_t)
                actual_h = min(decoded_h, expected_h)
                actual_w = min(decoded_w, expected_w)

                # Build blend mask
                t_mask_slice = t_mask[:actual_t] if len(t_mask) > actual_t else t_mask
                h_mask_slice = h_mask[:actual_h] if len(h_mask) > actual_h else h_mask
                w_mask_slice = w_mask[:actual_w] if len(w_mask) > actual_w else w_mask

                blend_mask = (
                    t_mask_slice.reshape(1, 1, -1, 1, 1)
                    * h_mask_slice.reshape(1, 1, 1, -1, 1)
                    * w_mask_slice.reshape(1, 1, 1, 1, -1)
                )

                # Slice tile output to match
                tile_output_slice = tile_output[
                    :, :, :actual_t, :actual_h, :actual_w
                ].astype(mx.float32)

                # Clear full tile_output
                del tile_output

                # Compute output coordinates
                t_out_start = out_t_slice.start
                t_out_end = t_out_start + actual_t
                h_out_start = out_h_slice.start
                h_out_end = h_out_start + actual_h
                w_out_start = out_w_slice.start
                w_out_end = w_out_start + actual_w

                # Weighted accumulation
                weighted_tile = tile_output_slice * blend_mask

                # Update output using slice assignment
                output[
                    :,
                    :,
                    t_out_start:t_out_end,
                    h_out_start:h_out_end,
                    w_out_start:w_out_end,
                ] = (
                    output[
                        :,
                        :,
                        t_out_start:t_out_end,
                        h_out_start:h_out_end,
                        w_out_start:w_out_end,
                    ]
                    + weighted_tile
                )
                weights[
                    :,
                    :,
                    t_out_start:t_out_end,
                    h_out_start:h_out_end,
                    w_out_start:w_out_end,
                ] = (
                    weights[
                        :,
                        :,
                        t_out_start:t_out_end,
                        h_out_start:h_out_end,
                        w_out_start:w_out_end,
                    ]
                    + blend_mask
                )

                # Force evaluation to free memory
                mx.eval(output, weights)

                # Clean up tile-specific arrays
                del tile_output_slice, weighted_tile, blend_mask
                del t_mask_slice, h_mask_slice, w_mask_slice

                tile_idx += 1

                # Periodic garbage collection and cache clearing
                if tile_idx % 4 == 0:
                    gc.collect()
                    try:
                        mx.clear_cache()
                    except Exception:
                        pass  # May not be available on all platforms

        # After completing all spatial tiles for this temporal tile,
        # check if any frames are now finalized (no future tiles will contribute)
        if on_frames_ready is not None and num_t_tiles > 1:
            # Determine the finalized frame boundary
            # Frames before the start of the next tile's output region are finalized
            if t_idx < num_t_tiles - 1:
                # Next tile starts at temporal_intervals.starts[t_idx + 1]
                next_tile_start_latent = temporal_intervals.starts[t_idx + 1]
                # Map to output frame index (first frame of next tile's contribution)
                if next_tile_start_latent == 0:
                    next_tile_start_out = 0
                elif causal_temporal:
                    next_tile_start_out = (
                        1 + (next_tile_start_latent - 1) * temporal_scale
                    )
                else:
                    next_tile_start_out = next_tile_start_latent * temporal_scale

                # We need to track how many frames we've already emitted
                if not hasattr(decode_with_tiling, "_emitted_frames"):
                    decode_with_tiling._emitted_frames = 0
                emitted = decode_with_tiling._emitted_frames

                if next_tile_start_out > emitted:
                    # Normalize and emit frames [emitted, next_tile_start_out)
                    finalized_weights = weights[:, :, emitted:next_tile_start_out, :, :]
                    finalized_weights = mx.maximum(finalized_weights, 1e-8)
                    finalized_output = (
                        output[:, :, emitted:next_tile_start_out, :, :]
                        / finalized_weights
                    )
                    finalized_output = finalized_output.astype(latents.dtype)
                    mx.eval(finalized_output)

                    on_frames_ready(finalized_output, emitted)
                    decode_with_tiling._emitted_frames = next_tile_start_out

                    del finalized_output, finalized_weights
                    gc.collect()

    # Normalize by weights
    weights = mx.maximum(weights, 1e-8)
    output = output / weights
    mx.eval(output)

    # Emit remaining frames if callback provided
    if on_frames_ready is not None:
        emitted = getattr(decode_with_tiling, "_emitted_frames", 0)
        if emitted < out_f:
            remaining_output = output[:, :, emitted:, :, :].astype(latents.dtype)
            mx.eval(remaining_output)
            on_frames_ready(remaining_output, emitted)
            del remaining_output

    # Reset emitted frames counter for next call
    if hasattr(decode_with_tiling, "_emitted_frames"):
        del decode_with_tiling._emitted_frames

    # Clean up weights
    del weights
    gc.collect()

    # Convert back to original dtype if needed
    return output.astype(latents.dtype)
