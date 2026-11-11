# M5 Max experiment (#1204, Gate M) — inject native MTP weights into the
# mlx-community Qwen3.8-27B-4bit checkpoint.
#
# The bf16 Qwen/Qwen3.8-27B repo ships 15 mtp.* tensors (shard 18), but the
# community 4bit conversion dropped them, so fusion-mlx's MTP attach gate
# (_checkpoint_has_mtp_weights) disables the head. This script:
#   1. loads bf16 mtp.* tensors from the downloaded shard,
#   2. quantizes every 2D weight with the checkpoint's scheme
#      (affine, group_size=64, bits=4),
#   3. remaps mtp.* -> language_model.mtp.* and shifts raw-HF RMSNorm
#      weights (mean < 0.5) by +1.0 — the mlx_vlm sanitize pass that would
#      normally do this is SKIPPED for already-mlx-format checkpoints
#      (is_mlx_format short-circuits sanitize_weights), so the overlay
#      must arrive in final form,
#   4. writes an overlay model dir: symlinked shards + tokenizer/config,
#      a new mtp.safetensors, and an index whose weight_map includes the
#      quantized mtp keys.
#
# Usage: python inject_mtp_qwen38_1108.py <shard18.safetensors> <overlay_dir>

import json
import sys
from pathlib import Path

BASE_SNAPSHOT = Path(
    "/Users/dahai/.fusion-mlx/models/"
    "models--mlx-community--Qwen3.8-27B-4bit/"
    "snapshots/10c35caafbb80f7dc6a7a432cdd11af10a6d4818"
)
GROUP_SIZE = 64
BITS = 4
NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
    ".pre_fc_norm_hidden.weight",
    ".pre_fc_norm_embedding.weight",
    "mtp.norm.weight",
)


def main():
    shard = Path(sys.argv[1])
    overlay = Path(sys.argv[2])
    overlay.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    tensors = mx.load(str(shard))
    mtp_keys = sorted(k for k in tensors if k.startswith("mtp."))
    print(f"bf16 mtp tensors found: {len(mtp_keys)}")
    if not mtp_keys:
        sys.exit("no mtp.* tensors in shard — wrong shard?")

    out = {}
    n_q = 0
    n_shift = 0
    for key in mtp_keys:
        w = tensors[key]
        if w.ndim == 2 and w.shape[-1] % GROUP_SIZE == 0:
            wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)
            # mlx sibling naming: scales/biases replace the .weight suffix
            # (e.g. gate_proj.weight -> gate_proj.scales), matching the
            # base checkpoint's quantized key layout.
            stem = f"language_model.{key[: -len('.weight')]}"
            out[f"language_model.{key}"] = wq
            out[f"{stem}.scales"] = scales
            out[f"{stem}.biases"] = biases
            n_q += 1
        elif w.ndim == 1 and key.endswith(NORM_SUFFIXES):
            mean = float(w.astype(mx.float32).mean().item())
            if mean < 0.5:
                out[f"language_model.{key}"] = w + 1.0
                n_shift += 1
                print(f"norm shifted +1.0 (raw-HF, mean={mean:.4f}): {key}")
            else:
                out[f"language_model.{key}"] = w
                print(f"norm kept (mean={mean:.4f}): {key}")
        else:
            out[f"language_model.{key}"] = w
    print(
        f"quantized 2D tensors: {n_q}; norms shifted: {n_shift}; "
        f"total output tensors: {len(out)}"
    )

    mtp_path = overlay / "mtp.safetensors"
    mx.save_safetensors(str(mtp_path), {k: v for k, v in out.items()})

    # Symlink every other checkpoint file (config, tokenizer, shards).
    weight_map_override = {}
    for f in BASE_SNAPSHOT.iterdir():
        if f.name == "model.safetensors.index.json" or f.name == mtp_path.name:
            continue
        link = overlay / f.name
        if not link.exists():
            link.symlink_to(f)

    index = json.loads((BASE_SNAPSHOT / "model.safetensors.index.json").read_text())
    for key in sorted(out.keys()):
        weight_map_override[key] = mtp_path.name
    index["weight_map"].update(weight_map_override)
    (overlay / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    print(f"overlay ready: {overlay}")
    print(f"index entries added for {len(weight_map_override)} mtp tensors")


if __name__ == "__main__":
    main()
