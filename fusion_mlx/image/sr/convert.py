import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

_PREFIXES = ("params_ema.", "params.")


def _strip_prefix(key: str) -> str:
    for p in _PREFIXES:
        if key.startswith(p):
            return key[len(p) :]
    return key


def _is_conv_weight(key: str) -> bool:
    return key.endswith(".weight")


def convert_pth_to_safetensors(pth_path: str, sf_path: str) -> int:
    import torch
    from safetensors.numpy import save_file as save_np_safetensors

    sd = torch.load(pth_path, map_location="cpu", weights_only=True)
    # Official RealESRGAN .pth nests the state dict: {"params_ema": OrderedDict(...)}.
    # Other variants use flat keys ("params_ema.conv_first.weight"). Unwrap the
    # nested form; keep the flat form as-is (prefix stripped below).
    if "params_ema" in sd and isinstance(sd["params_ema"], dict):
        sd = sd["params_ema"]
    elif "params" in sd and isinstance(sd["params"], dict):
        sd = sd["params"]
    out = {}
    n = 0
    for k, v in sd.items():
        mk = _strip_prefix(k)
        arr = v.detach().cpu().numpy()
        if _is_conv_weight(mk) and arr.ndim == 4:
            arr = np.transpose(arr, (0, 2, 3, 1))
        arr = np.ascontiguousarray(arr.astype(np.float32))
        out[mk] = arr
        n += 1
    os.makedirs(os.path.dirname(sf_path), exist_ok=True)
    save_np_safetensors(out, sf_path)
    logger.info("sr convert: wrote %d tensors to %s", n, sf_path)
    return n


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 3:
        print("usage: python -m fusion_mlx.image.sr.convert <in.pth> <out.safetensors>")
        sys.exit(1)
    convert_pth_to_safetensors(sys.argv[1], sys.argv[2])
