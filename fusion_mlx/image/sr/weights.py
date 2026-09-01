import logging
import os

from mlx.utils import tree_flatten

from fusion_mlx.image.sr.config import RealESRGANConfig
from fusion_mlx.image.sr.rrdb import RRDBNet

logger = logging.getLogger(__name__)

_PREFIXES = ("params_ema.", "params.")


def _strip_prefix(key: str) -> str:
    for p in _PREFIXES:
        if key.startswith(p):
            return key[len(p):]
    return key


def load_sr_model(model_path: str, config: RealESRGANConfig | None = None) -> RRDBNet:
    # Loads a RealESRGAN safetensors (MLX-NHWC conv layout, keys already
    # transpose-correct from the Task 3 Step 0 converter). Strips
    # params_ema./params. prefix if a raw-converted file is passed.
    import mlx.core as mx

    cfg = config or RealESRGANConfig()
    net = RRDBNet(cfg)
    if not os.path.exists(model_path):
        logger.warning("sr weights: %s missing, returning random-init net", model_path)
        return net

    raw = mx.load(model_path)
    mapped = {_strip_prefix(k): v for k, v in raw.items()}

    flat = tree_flatten(net.parameters())
    loaded = []
    matched = 0
    for k, v in flat:
        if k in mapped:
            mv = mapped[k]
            if mv.dtype != mx.float32:
                mv = mv.astype(mx.float32)
            if list(mv.shape) != list(v.shape):
                logger.warning(
                    "sr weights: shape mismatch %s %s vs %s",
                    k, list(mv.shape), list(v.shape),
                )
                continue
            loaded.append((k, mv))
            matched += 1

    net.load_weights(loaded, strict=False)

    missing = len(flat) - matched
    unmatched = len(mapped) - matched
    if missing or unmatched:
        logger.warning(
            "sr weights: matched=%d missing=%d unmatched=%d (%s)",
            matched, missing, unmatched, model_path,
        )
    else:
        logger.info(
            "sr weights: loaded %d/%d keys (strict) from %s",
            matched, len(flat), model_path,
        )
    return net
