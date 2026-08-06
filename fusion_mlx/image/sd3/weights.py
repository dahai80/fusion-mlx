import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def remap_transformer_weights(raw: dict, num_layers: int) -> list:
    P = "model.diffusion_model."
    pairs = []
    for k, v in raw.items():
        if not k.startswith(P):
            continue
        rest = k[len(P):]
        new = _map_one(rest, num_layers)
        if new is None:
            logger.debug("SD3 weights skip unmapped key: %s", k)
            continue
        if new == "x_embedder_proj.weight":
            v = mx.transpose(v, (0, 2, 3, 1))
        pairs.append((new, v))
    logger.info("SD3 transformer remapped %d / %d keys", len(pairs), len(raw))
    return pairs


def _map_one(rest: str, num_layers: int) -> str | None:
    if rest == "pos_embed":
        return "pos_embed"
    if rest.startswith("x_embedder.proj."):
        return "x_embedder_proj." + rest.split(".")[-1]
    if rest.startswith("t_embedder.mlp."):
        idx = rest.split(".")[2]
        attr = rest.split(".")[-1]
        return f"t_embedder.mlp.layers.{idx}.{attr}"
    if rest.startswith("y_embedder.mlp."):
        idx = rest.split(".")[2]
        attr = rest.split(".")[-1]
        return f"y_embedder.mlp.layers.{idx}.{attr}"
    if rest.startswith("context_embedder."):
        return "context_embedder." + rest.split(".")[-1]
    if rest.startswith("final_layer."):
        if rest.startswith("final_layer.adaLN_modulation.1."):
            return "final_layer.mod.linear." + rest.split(".")[-1]
        if rest.startswith("final_layer.linear."):
            return "final_layer.linear." + rest.split(".")[-1]
        return None
    if rest.startswith("joint_blocks."):
        parts = rest.split(".")
        idx = int(parts[1])
        block = f"joint_blocks.{idx}"
        sub = parts[2]
        if sub == "x_block":
            return _map_block_side(block, "x", parts[3:])
        if sub == "context_block":
            return _map_block_side(block, "context", parts[3:])
        return None
    return None


def _map_block_side(block, side, parts):
    if parts[0] == "adaLN_modulation" and parts[1] == "1":
        return f"{block}.{side}_mod.linear.{parts[-1]}"
    if parts[0] == "attn":
        if parts[1] == "qkv":
            return f"{block}.attn.{side}_qkv.{parts[-1]}"
        if parts[1] == "proj":
            return f"{block}.attn.{side}_proj.{parts[-1]}"
        return None
    if parts[0] == "mlp":
        if parts[1] == "fc1":
            return f"{block}.{side}_mlp.fc1.{parts[-1]}"
        if parts[1] == "fc2":
            return f"{block}.{side}_mlp.fc2.{parts[-1]}"
        return None
    return None


def load_transformer(model, raw: dict, num_layers: int) -> None:
    pairs = remap_transformer_weights(raw, num_layers)
    model.load_weights(pairs, strict=False)
    loaded = set(p for p, _ in pairs)
    missing = _find_missing(model, loaded)
    if missing:
        logger.warning(
            "SD3 transformer missing %d params (first 10): %s",
            len(missing), missing[:10],
        )


def _find_missing(model, loaded: set) -> list:
    flat = []
    _flatten(model.parameters(), "", flat)
    return [p for p in flat if p not in loaded]


def _flatten(tree, prefix, out):
    if isinstance(tree, dict):
        for k, v in tree.items():
            _flatten(v, f"{prefix}.{k}" if prefix else k, out)
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            _flatten(v, f"{prefix}.{i}", out)
    elif isinstance(tree, mx.array):
        out.append(prefix)
