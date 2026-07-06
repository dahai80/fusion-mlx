# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os

logger = logging.getLogger(__name__)

_SAMPLING_KEYS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presencePenalty",
    "frequencyPenalty",
)


def load_generation_config_sampling(model_path: str | None) -> dict[str, float | int]:
    if not model_path:
        return {}
    config_path = _resolve_config_path(model_path)
    if config_path is None:
        return {}
    try:
        with open(config_path) as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("generation_config: skip %s (read/parse failed: %s)", config_path, exc)
        return {}
    if not isinstance(raw, dict):
        logger.debug("generation_config: skip %s (not a JSON object)", config_path)
        return {}
    out: dict[str, float | int] = {}
    for key in _SAMPLING_KEYS:
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        if value != value or value in (float("inf"), float("-inf")):
            continue
        if key == "top_k":
            if isinstance(value, float) and not value.is_integer():
                continue
            out[key] = int(value)
            continue
        out[key] = value
    if out:
        logger.info("generation_config: loaded sampling defaults from %s: %s", config_path, out)
    return out


def load_generation_config_eos_ids(model_path: str | None) -> tuple[int, ...]:
    if not model_path:
        return ()
    config_path = _resolve_config_path(model_path)
    if config_path is None:
        return ()
    try:
        with open(config_path) as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("generation_config: eos read failed for %s: %s", config_path, exc)
        return ()
    if not isinstance(raw, dict):
        return ()
    value = raw.get("eos_token_id")
    if isinstance(value, bool):
        return ()
    if isinstance(value, int):
        items: list = [value]
    elif isinstance(value, list):
        items = value
    else:
        return ()
    out: list[int] = []
    for item in items:
        if isinstance(item, bool):
            continue
        if not isinstance(item, int):
            continue
        out.append(item)
    if out:
        logger.info("generation_config: loaded extra EOS token ids from %s: %s", config_path, out)
    return tuple(out)


def _resolve_config_path(model_path: str) -> str | None:
    if os.path.isdir(model_path):
        candidate = os.path.join(model_path, "generation_config.json")
        return candidate if os.path.isfile(candidate) else None
    if "/" in model_path and ":" not in model_path:
        hub = os.environ.get("HF_HUB_CACHE") or os.path.expanduser(
            "~/.cache/huggingface/hub"
        )
        cache_root = os.path.join(hub, "models--" + model_path.replace("/", "--"))
        ref_path = os.path.join(cache_root, "refs", "main")
        if os.path.isfile(ref_path):
            try:
                with open(ref_path) as fh:
                    sha = fh.read().strip()
            except OSError:
                sha = ""
            if sha:
                candidate = os.path.join(
                    cache_root, "snapshots", sha, "generation_config.json"
                )
                if os.path.isfile(candidate):
                    return candidate
        repo_dir = os.path.join(cache_root, "snapshots")
        if os.path.isdir(repo_dir):
            try:
                snapshots = sorted(os.listdir(repo_dir))
            except OSError:
                return None
            for snap in snapshots:
                candidate = os.path.join(repo_dir, snap, "generation_config.json")
                if os.path.isfile(candidate):
                    return candidate
    return None
