# SPDX-License-Identifier: Apache-2.0
"""HF model config analyzer — fetches config.json and detects architecture.

Callers: fusion_mlx.admin.migrate_route
API: analyze_model(hf_id, mirror) -> ModelAnalysis
Schema: ModelAnalysis dataclass
User instruction: "做一个端到端的功能，做模型迁移和量化的功能"
"""

import json
import logging
from dataclasses import dataclass, field
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .architectures import ArchTemplate, match_template

logger = logging.getLogger(__name__)


@dataclass
class ModelAnalysis:
    hf_id: str
    model_type: str = ""
    architectures: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    template: ArchTemplate | None = None
    diff: list[str] = field(default_factory=list)
    estimated_size_gb: float = 0.0
    num_params_b: float = 0.0
    compatible: bool = False
    warnings: list[str] = field(default_factory=list)
    error: str = ""


def _hf_api_url(hf_id: str, filename: str, mirror: bool = False) -> str:
    base = "https://hf-mirror.com" if mirror else "https://huggingface.co"
    return f"{base}/api/models/{hf_id}"


def _hf_file_url(hf_id: str, filename: str, mirror: bool = False) -> str:
    base = "https://hf-mirror.com" if mirror else "https://huggingface.co"
    return f"{base}/{hf_id}/resolve/main/{filename}"


def _fetch_json(url: str, timeout: int = 30) -> dict | None:
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        logger.error("HTTP %d fetching %s: %s", e.code, url, e.reason)
        return None
    except Exception as e:
        logger.error("Error fetching %s: %s", url, e)
        return None


def analyze_model(hf_id: str, mirror: bool = False) -> ModelAnalysis:
    result = ModelAnalysis(hf_id=hf_id)

    api_url = _hf_api_url(hf_id, "config.json", mirror)
    api_data = _fetch_json(api_url)
    if not api_data:
        result.error = f"Failed to fetch model info for {hf_id}"
        return result

    siblings = api_data.get("siblings", [])
    safetensors_files = [
        s.get("rfilename", "")
        for s in siblings
        if s.get("rfilename", "").endswith(".safetensors")
    ]
    pytorch_files = [
        s.get("rfilename", "")
        for s in siblings
        if s.get("rfilename", "").endswith(".bin")
    ]
    has_safetensors = len(safetensors_files) > 0
    has_pytorch = len(pytorch_files) > 0

    if not has_safetensors and not has_pytorch:
        result.error = f"No model weight files found for {hf_id}"
        return result

    if not has_safetensors:
        result.warnings.append(
            "No safetensors files found — will need PyTorch->safetensors pre-conversion"
        )

    config_url = _hf_file_url(hf_id, "config.json", mirror)
    config = _fetch_json(config_url)
    if not config:
        result.error = f"Failed to fetch config.json for {hf_id}"
        return result

    result.config = config
    result.model_type = config.get("model_type", "unknown")
    result.architectures = config.get("architectures", [])

    model_size = api_data.get("size_on_disk", api_data.get("usedStorage", 0))
    if model_size > 0:
        result.estimated_size_gb = model_size / (1024**3)

    if not result.estimated_size_gb:
        n_layers = config.get("num_hidden_layers", config.get("n_layer", 0))
        hidden = config.get("hidden_size", config.get("n_embd", 0))
        intermediate = config.get(
            "intermediate_size", config.get("n_inner", 4 * hidden)
        )
        vocab = config.get("vocab_size", 32000)
        n_heads = config.get("num_attention_heads", config.get("n_head", 0))
        if n_layers and hidden:
            embed_params = vocab * hidden
            attn_params = n_layers * (
                (hidden * hidden + (hidden * hidden // n_heads) * 2) + hidden * hidden
            )
            mlp_params = n_layers * (hidden * intermediate * 3)
            norm_params = n_layers * hidden * 2 + hidden
            total = embed_params + attn_params + mlp_params + norm_params
            result.num_params_b = total / 1e9
            result.estimated_size_gb = total * 2 / (1024**3)

    arch_name = result.architectures[0] if result.architectures else result.model_type
    template, diff = match_template(arch_name, config)
    result.template = template
    result.diff = diff

    if template:
        result.compatible = True
        if template.attention_type == "mla":
            result.warnings.append(
                "MLA attention requires custom implementation — advanced migration"
            )
        if template.codegen_hints.get("moe"):
            result.warnings.append(
                "MoE architecture requires expert routing — advanced migration"
            )
    else:
        result.compatible = False
        result.warnings.append(
            f"Unknown architecture '{arch_name}' — manual migration required"
        )

    logger.info(
        "Analyzed %s: type=%s arch=%s compatible=%s size=%.1fGB",
        hf_id,
        result.model_type,
        arch_name,
        result.compatible,
        result.estimated_size_gb,
    )
    return result
