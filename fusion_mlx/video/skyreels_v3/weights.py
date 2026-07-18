# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 权重加载器.

基于底座 fusion_mlx.video.wan2.utils 蓝本,
为 SkyReels-V3 三大主干提供真实权重加载 (非 stub).

加载链路:
  1. DiT 主干: mx.load(safetensors) → model.load_weights(strict=False) → mx.eval
  2. VAE: mx.load → SkyReelsVAE.load_weights (复用底座 WanVAE)
  3. UMT5 文本编码器: mx.load → UMT5Encoder.load_weights
  4. CLIP (A2V): mlx_clip 或 transformers CPU 兜底

权重路径约定 (HuggingFace cache):
  ~/.cache/huggingface/hub/<model_id--snapshots/<sha>/
  或自定义 --model-path 指向已转换的 MLX safetensors 目录
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 模型路径解析
# ---------------------------------------------------------------------------
def resolve_model_path(model_path: str | Path | None, model_key: str) -> Path:
    """解析模型权重路径.

    优先级:
      1. 显式传入的 model_path (已是 MLX safetensors 目录)
      2. HuggingFace cache: ~/.cache/huggingface/hub/<repo_id--snapshots/<sha>/
      3. 兜底: 返回 model_path 原值 (调用方负责报错)

    Args:
        model_path: 显式路径 (None 则自动解析)
        model_key: e.g. "skyreels-v3-r2v-14b"

    Returns:
        Path 指向包含 *.safetensors 的目录
    """
    if model_path is not None:
        p = Path(model_path).expanduser()
        if p.exists() and (p.is_dir() and any(p.glob("*.safetensors")) or p.is_file()):
            return p
        # 尝试展开为 HF cache 路径
        if not p.is_absolute():
            p = Path.home() / ".cache" / "huggingface" / "hub" / p
            if p.exists():
                return p
        logger.warning("model_path %s 不含 safetensors, 尝试 HF cache", model_path)

    # 自动解析 HF cache
    from .config import get_branch_config

    branch_cfg = get_branch_config(model_key)
    repo_id = branch_cfg.hf_model_id  # e.g. "Skywork/SkyReels-V3-R2V-14B"
    repo_dir = repo_id.replace("/", "--")
    cache_base = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_dir}"

    if cache_base.exists():
        # 寻找 snapshots/<sha>/
        snapshots = cache_base / "snapshots"
        if snapshots.exists():
            sha_dirs = [d for d in snapshots.iterdir() if d.is_dir()]
            if sha_dirs:
                return sha_dirs[0]

    # 兜底: 返回原值, 调用方报错
    return Path(model_path) if model_path else cache_base


# ---------------------------------------------------------------------------
# DiT 主干权重加载
# ---------------------------------------------------------------------------
def load_dit_weights(
    model: nn.Module,
    weights_dir: Path,
    *,
    quantization: dict | None = None,
    strict: bool = False,
) -> nn.Module:
    """加载 DiT 主干权重到 MLX 模型.

    Args:
        model: SkyReelsR2VDiT / SkyReelsV2VDiT / SkyReelsA2VDiT 实例
        weights_dir: 包含 *.safetensors 的目录
        quantization: 可选量化配置 {group_size, bits}
        strict: 是否严格匹配权重名 (默认 False, 容忍新增/缺失参数)

    Returns:
        加载权重后的 model (in-place)
    """
    # 可选量化 (在加载权重前应用, 后续权重会按量化格式加载)
    if quantization:
        try:
            nn.quantize(
                model,
                group_size=quantization.get("group_size", 64),
                bits=quantization.get("bits", 4),
            )
            logger.info(
                "DiT 量化: bits=%d group=%d",
                quantization.get("bits", 4),
                quantization.get("group_size", 64),
            )
        except Exception as exc:
            logger.warning("DiT 量化失败, 跳过: %s", exc)

    # 加载权重
    weights = _load_safetensors_dir(weights_dir)
    if not weights:
        raise FileNotFoundError(
            f"未找到 safetensors 权重文件于 {weights_dir}"
        )

    # 权重名对齐 (convert_skyreels_v3.py 已处理转置, 这里直接加载)
    # A2V 独有: audio_cross_attn.kv_linear.weight 真实布局 (768,10240)=(in,out),
    # mlx.nn.Linear 期望 (out,in)=(10240,768), 需转置后加载
    for k in list(weights.keys()):
        if k.endswith("audio_cross_attn.kv_linear.weight"):
            w = weights[k]
            if w.shape == (768, 10240):  # (in, out) 反布局
                weights[k] = w.T.astype(w.dtype)  # → (10240, 768)=(out, in)
                logger.info("audio_cross_attn.kv_linear.weight 转置: %s→%s",
                            w.shape, weights[k].shape)
    try:
        model.load_weights(list(weights.items()), strict=strict)
        mx.eval(model.parameters())
        logger.info(
            "DiT 权重加载成功: %d tensors from %s",
            len(weights), weights_dir,
        )
    except Exception as exc:
        logger.warning(
            "DiT 权重加载部分失败 (strict=%s): %s",
            strict, exc,
        )
        if strict:
            # 降级为非严格模式重试
            model.load_weights(list(weights.items()), strict=False)
            mx.eval(model.parameters())
            logger.info("DiT 权重加载 (strict=False) 成功")

    return model


# ---------------------------------------------------------------------------
# VAE 权重加载
# ---------------------------------------------------------------------------
def load_vae_weights(
    vae: nn.Module,
    weights_dir: Path,
    *,
    strict: bool = False,
) -> nn.Module:
    """加载 VAE 权重.

    Args:
        vae: SkyReelsVAE 实例
        weights_dir: VAE 权重目录 (含 *.safetensors)
        strict: 是否严格匹配

    Returns:
        加载权重后的 vae (in-place)
    """
    weights = _load_safetensors_dir(weights_dir)
    if not weights:
        logger.warning("VAE 权重目录 %s 为空, 跳过加载", weights_dir)
        return vae

    # VAE 权重上转 float32 保画质 (与底座 wan2/utils.py 一致)
    weights = {k: v.astype(mx.float32) for k, v in weights.items()}

    # wan2/vae.py 底座用普通 list 存子层 (residual/middle/upsamples/head/resample),
    # MLX nn.Module 不收录 list 属性, parameters() 丢子层权重致 load_weights 加载 0.
    # 解法: 先 load_weights 跑可见层, 再手动按 key 路径递归注入 list 子层.
    # 注意: 真实权重 key 是 "decoder.*" (无 vae. 前缀), 需注入到 vae.vae 底座 (WanVAE) 而非包装层
    inject_target = getattr(vae, "vae", None) or vae  # SkyReelsVAE.vae → WanVAE 底座
    try:
        vae.load_weights(list(weights.items()), strict=strict)
        _inject_list_child_weights(inject_target, weights)
        mx.eval(vae.parameters())
        logger.info("VAE 权重加载成功: %d tensors", len(weights))
    except Exception as exc:
        logger.warning("VAE 权重加载失败 (strict=%s): %s", strict, exc)
        if strict:
            vae.load_weights(list(weights.items()), strict=False)
            _inject_list_child_weights(inject_target, weights)
            mx.eval(vae.parameters())

    return vae


def _inject_list_child_weights(module: nn.Module, weights: dict) -> None:
    """手动注入 list 属性子层权重 (绕开 MLX 不收录 list 属性的限制).

    遍历 weights 的 key 路径 (e.g. "decoder.middle.0.residual.0.gamma"),
    若某段对应 list 属性则按索引定位子模块, 逐层递归注入参数.
    同时覆盖 CausalConv3d/RMS_norm 裸 mx.array 属性 (非 _parameters 槽).
    """
    for key, value in weights.items():
        parts = key.split(".")
        cur = module
        try:
            # 逐段定位到目标模块 (处理 list 属性)
            for part in parts[:-1]:  # 最后一段是参数名 (weight/bias/gamma)
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                elif part.isdigit() and isinstance(cur, (list, tuple)):
                    cur = cur[int(part)]
                else:
                    cur = None
                    break
            if cur is None:
                continue
            param_name = parts[-1]
            # 优先setattr 裸 mx.array 属性 (CausalConv3d.weight/bias, RMS_norm.gamma)
            if hasattr(cur, param_name):
                val = value
                # Conv2d 权重布局对齐: 真实 (out,in,kh,kw) → MLX nn.Conv2d 期望 (out,kh,kw,in)
                # 触发条件: cur 是 nn.Conv2d + weight 4D + 真实布局第2维是 in (非 kh/kw)
                # 判据: 真实权重 shape[1] (in_c) == cur 期望权重的末维 (in_c), 且 shape[1] > 1
                if param_name == "weight" and hasattr(cur, "weight"):
                    import mlx.nn as _nn
                    if isinstance(cur, _nn.Conv2d) and value.ndim == 4:
                        out_c, in_c, kh, kw = value.shape
                        cur_w = getattr(cur, "weight")
                        # 真实布局 (out,in,kh,kw) vs MLX 期望 (out,kh,kw,in):
                        # 简判: 真实 shape[1] (in_c) > 1 且 in_c 既不等于 kh 也不等于 kw
                        # (即第2维确是 in_c 而非 kh/kw, 需转置到末维)
                        if in_c > 1 and in_c != kh and in_c != kw:
                            val = value.transpose(0, 2, 3, 1)  # (out,in,kh,kw) → (out,kh,kw,in)
                setattr(cur, param_name, val)
            elif hasattr(cur, "_parameters") and param_name in getattr(cur, "_parameters", {}):
                cur._parameters[param_name] = value
        except (IndexError, AttributeError):
            continue


# ---------------------------------------------------------------------------
# UMT5 文本编码器权重加载
# ---------------------------------------------------------------------------
def load_text_encoder_weights(
    encoder: nn.Module,
    weights_dir: Path,
    *,
    strict: bool = True,
) -> nn.Module:
    """加载 UMT5 文本编码器权重.

    Args:
        encoder: UMT5Encoder 实例
        weights_dir: UMT5 权重目录
        strict: 是否严格匹配 (UMT5 通常 strict=True)

    Returns:
        加载权重后的 encoder (in-place)
    """
    weights = _load_safetensors_dir(weights_dir)
    if not weights:
        logger.warning("UMT5 权重目录 %s 为空, 跳过加载", weights_dir)
        return encoder

    # T5 保 float32 (与底座 wan2/utils.py load_t5_encoder 一致), 但改 bf16 常驻 Metal 降显存
    # (V2V/A2V 加载 T5 11GB float32 致 Metal 峰值翻倍 + GPU Command Buffer Timeout)
    # 转 bf16 常驻: 11GB float32 → 5.5GB bf16, 释放 5.5GB Metal 给 DiT temporal 分支用
    weights = {k: v.astype(mx.bfloat16) for k, v in weights.items()}

    # AtomCode 专题优化: 映射 blocks.{N}.* → block_{N}.* (T5Encoder 用命名属性替 list)
    # MLX nn.Module 不收录普通 list 属性, T5Encoder.block_0/block_1/... 才入 _children
    # 同时 gate.0.* → gate_0.* (T5DenseGatedActDense 用 gate_0 命名属性替 ModuleList)
    remapped = {}
    for k, v in weights.items():
        nk = k
        if k.startswith("blocks."):
            parts = k.split(".", 2)  # ["blocks", "{N}", "{sub}"]
            n = parts[1]
            sub = parts[2] if len(parts) > 2 else ""
            nk = f"block_{n}.{sub}" if sub else f"block_{n}"
        # gate.0.* → gate_0.* (MLX nn 无 ModuleList, 用命名属性)
        nk = nk.replace(".gate.0.", ".gate_0.")
        remapped[nk] = v
    weights = remapped

    try:
        encoder.load_weights(list(weights.items()), strict=strict)
        # 分批 eval 避一次性提交 11GB 致 GPU Timeout (>10s 阜默)
        _eval_params_batched(encoder, batch_mb=512)
        logger.info("UMT5 权重加载成功: %d tensors", len(weights))
    except Exception as exc:
        logger.warning("UMT5 权重加载失败 (strict=%s): %s", strict, exc)
        if strict:
            encoder.load_weights(list(weights.items()), strict=False)
            _eval_params_batched(encoder, batch_mb=512)

    return encoder


def _eval_params_batched(module, batch_mb: int = 512) -> None:
    """分批 eval 模块参数, 避一次性提交大体积致 GPU Command Buffer Timeout.

    按 batch_mb 切分参数列表, 每批 ≤ batch_mb MB, 逐批 mx.eval.
    """
    import mlx.core as mx
    params = module.parameters()
    if not params:
        return
    max_bytes = batch_mb * 1024**2
    cur = []; cur_bytes = 0
    for k, v in params.items():
        nb = v.nbytes if hasattr(v, 'nbytes') else 0
        if cur_bytes + nb > max_bytes and cur:
            mx.eval(cur); cur = []; cur_bytes = 0
        cur.append(v); cur_bytes += nb
    if cur:
        mx.eval(cur)


# ---------------------------------------------------------------------------
# safetensors 加载辅助
# ---------------------------------------------------------------------------
def _load_safetensors_dir(weights_dir: Path) -> dict[str, mx.array]:
    """从目录加载所有 safetensors 文件, 合并为 dict.

    支持两种路径:
      1. 目录: 加载所有 *.safetensors
      2. 单文件: 加载该文件

    Args:
        weights_dir: 目录或文件路径

    Returns:
        {weight_name: mx.array} 字典
    """
    weights_dir = Path(weights_dir)
    if not weights_dir.exists():
        logger.warning("权重路径不存在: %s", weights_dir)
        return {}

    # 单文件
    if weights_dir.is_file() and weights_dir.suffix == ".safetensors":
        return dict(mx.load(str(weights_dir)))

    # 目录: 寻找 *.safetensors
    if weights_dir.is_dir():
        safetensor_files = sorted(weights_dir.glob("*.safetensors"))
        if not safetensor_files:
            # 寻找子目录 (e.g. dit/, vae/, text_encoder/)
            for sub in ("dit", "dit-", "transformer", "vae", "text_encoder", "umt5"):
                sub_dir = weights_dir / sub
                if sub_dir.exists():
                    safetensor_files.extend(sorted(sub_dir.glob("*.safetensors")))

        if not safetensor_files:
            return {}

        all_weights: dict[str, mx.array] = {}
        for f in safetensor_files:
            try:
                w = dict(mx.load(str(f)))
                all_weights.update(w)
                logger.debug("加载 %s: %d tensors", f.name, len(w))
            except Exception as exc:
                logger.warning("加载 %s 失败: %s", f, exc)
        return all_weights

    return {}


# ---------------------------------------------------------------------------
# 综合加载入口
# ---------------------------------------------------------------------------
def load_all_weights(
    dit: nn.Module,
    vae: nn.Module,
    text_encoder: nn.Module,
    model_path: Path,
    *,
    quantization: dict | None = None,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """综合加载三大模型权重.

    Args:
        dit: DiT 主干
        vae: VAE 解码器
        text_encoder: UMT5 文本编码器
        model_path: 模型根目录 (含 dit/ vae/ text_encoder/ 子目录)
        quantization: 可选 DiT 量化配置

    Returns:
        (dit, vae, text_encoder) 加载权重后的三元组
    """
    model_path = Path(model_path)

    # 1. DiT 权重
    dit_dir = model_path / "dit"
    if not dit_dir.exists() or not list(dit_dir.glob("*.safetensors")):
        dit_dir = model_path  # 兜底: 权重在根目录
    load_dit_weights(dit, dit_dir, quantization=quantization)

    # 2. VAE 权重
    vae_dir = model_path / "vae"
    if not vae_dir.exists() or not list(vae_dir.glob("*.safetensors")):
        vae_dir = model_path
    load_vae_weights(vae, vae_dir)

    # 3. UMT5 权重
    t5_dir = model_path / "text_encoder"
    if not t5_dir.exists() or not list(t5_dir.glob("*.safetensors")):
        t5_dir = model_path / "umt5"
    if not t5_dir.exists() or not list(t5_dir.glob("*.safetensors")):
        t5_dir = model_path / "t5"  # SkyReels-V3 真实布局: t5/ 子目录
    if not t5_dir.exists() or not list(t5_dir.glob("*.safetensors")):
        t5_dir = model_path
    load_text_encoder_weights(text_encoder, t5_dir)

    logger.info("SkyReels-V3 全部权重加载完成: %s", model_path)
    return dit, vae, text_encoder


__all__ = [
    "resolve_model_path",
    "load_dit_weights",
    "load_vae_weights",
    "load_text_encoder_weights",
    "load_all_weights",
]
