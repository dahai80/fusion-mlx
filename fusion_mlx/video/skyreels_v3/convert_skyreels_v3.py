# SPDX-License-Identifier: Apache-2.0
"""SkyReels-V3 权重转换器: PyTorch state_dict -> MLX safetensors.

支持三大主干:
  - r2v_14b: Reference-to-Video 14B-720P (SkyReelsA2WanI2v3DModel)
  - v2v_14b: Video Extension 14B-720P (WanModel i2v)
  - a2v_19b: Talking Avatar 19B-720P (WanModel a2v + 音频分支)

分层映射要点:
  - Conv3d patch_embedding: 权重 [out,in,t,h,w] -> MLX Conv3d [out,in,t*h*w]
  - QKV Linear: 权重转置 (PyTorch [out,in] -> MLX [in,out])
  - modulation (1,6,dim) / (1,2,dim): 直接转 mx.array, 保 float32
  - freqs (rope buffer): 转 mx.array 常量
  - 时序维度参数 (num_frame_list/grid_size_list/context_window_size):
    非权重, 由 config 持有

增量加载: .safetensors 分块写, 单次载入 < 4GB, 防止 M5 统一内存冲高.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 模型类型注册
# ---------------------------------------------------------------------------
MODEL_TYPES: dict[str, dict[str, Any]] = {
    "r2v_14b": {
        "src_module": "skyreels_v3.modules.reference_to_video.transformer",
        "src_class": "SkyReelsA2WanI2v3DModel",
        "dim": 5120,
        "ffn_dim": 13824,
        "num_heads": 40,
        "num_layers": 40,
        "patch_size": (1, 2, 2),
        "in_dim": 16,
        "out_dim": 16,
        "text_dim": 4096,
        "text_len": 512,
        "window_size": (-1, -1),
        "model_type": "i2v",
        "cross_attn_type": "i2v_cross_attn",
        "has_audio": False,
    },
    "v2v_14b": {
        "src_module": "skyreels_v3.modules.transformer",
        "src_class": "WanModel",
        "dim": 5120,
        "ffn_dim": 13824,
        "num_heads": 40,
        "num_layers": 40,
        "patch_size": (1, 2, 2),
        "in_dim": 16,
        "out_dim": 16,
        "text_dim": 4096,
        "text_len": 512,
        "window_size": (-1, -1),
        "model_type": "i2v",
        "cross_attn_type": "i2v_cross_attn",
        "has_audio": False,
    },
    "a2v_19b": {
        "src_module": "skyreels_v3.modules.transformer_a2v",
        "src_class": "WanModel",
        "dim": 6144,
        "ffn_dim": 24576,
        "num_heads": 48,
        "num_layers": 60,
        "patch_size": (1, 2, 2),
        "in_dim": 16,
        "out_dim": 16,
        "text_dim": 4096,
        "text_len": 512,
        "window_size": (-1, -1),
        "model_type": "i2v",
        "cross_attn_type": "i2v_cross_attn",
        "has_audio": True,
    },
}


# ---------------------------------------------------------------------------
# PyTorch state_dict 加载 (无 torch 依赖时用 safetensors 直接读)
# ---------------------------------------------------------------------------
def load_pytorch_state_dict(checkpoint_path: str | Path) -> dict[str, Any]:
    """加载 PyTorch checkpoint, 返回 state_dict (numpy arrays).

    优先用 safetensors 直接读取 (避免 torch 依赖);
    fallback 用 torch.load (需要 torch 安装).
    """
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        # 寻找目录下的 safetensors 或 bin 文件 (含子目录 transformer/ vae/ text_encoder/ 等)
        safetensor_files = sorted(checkpoint_path.glob("*.safetensors"))
        bin_files = sorted(checkpoint_path.glob("*.bin"))
        # 递归子目录 (SkyReels-V3 HF 真实布局: transformer/*.safetensors, vae/*.safetensors, text_encoder/*.safetensors)
        if not safetensor_files:
            safetensor_files = sorted(checkpoint_path.rglob("*.safetensors"))
        if not bin_files:
            bin_files = sorted(checkpoint_path.rglob("*.bin"))
        if safetensor_files:
            return _load_safetensors_dir(safetensor_files)
        if bin_files:
            return _load_bin_files(bin_files)
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")
    elif checkpoint_path.is_file():
        if checkpoint_path.suffix == ".safetensors":
            return _load_safetensors_file(checkpoint_path)
        if checkpoint_path.suffix in (".bin", ".pt", ".pth"):
            return _load_torch_file(checkpoint_path)
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def _load_safetensors_dir(files: list[Path]) -> dict[str, Any]:
    """从多个 safetensors 文件加载, 合并 state_dict.

    优先用 framework="pt" (torch 支持 bfloat16);
    fallback framework="numpy" 但 bfloat16 需 upcast 到 float32.
    """
    from safetensors import safe_open

    state_dict: dict[str, Any] = {}
    # 尝试 torch 路径 (支持 bfloat16)
    try:
        import torch  # noqa: F401

        use_pt = True
    except ImportError:
        use_pt = False

    for f in files:
        logger.info("Loading %s ...", f.name)
        fw = "pt" if use_pt else "numpy"
        with safe_open(str(f), framework=fw, device="cpu") as st:
            for key in st:
                tensor = st.get_tensor(key)
                if (
                    not use_pt
                    and hasattr(tensor, "dtype")
                    and "bfloat16" in str(getattr(tensor, "dtype", ""))
                ):
                    # numpy 不认 bfloat16, 上转 float32
                    tensor = (
                        tensor.astype("float32")
                        if hasattr(tensor, "astype")
                        else tensor
                    )
                if use_pt:
                    # torch.Tensor → numpy (bfloat16 先 upcast float32 保精度, 其余原 dtype)
                    import torch as _torch

                    if hasattr(tensor, "detach"):
                        tensor = tensor.detach().cpu()
                        if tensor.dtype == _torch.bfloat16:
                            tensor = tensor.to(_torch.float32)
                        tensor = tensor.numpy()
                state_dict[key] = tensor
    return state_dict


def _load_safetensors_file(file: Path) -> dict[str, Any]:
    from safetensors import safe_open

    state_dict: dict[str, Any] = {}
    with safe_open(str(file), framework="numpy", device="cpu") as st:
        for key in st:
            state_dict[key] = st.get_tensor(key)
    return state_dict


def _load_bin_files(files: list[Path]) -> dict[str, Any]:
    """Fallback: 用 torch.load 加载 .bin/.pt/.pth 文件."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required to load .bin/.pt/.pth checkpoints. "
            "Install with: pip install torch"
        ) from exc

    state_dict: dict[str, Any] = {}
    for f in files:
        logger.info("Loading %s ...", f.name)
        raw = torch.load(str(f), map_location="cpu")
        if isinstance(raw, dict):
            # 处理嵌套 state_dict (e.g. {"state_dict": ...})
            if "state_dict" in raw:
                raw = raw["state_dict"]
            for k, v in raw.items():
                state_dict[k] = v.detach().cpu().numpy()
    return state_dict


def _load_torch_file(file: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required to load .bin/.pt/.pth checkpoints."
        ) from exc

    raw = torch.load(str(file), map_location="cpu")
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    return {k: v.detach().cpu().numpy() for k, v in raw.items()}


# ---------------------------------------------------------------------------
# 权重名映射: PyTorch state_dict key -> MLX module path
# ---------------------------------------------------------------------------
def _map_linear_weight(pt_key: str, weight: np.ndarray) -> tuple[str, mx.array]:
    """PyTorch Linear weight [out, in] -> MLX [in, out] (转置).

    MLX nn.Linear 期望 weight shape [input_dims, output_dims].
    """
    mlx_key = pt_key.replace(".weight", ".weight")  # 保持 key 名
    mlx_weight = mx.array(weight.T)  # 转置
    return mlx_key, mlx_weight


def _map_conv3d_weight(
    pt_key: str, weight: np.ndarray, patch_size: tuple
) -> tuple[str, mx.array]:
    """PyTorch Conv3d weight [out, in, t, h, w] -> MLX Conv3d [out, in, t*h*w].

    MLX Conv3d 期望 weight shape [out_channels, in_channels, kernel_t*kernel_h*kernel_w].
    """
    # weight shape: [out, in, t, h, w]
    out_c, in_c, kt, kh, kw = weight.shape
    # 重排为 [out, in, t*h*w]  (kernel 展开顺序: t -> h -> w)
    reshaped = weight.reshape(out_c, in_c, kt * kh * kw)
    mlx_weight = mx.array(reshaped)
    return pt_key, mlx_weight


def _map_modulation(pt_key: str, weight: np.ndarray) -> tuple[str, mx.array]:
    """modulation 参数 [1, 6, dim] 或 [1, 2, dim], 保 float32."""
    mlx_weight = mx.array(weight).astype(mx.float32)
    return pt_key, mlx_weight


def _map_norm_weight(pt_key: str, weight: np.ndarray) -> tuple[str, mx.array]:
    """LayerNorm/RMSNorm weight/bias, 直接转."""
    return pt_key, mx.array(weight)


def _map_buffer(pt_key: str, weight: np.ndarray) -> tuple[str, mx.array]:
    """freqs 等 buffer, 转常量."""
    return pt_key, mx.array(weight)


# ---------------------------------------------------------------------------
# 核心转换函数
# ---------------------------------------------------------------------------
def convert_skyreels_v3(
    checkpoint_path: str | Path,
    mlx_out_dir: str | Path,
    model_type: str = "r2v_14b",
    *,
    dtype: mx.Dtype = mx.bfloat16,
    quantization_bits: int = 0,
    max_shard_size_mb: int = 4096,
) -> Path:
    """转换 SkyReels-V3 PyTorch checkpoint -> MLX safetensors.

    Args:
        checkpoint_path: PyTorch checkpoint 路径 (目录或文件).
        mlx_out_dir: MLX 权重输出目录.
        model_type: 模型类型 (r2v_14b / v2v_14b / a2v_19b).
        dtype: 权重 dtype (默认 bfloat16).
        quantization_bits: 量化位数 (0=不量化, 4=NF4, 8=FP8).
        max_shard_size_mb: 单个 shard 最大大小 (MB), 控制增量加载.

    Returns:
        输出目录 Path.
    """
    if model_type not in MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type: {model_type}. Valid: {list(MODEL_TYPES)}"
        )

    config = MODEL_TYPES[model_type]
    mlx_out_dir = Path(mlx_out_dir)
    mlx_out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Converting SkyReels-V3 [%s] ...", model_type)
    logger.info("  Source: %s", checkpoint_path)
    logger.info("  Output: %s", mlx_out_dir)

    # 1) 加载 PyTorch state_dict
    pt_state_dict = load_pytorch_state_dict(checkpoint_path)
    logger.info("  Loaded %d keys from PyTorch checkpoint", len(pt_state_dict))

    # 2) 转换为 MLX weights (分层映射)
    mlx_weights: dict[str, mx.array] = {}
    patch_size = config["patch_size"]

    for pt_key, np_weight in pt_state_dict.items():
        mlx_key, mlx_weight = _convert_one_key(pt_key, np_weight, patch_size, config)
        if mlx_weight is not None:
            mlx_weights[mlx_key] = mlx_weight.astype(dtype)

    logger.info("  Converted %d MLX weight tensors", len(mlx_weights))

    # 3) 写入 config.json
    config_data = {
        "model_type": model_type,
        "arch": "skyreels_v3",
        "config": config,
        "dtype": str(dtype),
        "quantization_bits": quantization_bits,
    }
    with open(mlx_out_dir / "config.json", "w") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    # 4) 分块写入 safetensors (增量加载, 控制单次 < 4GB)
    _write_sharded_safetensors(mlx_weights, mlx_out_dir, max_shard_size_mb)

    logger.info("  Done. Output: %s", mlx_out_dir)
    return mlx_out_dir


def _convert_one_key(
    pt_key: str,
    np_weight: np.ndarray,
    patch_size: tuple,
    config: dict,
) -> tuple[str, mx.array | None]:
    """转换单个权重 key, 返回 (mlx_key, mlx_weight).

    分层映射规则:
      - patch_embedding.weight (Conv3d): 转置 + reshape
      - *.weight (Linear): 转置
      - *.bias: 直接转
      - modulation: 保 float32
      - norm.weight / norm.bias: 直接转
      - freqs (rope buffer): 转 mx.array
    """
    # 跳过空权重
    if np_weight is None:
        return pt_key, None

    # Conv3d patch_embedding
    if "patch_embedding" in pt_key and ".weight" in pt_key:
        return _map_conv3d_weight(pt_key, np_weight, patch_size)

    # modulation 参数 (AdaLN-Zero 核心, 保 float32)
    if "modulation" in pt_key:
        return _map_modulation(pt_key, np_weight)

    # norm 参数
    if ".norm" in pt_key or "norm_q" in pt_key or "norm_k" in pt_key:
        return _map_norm_weight(pt_key, np_weight)

    # rope freqs buffer
    if "freqs" in pt_key:
        return _map_buffer(pt_key, np_weight)

    # Linear weight (转置)
    if pt_key.endswith(".weight"):
        # 检查是否是 4D (卷积) - Conv3d 已处理, 这里处理 2D Linear
        if np_weight.ndim == 4:
            # 可能是 Conv2d, 按卷积处理
            out_c, in_c, kh, kw = np_weight.shape
            reshaped = np_weight.reshape(out_c, in_c, kh * kw)
            return pt_key, mx.array(reshaped)
        elif np_weight.ndim == 2:
            return _map_linear_weight(pt_key, np_weight)
        elif np_weight.ndim == 1:
            # 1D 权重 (e.g. embedding), 直接转
            return pt_key, mx.array(np_weight)
        else:
            logger.warning("  Unhandled weight ndim=%d: %s", np_weight.ndim, pt_key)
            return pt_key, mx.array(np_weight)

    # bias
    if pt_key.endswith(".bias"):
        return pt_key, mx.array(np_weight)

    # 其他 (embedding 等)
    return pt_key, mx.array(np_weight)


# ---------------------------------------------------------------------------
# 分块写入 safetensors
# ---------------------------------------------------------------------------
def _write_sharded_safetensors(
    weights: dict[str, mx.array],
    out_dir: Path,
    max_shard_size_mb: int,
) -> None:
    """分块写入 safetensors, 控制单 shard 大小.

    Args:
        weights: MLX 权重字典.
        out_dir: 输出目录.
        max_shard_size_mb: 单 shard 最大大小 (MB).
    """
    from safetensors.numpy import save_file as save_numpy

    max_shard_bytes = max_shard_size_mb * 1024 * 1024
    current_shard: dict[str, np.ndarray] = {}
    current_size = 0
    shard_idx = 0
    weight_map: dict[str, str] = {}  # key -> shard filename

    def _flush_shard() -> None:
        nonlocal current_shard, current_size, shard_idx
        if not current_shard:
            return
        shard_name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        shard_path = out_dir / shard_name
        save_numpy(current_shard, str(shard_path))
        for k in current_shard:
            weight_map[k] = shard_name
        logger.info(
            "  Wrote shard %s (%d tensors, %.1f MB)",
            shard_name,
            len(current_shard),
            current_size / 1024 / 1024,
        )
        current_shard = {}
        current_size = 0
        shard_idx += 1

    for key, mlx_weight in weights.items():
        # 转 numpy for safetensors (bfloat16 先 upcast float32, numpy 不认 bf16 buffer)
        try:
            np_weight = np.array(mlx_weight)
        except (RuntimeError, TypeError):
            # mlx bfloat16 → numpy 不兼容, 上转 float32 保精度
            import mlx.core as mx

            if mlx_weight.dtype == mx.bfloat16:
                np_weight = np.array(mlx_weight.astype(mx.float32))
            else:
                raise
        tensor_bytes = np_weight.nbytes

        if current_size + tensor_bytes > max_shard_bytes and current_shard:
            _flush_shard()

        current_shard[key] = np_weight
        current_size += tensor_bytes

    _flush_shard()

    # 写入 weight_map.json (索引文件)
    with open(out_dir / "weight_map.json", "w") as f:
        json.dump(weight_map, f, indent=2)

    # 写入 model.safetensors.index.json (HuggingFace 兼容格式)
    # bfloat16 先 upcast float32 求 nbytes (numpy 不认 bf16 buffer)
    import mlx.core as mx

    def _safe_nbytes(w):
        try:
            return np.array(w).nbytes
        except (RuntimeError, TypeError):
            return np.array(
                w.astype(mx.float32) if w.dtype == mx.bfloat16 else w
            ).nbytes

    total_size = sum(_safe_nbytes(w) for w in weights.values())
    index_data = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=2)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SkyReels-V3 PyTorch checkpoint to MLX safetensors."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to PyTorch checkpoint (dir or file)"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for MLX weights"
    )
    parser.add_argument(
        "--model-type",
        choices=list(MODEL_TYPES.keys()),
        default="r2v_14b",
        help="Model type (default: r2v_14b)",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Weight dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--quantization-bits",
        type=int,
        default=0,
        help="Quantization bits (0=none, 4=NF4, 8=FP8)",
    )
    parser.add_argument(
        "--max-shard-size-mb",
        type=int,
        default=4096,
        help="Max shard size in MB (default: 4096)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dtype_map = {
        "bfloat16": mx.bfloat16,
        "float16": mx.float16,
        "float32": mx.float32,
    }

    convert_skyreels_v3(
        checkpoint_path=args.checkpoint,
        mlx_out_dir=args.output,
        model_type=args.model_type,
        dtype=dtype_map[args.dtype],
        quantization_bits=args.quantization_bits,
        max_shard_size_mb=args.max_shard_size_mb,
    )


if __name__ == "__main__":
    main()
