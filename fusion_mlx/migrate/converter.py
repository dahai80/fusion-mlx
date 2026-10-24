# SPDX-License-Identifier: Apache-2.0
"""Weight conversion orchestration — HF safetensors to MLX format.

Callers: fusion_mlx.admin.migrate_route
API: convert_model(hf_dir, output_dir, config, template, ...) -> ConvertResult
Schema: ConvertResult(dataclass) — output_dir, num_weights, total_params_b, orphans, missing, error
User instruction verbatim: "做一个端到端的功能，做模型迁移和量化的功能，以openpangu为例，把迁移的每个步骤展现在GUI上"
"""

import json
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open

from .architectures import ArchTemplate
from .weight_mapper import build_weight_map, find_missing_keys, find_orphan_keys

logger = logging.getLogger(__name__)

# Shard cap for the streaming writer: at most this many bytes of converted
# tensors are held in memory at once, so a 70B model no longer peaks at
# ~4x its weight size (pre-H10 it materialized every shard as fp32 in one dict).
_DEFAULT_MAX_SHARD_BYTES = 2 * 1024**3

_QUANT_SKIP_SUFFIXES = ("norm.weight", "embed_tokens.weight", "lm_head.weight")

_CAST_DTYPES = {
    "float32": mx.float32,
    "fp32": mx.float32,
    "f32": mx.float32,
    "float16": mx.float16,
    "fp16": mx.float16,
    "f16": mx.float16,
    "bfloat16": mx.bfloat16,
    "bf16": mx.bfloat16,
}

_SAFETENSORS_ITEMSIZE = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "U16": 2,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


@dataclass
class ConvertResult:
    output_dir: str
    num_weights: int = 0
    total_params_b: float = 0.0
    orphans: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class _SourceTensor:
    path: str
    name: str
    nbytes: int


class _QuantizeFailed(Exception):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name


def _resolve_cast_dtype(dtype: str | None):
    if dtype is None:
        return None
    resolved = _CAST_DTYPES.get(dtype.strip().lower())
    if resolved is None:
        raise ValueError(
            f"Unsupported conversion dtype '{dtype}'; expected one of "
            f"{sorted(_CAST_DTYPES)}"
        )
    logger.info("Operator requested dtype %s — casting every weight", dtype)
    return resolved


def tensor_to_mx(tensor):
    import torch

    if isinstance(tensor, torch.Tensor):
        if tensor.dtype == torch.bfloat16:
            # numpy has no bfloat16; reinterpret the bits instead of
            # round-tripping through fp32 so the dtype is preserved exactly.
            bits = tensor.contiguous().view(torch.uint16).numpy()
            return mx.array(bits).view(mx.bfloat16)
        arr = tensor.numpy()
    else:
        arr = np.asarray(tensor)
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    return mx.array(arr)


def _scan_source(hf_dir: str) -> tuple[list[_SourceTensor], int]:
    paths = sorted(Path(hf_dir).glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"No *.safetensors weight files found in {hf_dir}")
    manifest: list[_SourceTensor] = []
    total_bytes = 0
    for path in paths:
        before = len(manifest)
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():  # noqa: SIM118 - safe_open is not iterable
                sl = f.get_slice(key)
                elements = 1
                for dim in sl.get_shape():
                    elements *= int(dim)
                nbytes = elements * _SAFETENSORS_ITEMSIZE.get(sl.get_dtype(), 1)
                manifest.append(_SourceTensor(str(path), key, nbytes))
                total_bytes += nbytes
        logger.info("Scanned %s: %d tensors", path.name, len(manifest) - before)
    logger.info(
        "Scanned %d tensors (%.2f GiB) across %d shard(s) in %s",
        len(manifest),
        total_bytes / 2**30,
        len(paths),
        hf_dir,
    )
    return manifest, total_bytes


def _iter_source_tensors(manifest: list[_SourceTensor]):
    index = 0
    total = len(manifest)
    while index < total:
        path = manifest[index].path
        end = index
        while end < total and manifest[end].path == path:
            end += 1
        with safe_open(path, framework="pt") as f:
            for entry in manifest[index:end]:
                tensor = f.get_tensor(entry.name)
                arr = tensor_to_mx(tensor)
                del tensor
                yield entry.name, arr
        index = end


def _fsync_file(path: str) -> None:
    # Open the file writable: fsync on a read-only descriptor does not
    # enforce the durability guard this is meant to provide (#811 P0, M24).
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError as e:
        logger.warning("Could not open %s for fsync (non-fatal): %s", path, e)
        return
    try:
        os.fsync(fd)
    except OSError as e:
        logger.warning("fsync of %s failed (non-fatal): %s", path, e)
    finally:
        os.close(fd)


class _ShardWriter:
    def __init__(
        self,
        output_dir: str,
        max_shard_bytes: int,
        progress_cb: Callable[[float, str], None] | None = None,
        total_source_bytes: int = 0,
    ):
        self._output_dir = output_dir
        self._max_bytes = max(int(max_shard_bytes), 1)
        self._progress_cb = progress_cb
        self._total_source_bytes = max(int(total_source_bytes), 1)
        self._pending: dict[str, mx.array] = {}
        self._pending_bytes = 0
        self._paths: list[str] = []
        self._tensor_shard: dict[str, int] = {}
        self._written_bytes = 0
        self._num_tensors = 0
        self._num_elements = 0

    @property
    def num_tensors(self) -> int:
        return self._num_tensors

    @property
    def num_elements(self) -> int:
        return self._num_elements

    def paths(self) -> list[str]:
        return list(self._paths)

    def add(self, name: str, tensor: mx.array) -> None:
        nbytes = int(tensor.nbytes)
        if self._pending and self._pending_bytes + nbytes > self._max_bytes:
            self.flush()
        self._pending[name] = tensor
        self._pending_bytes += nbytes
        self._tensor_shard[name] = len(self._paths)
        self._num_tensors += 1
        self._num_elements += int(np.prod(tensor.shape))
        if self._pending_bytes >= self._max_bytes:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        path = os.path.join(
            self._output_dir, f"model-{len(self._paths):05d}.safetensors"
        )
        mx.save_safetensors(path, self._pending)
        _fsync_file(path)
        self._paths.append(path)
        self._written_bytes += self._pending_bytes
        self._pending.clear()
        self._pending_bytes = 0
        logger.info(
            "Wrote shard %s (%d tensors so far, %.2f GiB)",
            os.path.basename(path),
            self._num_tensors,
            self._written_bytes / 2**30,
        )
        if self._progress_cb is not None:
            fraction = min(self._written_bytes / self._total_source_bytes, 1.0)
            self._progress_cb(
                0.25 + 0.65 * fraction,
                f"Converting weights ({self._num_tensors} tensors)",
            )

    def abort(self) -> None:
        self._pending.clear()
        for path in self._paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.warning("Could not remove partial shard %s: %s", path, e)
        self._paths.clear()
        self._tensor_shard.clear()

    def finish(self) -> list[str]:
        self.flush()
        if not self._paths:
            # Empty model (no mapped tensors): still emit a valid, empty
            # safetensors file so callers see a consistent directory layout.
            empty_path = os.path.join(self._output_dir, "model.safetensors")
            mx.save_safetensors(empty_path, {})
            _fsync_file(empty_path)
            self._paths.append(empty_path)
        if len(self._paths) == 1:
            final_path = os.path.join(self._output_dir, "model.safetensors")
            if self._paths[0] != final_path:
                os.replace(self._paths[0], final_path)
            self._paths = [final_path]
            return ["model.safetensors"]

        total = len(self._paths)
        names = []
        for index, path in enumerate(self._paths):
            name = f"model-{index + 1:05d}-of-{total:05d}.safetensors"
            os.replace(path, os.path.join(self._output_dir, name))
            names.append(name)
        weight_map = {name: names[idx] for name, idx in self._tensor_shard.items()}
        index_doc = {
            "metadata": {"total_size": self._written_bytes},
            "weight_map": weight_map,
        }
        index_path = os.path.join(self._output_dir, "model.safetensors.index.json")
        with open(index_path, "w") as f:
            json.dump(index_doc, f, indent=2)
        _fsync_file(index_path)
        self._paths = [os.path.join(self._output_dir, name) for name in names]
        return names


def write_safetensors_stream(
    output_dir: str,
    tensors,
    max_shard_bytes: int = _DEFAULT_MAX_SHARD_BYTES,
) -> list[str]:
    writer = _ShardWriter(output_dir, max_shard_bytes)
    try:
        for name, tensor in tensors:
            writer.add(name, tensor)
        return writer.finish()
    except Exception:
        writer.abort()
        raise


def _quantize_tensor(
    mlx_name: str,
    tensor: mx.array,
    quant_bits: int,
    quant_group_size: int,
) -> dict[str, mx.array] | None:
    if any(mlx_name.endswith(p) for p in _QUANT_SKIP_SUFFIXES) or len(tensor.shape) < 2:
        return {mlx_name: tensor}

    base_name = mlx_name
    if base_name.endswith(".weight"):
        base_name = base_name[: -len(".weight")]

    try:
        q_weight, q_scale, q_bias = mx.quantize(
            tensor,
            group_size=quant_group_size,
            bits=quant_bits,
        )
    except Exception:
        logger.warning("Quantize failed for %s", mlx_name, exc_info=True)
        return None

    return {
        base_name + ".weight": q_weight,
        base_name + ".scales": q_scale,
        base_name + ".biases": q_bias,
    }


def _stream_pass(
    manifest: list[_SourceTensor],
    weight_map: dict[str, str],
    output_dir: str,
    quant_bits: int,
    quant_group_size: int,
    cast_dtype,
    max_shard_bytes: int,
    progress_cb: Callable[[float, str], None] | None,
    total_source_bytes: int,
) -> _ShardWriter:
    writer = _ShardWriter(
        output_dir,
        max_shard_bytes,
        progress_cb=progress_cb,
        total_source_bytes=total_source_bytes,
    )
    try:
        for hf_name, tensor in _iter_source_tensors(manifest):
            mlx_name = weight_map.get(hf_name)
            if mlx_name is None:
                # Orphan key; reported via result.orphans.
                continue
            if cast_dtype is not None:
                tensor = tensor.astype(cast_dtype)
            if quant_bits > 0:
                quantized = _quantize_tensor(
                    mlx_name, tensor, quant_bits, quant_group_size
                )
                if quantized is None:
                    raise _QuantizeFailed(hf_name)
                for name, arr in quantized.items():
                    writer.add(name, arr)
            else:
                writer.add(mlx_name, tensor)
        writer.finish()
        return writer
    except Exception:
        writer.abort()
        raise


def _write_weights_streamed(
    manifest: list[_SourceTensor],
    weight_map: dict[str, str],
    output_dir: str,
    quant_bits: int,
    quant_group_size: int,
    cast_dtype,
    max_shard_bytes: int,
    progress_cb: Callable[[float, str], None] | None,
    total_source_bytes: int,
) -> tuple[_ShardWriter, bool]:
    if quant_bits <= 0:
        writer = _stream_pass(
            manifest,
            weight_map,
            output_dir,
            0,
            quant_group_size,
            cast_dtype,
            max_shard_bytes,
            progress_cb,
            total_source_bytes,
        )
        return writer, True

    try:
        writer = _stream_pass(
            manifest,
            weight_map,
            output_dir,
            quant_bits,
            quant_group_size,
            cast_dtype,
            max_shard_bytes,
            progress_cb,
            total_source_bytes,
        )
        return writer, True
    except _QuantizeFailed as exc:
        # #1059: a half-quantized model mis-loads (some weights carry
        # .scales/.biases, others do not). Drop quantization for the whole
        # model: discard any shards already written and re-stream the
        # source un-quantized so config + weights stay consistent.
        logger.warning(
            "Quantize failed for %s — dropping quantization entirely to "
            "avoid a half-quantized model (config/weight mismatch on "
            "load). Re-streaming weights without quantization.",
            exc.name,
        )
        writer = _stream_pass(
            manifest,
            weight_map,
            output_dir,
            0,
            quant_group_size,
            cast_dtype,
            max_shard_bytes,
            progress_cb,
            total_source_bytes,
        )
        return writer, False


def _build_mlx_config(
    hf_config: dict,
    template: ArchTemplate,
) -> dict:
    mlx_config = {
        "model_type": template.name,
        "model_file": f"{template.name}.py",
        "num_hidden_layers": hf_config.get(
            "num_hidden_layers", hf_config.get("n_layer", 0)
        ),
        "hidden_size": hf_config.get("hidden_size", hf_config.get("d_model", 0)),
        "intermediate_size": hf_config.get("intermediate_size", 0),
        "num_attention_heads": hf_config.get("num_attention_heads", 0),
        "num_key_value_heads": hf_config.get(
            "num_key_value_heads", hf_config.get("num_attention_heads", 0)
        ),
        "rms_norm_eps": hf_config.get("rms_norm_eps", 1e-6),
        "vocab_size": hf_config.get("vocab_size", 0),
        "tie_word_embeddings": hf_config.get("tie_word_embeddings", False),
    }

    rope_theta = hf_config.get("rope_theta")
    if rope_theta:
        mlx_config["rope_theta"] = rope_theta

    rope_traditional = hf_config.get("rope_traditional", False)
    mlx_config["rope_traditional"] = rope_traditional

    if template.has_bias:
        mlx_config["bias"] = True
    if template.has_mlp_bias:
        mlx_config["mlp_bias"] = True

    max_position_embeddings = hf_config.get("max_position_embeddings")
    if max_position_embeddings:
        mlx_config["max_position_embeddings"] = max_position_embeddings

    return mlx_config


def convert_model(
    hf_dir: str,
    output_dir: str,
    config: dict,
    template: ArchTemplate,
    quant_bits: int = 0,
    quant_group_size: int = 64,
    progress_cb: Callable[[float, str], None] | None = None,
    dtype: str | None = None,
    max_shard_bytes: int = _DEFAULT_MAX_SHARD_BYTES,
) -> ConvertResult:
    result = ConvertResult(output_dir=output_dir)

    tmp_dir: str | None = None
    try:
        # #1059: if num_hidden_layers resolved to 0, the weight map only
        # contains embed/norm/lm_head — ALL per-layer weights become
        # orphans. Writing this to disk produces a 0-layer "model" that
        # mlx_lm.load accepts as valid but generates garbage. Hard-fail
        # before touching the source instead of warning + returning success.
        num_layers = config.get("num_hidden_layers", config.get("n_layer", 0))
        if num_layers == 0:
            result.error = (
                "num_hidden_layers resolved to 0 — cannot build a valid "
                "weight map (no per-layer entries). Check the HF config "
                "for 'num_hidden_layers' or 'n_layer'."
            )
            logger.error("Conversion aborted: %s", result.error)
            return result

        cast_dtype = _resolve_cast_dtype(dtype)

        if progress_cb:
            progress_cb(0.0, "Scanning HF weights")
        manifest, total_source_bytes = _scan_source(hf_dir)
        hf_keys = [entry.name for entry in manifest]

        if progress_cb:
            progress_cb(0.15, "Building weight map")
        weight_map = build_weight_map(config, template)
        result.orphans = find_orphan_keys(hf_keys, weight_map)
        result.missing = find_missing_keys(weight_map, hf_keys)
        if result.missing:
            logger.warning(
                "Missing %d expected keys — conversion may be incomplete",
                len(result.missing),
            )

        # Atomic-ish write: build the full output tree in a sibling temp dir,
        # then swap it into output_dir on success. A crash mid-write leaves
        # the temp dir (cleaned below) rather than a half-written
        # model.safetensors + config.json that mlx_lm.load would silently
        # load as a valid but corrupt model (#811 P0).
        os.makedirs(os.path.dirname(os.path.abspath(output_dir)) or ".", exist_ok=True)
        tmp_dir = f"{output_dir}.tmp.{os.getpid()}"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        if progress_cb:
            progress_cb(0.25, "Streaming weights")
        logger.info(
            "Streaming conversion of %d tensors (%.2f GiB) into %s",
            len(manifest),
            total_source_bytes / 2**30,
            tmp_dir,
        )
        writer, quant_ok = _write_weights_streamed(
            manifest,
            weight_map,
            tmp_dir,
            quant_bits,
            quant_group_size,
            cast_dtype,
            max_shard_bytes,
            progress_cb,
            total_source_bytes,
        )
        logger.info(
            "Wrote %d tensors into %d safetensors shard(s)",
            writer.num_tensors,
            len(writer.paths()),
        )

        mlx_config = _build_mlx_config(config, template)
        # #1059: if quantization was requested but partially failed
        # (_write_weights_streamed returned all_ok=False), the weights are
        # un-quantized. Strip any quantization-related fields from config
        # so mlx_lm.load doesn't expect .scales/.biases keys.
        if quant_bits > 0 and not quant_ok:
            mlx_config.pop("quantization", None)
            mlx_config.pop("quant_bits", None)
            mlx_config.pop("quant_group_size", None)
            logger.warning(
                "Quantization requested (bits=%d) but failed — writing "
                "un-quantized model with no quantization config",
                quant_bits,
            )
        config_path = os.path.join(tmp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(mlx_config, f, indent=2)
        _fsync_file(config_path)
        logger.info("Wrote config.json to %s", tmp_dir)

        tokenizer_src = Path(hf_dir)
        for tok_name in (
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
        ):
            src = tokenizer_src / tok_name
            if src.exists():
                shutil.copy2(str(src), os.path.join(tmp_dir, tok_name))
                logger.info("Copied %s", tok_name)
        for custom_tok in sorted(tokenizer_src.glob("tokenization_*.py")):
            shutil.copy2(str(custom_tok), os.path.join(tmp_dir, custom_tok.name))
            logger.info("Copied custom tokenizer %s", custom_tok.name)

        # #1058: swap via rename-to-backup so a failed rename does NOT
        # destroy the previously-available output_dir. The old code did
        # rmtree(output_dir) then rename(tmp_dir, output_dir) — if rmtree
        # succeeded but rename failed (cross-device / permission), the
        # old model was already gone and tmp_dir was cleaned by the outer
        # finally, losing everything. Now: rename old → backup, rename
        # tmp → final, rmtree backup only after the swap succeeds.
        backup_dir = f"{output_dir}.old.{os.getpid()}"
        if os.path.exists(output_dir):
            os.rename(output_dir, backup_dir)
        try:
            os.rename(tmp_dir, output_dir)
            tmp_dir = None  # consumed
        except OSError:
            # rename failed — restore old output if we moved it.
            if os.path.exists(backup_dir) and not os.path.exists(output_dir):
                try:
                    os.rename(backup_dir, output_dir)
                except OSError:
                    logger.error(
                        "Failed to restore backup %s after rename failure",
                        backup_dir,
                        exc_info=True,
                    )
            raise
        # Swap succeeded — safe to remove the old output.
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir, ignore_errors=True)
        logger.info("Atomically installed converted model to %s", output_dir)

        result.num_weights = writer.num_tensors
        result.total_params_b = float(writer.num_elements) / 1e9

        if progress_cb:
            progress_cb(1.0, "Conversion complete")

    except Exception as e:
        logger.exception("Conversion failed: %s", e)
        result.error = str(e)
    finally:
        # Clean up a half-written temp dir from a failed conversion so it is
        # not mistaken for a valid model on the next run.
        if tmp_dir is not None and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return result
