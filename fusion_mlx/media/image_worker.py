#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Image generation subprocess worker (S3, audit 0910 §6.2).
#
# Isolated subprocess that loads an mflux model, generates image(s), writes
# output to temp files, and exits. Memory (weights + activations + compile
# cache) is fully released on process exit — the main LLM process is never
# touched by the image Metal allocator.
#
# Protocol (stdout JSONL, one event per line):
#   {"type":"progress","image":0,"step":1,"total":30}
#   {"type":"result","outputs":["/tmp/fusion_img_xxx/0.png"]}
#   {"type":"error","message":"..."}
#
# Job spec (argv --spec <path> or stdin): JSON with fields:
#   variant, model_path, quantize, config_label, output_dir,
#   output_format, n_images, gen_params (dict for build_gen_kwargs)
#
# Exit codes: 0=ok, 137=OOM (jetsam), 139=segfault/Metal crash, 1=other error.
import json
import os
import sys
import traceback


def _emit(event: dict) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def _setup_mlx() -> None:
    import mlx.core as mx

    cache_limit = int(os.environ.get("FUSION_IMAGE_CACHE_LIMIT", str(1 << 30)))
    try:
        mx.metal.set_cache_limit(cache_limit)
    except Exception:
        pass
    wired = os.environ.get("FUSION_IMAGE_WIRED_LIMIT_MB")
    if wired:
        try:
            mx.metal.set_wired_limit(int(wired) * (1 << 20))
        except Exception:
            pass


class _ProgressEmitter:
    def __init__(self, image_idx: int, total: int):
        self._image_idx = image_idx
        self._total = total
        self._count = 0

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps) -> None:
        self._count += 1
        _emit(
            {
                "type": "progress",
                "image": self._image_idx,
                "step": self._count,
                "total": self._total,
            }
        )


def _load_model(variant: str, model_path: str, quantize, config_label):
    import importlib

    from fusion_mlx.engines.image_gen import VARIANT_MAP, _infer_flux2_config

    module_path, cls_name, default_label, _ = VARIANT_MAP[variant]
    if variant == "txt2img":
        config_label = config_label or _infer_flux2_config(model_path)
    else:
        config_label = config_label or default_label

    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    if module_path.startswith("fusion_mlx."):
        flux = cls(model_config=None, model_path=model_path, quantize=quantize)
    else:
        from mflux.models.common.config.model_config import ModelConfig

        mc = getattr(ModelConfig, config_label)()
        flux = cls(model_config=mc, model_path=model_path, quantize=quantize)
    return flux


def _save_image(gen, output_path: str, output_format: str) -> str:
    if output_format == "raw":
        import numpy as np

        arr = np.array(gen.image)
        np.save(output_path + ".npy", arr)
        return output_path + ".npy"
    else:
        fmt = output_format if output_format != "raw" else "PNG"
        gen.image.save(output_path, format=fmt)
        return output_path


def run_worker(spec: dict) -> int:
    _setup_mlx()
    variant = spec["variant"]
    model_path = spec["model_path"]
    quantize = spec.get("quantize")
    config_label = spec.get("config_label")
    output_dir = spec["output_dir"]
    output_format = spec.get("output_format", "PNG")
    n_images = max(1, spec.get("n_images", 1))
    gen_params = spec["gen_params"]
    os.makedirs(output_dir, exist_ok=True)

    flux = _load_model(variant, model_path, quantize, config_label)

    # S4: apply CFG-free patch for Qwen-Image variants
    if variant in ("qwen_image", "qwen_image_edit"):
        from fusion_mlx.engines._qwen_image_cfg_patch import apply_qwen_image_cfg_patch

        apply_qwen_image_cfg_patch(flux)

    from fusion_mlx.engines._gen_kwargs import build_gen_kwargs

    ext = (
        "npy"
        if output_format == "raw"
        else (output_format.lower() if output_format != "raw" else "png")
    )
    outputs = []
    for i in range(n_images):
        gk = build_gen_kwargs(
            variant=variant,
            seed=gen_params["seed"] + i,
            **{k: v for k, v in gen_params.items() if k != "seed"},
        )
        steps = gk.get("num_inference_steps", 4)
        subscriber = None
        if getattr(flux, "callbacks", None) is not None:
            subscriber = _ProgressEmitter(i, steps)
            try:
                flux.callbacks.register(subscriber)
            except Exception:
                subscriber = None
        try:
            gen = flux.generate_image(**gk)
        finally:
            if subscriber is not None:
                try:
                    flux.callbacks.in_loop.remove(subscriber)
                except (ValueError, AttributeError):
                    pass
        out_path = os.path.join(output_dir, f"{i}.{ext}")
        _save_image(gen, out_path, output_format)
        outputs.append(out_path)
        # Per-image cache clear (flyto pattern)
        try:
            import gc

            import mlx.core as mx

            gc.collect()
            mx.metal.clear_cache()
        except Exception:
            pass

    _emit({"type": "result", "outputs": outputs})
    return 0


def main() -> int:
    # Programmatic entry: spec path as argv[1] (positional), or stdin.
    # No argparse — this is invoked by MediaJobManager, not users.
    spec_path = sys.argv[1] if len(sys.argv) > 1 else None
    if spec_path:
        with open(spec_path) as f:
            spec = json.load(f)
    else:
        spec = json.loads(sys.stdin.read())
    try:
        return run_worker(spec)
    except Exception as exc:
        _emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
