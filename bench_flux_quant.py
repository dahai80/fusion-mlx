import asyncio
import os
import time

os.environ.setdefault("METAL_DEVICE_WRAPPER_TYPE", "0")

from fusion_mlx.engines.image_gen import ImageGenEngine

MODEL = os.path.expanduser("~/.fusion-mlx/models/FLUX.2-klein-base-4B")


async def bench(quantize, label):
    print(f"=== {label} (quantize={quantize}) ===", flush=True)
    eng = ImageGenEngine(model_name=MODEL, quantize=quantize)
    await eng.start()
    t0 = time.time()
    imgs = await eng.generate(
        prompt="a cat sitting on a wooden table",
        width=1024,
        height=1024,
        steps=4,
        seed=42,
    )
    dt = time.time() - t0
    await eng.stop()
    print(f"{label} 4-step 1024x1024: {dt:.2f}s ({len(imgs)} img)", flush=True)
    return dt


async def main():
    dt_bf = await bench(None, "bf16")
    dt_q = await bench(8, "w8a16")
    speedup = (dt_bf - dt_q) / dt_bf * 100 if dt_bf > 0 else 0
    print(
        f"\nRESULT: bf16={dt_bf:.2f}s w8a16={dt_q:.2f}s delta={speedup:+.1f}%",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
