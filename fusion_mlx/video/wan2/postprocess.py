from pathlib import Path

import numpy as np


def save_video(frames: np.ndarray, output_path: str, fps: int = 16):
    try:
        import imageio

        writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    except ImportError:
        try:
            import cv2

            h, w = frames.shape[1], frames.shape[2]
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        except (ImportError, Exception):
            # Last resort: save as individual PNGs
            from PIL import Image

            out_dir = Path(output_path).parent / Path(output_path).stem
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(out_dir / f"frame_{i:04d}.png")
            print(
                f"  (no video encoder available, saved {len(frames)} frames to {out_dir}/)"
            )
