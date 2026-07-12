# SPDX-License-Identifier: Apache-2.0
# MSL -> metallib compiler helper.
# Compiles Metal Shading Language source files into precompiled .metallib bundles.

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_metal_tool(name: str) -> str | None:
    xcode_paths = [
        "/usr/bin",
        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin",
        "/Library/Developer/CommandLineTools/usr/bin",
    ]
    for p in xcode_paths:
        full = os.path.join(p, name)
        if os.path.exists(full):
            return full
    which = subprocess.run(["which", name], capture_output=True, text=True)
    if which.returncode == 0:
        return which.stdout.strip()
    return None


def is_metal_compiler_available() -> bool:
    if platform.processor() != "arm":
        return False
    return _find_metal_tool("metal") is not None


def compile_msl(
    source_path: str | Path,
    output_path: str | Path,
    *,
    optimize: bool = True,
    sdk: str | None = None,
) -> bool:
    source_path = Path(source_path)
    output_path = Path(output_path)

    if not source_path.exists():
        logger.error("MSL source not found: %s", source_path)
        return False

    if not is_metal_compiler_available():
        logger.error(
            "Metal compiler tools not found. Install Xcode Command Line Tools."
        )
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    air_path = output_path.with_suffix(".air")

    # Compile MSL -> AIR using xcrun -sdk macosx metal
    metal_cmd = [
        "xcrun",
        "-sdk",
        "macosx",
        "metal",
        "-c",
        str(source_path),
        "-o",
        str(air_path),
        "-std=macos-metal2.4",
        "-ffast-math",
    ]
    if optimize:
        metal_cmd.append("-O2")

    try:
        logger.info("Compiling MSL: %s", " ".join(metal_cmd))
        subprocess.run(metal_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        logger.error(
            "MSL compilation failed:\n%s", exc.stderr.decode() if exc.stderr else ""
        )
        return False

    # Link AIR -> metallib using xcrun -sdk macosx metallib
    metallib_cmd = [
        "xcrun",
        "-sdk",
        "macosx",
        "metallib",
        str(air_path),
        "-o",
        str(output_path),
    ]
    try:
        subprocess.run(metallib_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        logger.error(
            "metallib linking failed:\n%s", exc.stderr.decode() if exc.stderr else ""
        )
        return False
    finally:
        if air_path.exists():
            air_path.unlink()

    logger.info(
        "Compiled: %s -> %s (%d bytes)",
        source_path.name,
        output_path.name,
        output_path.stat().st_size,
    )
    return True


def compile_metallib(output_dir: str | Path | None = None) -> bool:
    if not is_metal_compiler_available():
        logger.error(
            "Metal compiler not available (not on Apple Silicon or Xcode not installed)"
        )
        return False

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "precompiled"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mfa_dir = Path(__file__).resolve().parent
    msl_sources = list(mfa_dir.glob("*.metal")) + list(mfa_dir.glob("csrc/*.metal"))
    if not msl_sources:
        logger.warning("No MSL sources found in %s", mfa_dir)
        try:
            import mlx_mfa

            mfa_pkg_dir = Path(mlx_mfa.__file__).resolve().parent
            msl_sources = list(mfa_pkg_dir.glob("csrc/*.metal"))
            logger.info("Found %d MSL sources in mlx_mfa", len(msl_sources))
        except (ImportError, Exception):
            logger.info(
                "No MSL sources available - install mlx_mfa for precompiled kernels"
            )
            return False

    success = True
    for src in msl_sources:
        out = output_dir / src.with_suffix(".metallib").name
        if not compile_msl(src, out):
            success = False

    return success


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = compile_metallib()
    sys.exit(0 if success else 1)
