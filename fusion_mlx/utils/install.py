# SPDX-License-Identifier: Apache-2.0
import shutil
import sys


def get_cli_prefix() -> str:
    if shutil.which("fusion-mlx"):
        return "fusion-mlx"
    return f"{sys.executable} -m fusion_mlx"
