#!/usr/bin/env python3
"""Sync fusion-mlx changes to omlx fork.

Cherry-picks relevant commits from fusion-mlx and pushes to the
omlx fork. Runs manually after significant changes to cache/,
engines/, or scheduler modules.
"""

import argparse
import subprocess
import sys
from pathlib import Path

OMLX_FORK_REMOTE = "omlx-fork"
OMLX_FORK_BRANCH = "main"

# Maps fusion-mlx paths to omlx paths
PATH_MAPPING = {
     "fusion_mlx/cache/": "omlx/cache/",
     "fusion_mlx/engines/": "omlx/engine/",
     "fusion_mlx/pool/": "omlx/",
}


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(description="Sync fusion-mlx -> omlx fork")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be synced")
    parser.add_argument("--since", default="1 week ago", help="Sync commits since this date")
    args = parser.parse_args()

    print(f"Scanning commits since {args.since}...")
    # This is a stub — full implementation uses git log + cherry-pick
    print("TODO: implement commit discovery and cherry-pick logic")
    print(f"Target remote: {OMLX_FORK_REMOTE}/{OMLX_FORK_BRANCH}")


if __name__ == "__main__":
    main()
