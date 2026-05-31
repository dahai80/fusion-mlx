#!/usr/bin/env python3
"""Sync fusion-mlx changes to Rapid-MLX fork.

Cherry-picks relevant commits from fusion-mlx and pushes to the
Rapid-MLX fork. Runs manually after significant changes to
speculative/, parsers/, or router modules.
"""

import argparse
import subprocess
import sys

RAPID_FORK_REMOTE = "rapid-fork"
RAPID_FORK_BRANCH = "main"

# Maps fusion-mlx paths to Rapid-MLX paths
PATH_MAPPING = {
     "fusion_mlx/speculative/": "vllm_mlx/speculative/",
     "fusion_mlx/parsers/": "vllm_mlx/tool_parser/",
     "fusion_mlx/router/": "vllm_mlx/",
}


def main():
    parser = argparse.ArgumentParser(description="Sync fusion-mlx -> Rapid-MLX fork")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be synced")
    parser.add_argument("--since", default="1 week ago", help="Sync commits since this date")
    args = parser.parse_args()

    print(f"Scanning commits since {args.since}...")
    print("TODO: implement commit discovery and cherry-pick logic")
    print(f"Target remote: {RAPID_FORK_REMOTE}/{RAPID_FORK_BRANCH}")


if __name__ == "__main__":
    main()
