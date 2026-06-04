#!/usr/bin/env python3
"""Sync fusion-mlx changes to Rapid-MLX fork.

Cherry-picks relevant commits from fusion-mlx and applies them to the
Rapid-MLX fork. Covers speculative/, parsers/, router/ modules.

Usage:
    python downstream/sync-to-rapid.py --dry-run
    python downstream/sync-to-rapid.py --since "2 days ago"
    python downstream/sync-to-rapid.py --apply
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

RAPID_FORK_REMOTE = "rapid-fork"
RAPID_FORK_BRANCH = "fusion-sync"

# Maps fusion-mlx paths to Rapid-MLX paths
PATH_MAPPING = {
    "fusion_mlx/speculative/": "vllm_mlx/speculative/",
    "fusion_mlx/parsers/": "vllm_mlx/parsers/",
    "fusion_mlx/router/": "vllm_mlx/router/",
    "fusion_mlx/mllm_scheduler.py": "vllm_mlx/mllm_scheduler.py",
    "fusion_mlx/mllm_batch_generator.py": "vllm_mlx/mllm_batch_generator.py",
    "fusion_mlx/mllm_cache.py": "vllm_mlx/mllm_cache.py",
    "fusion_mlx/multimodal_processor.py": "vllm_mlx/multimodal_processor.py",
}

FUSION_ROOT = Path(__file__).resolve().parent.parent
RAPID_ROOT = FUSION_ROOT.parent / "Rapid-MLX"


def run(cmd: list[str], check: bool = True, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True, cwd=cwd)


def get_syncable_commits(since: str) -> list[dict]:
    """Get commits that touch syncable files."""
    syncable_paths = list(PATH_MAPPING.keys())
    cmd = [
        "git", "log", f"--since={since}", "--oneline",
        "--name-only",
        "--",
    ]
    result = run(cmd + syncable_paths, cwd=FUSION_ROOT)
    if not result.stdout:
        return []

    commits = []
    current_hash = None
    current_msg = None
    current_files = []

    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            if current_hash and current_files:
                commits.append({
                    "hash": current_hash,
                    "msg": current_msg,
                    "files": current_files,
                })
            current_hash = None
            current_msg = None
            current_files = []
        elif current_hash is None and " " in line:
            parts = line.split(" ", 1)
            current_hash = parts[0]
            current_msg = parts[1]
        elif current_hash is not None:
            current_files.append(line)

    if current_hash and current_files:
        commits.append({
            "hash": current_hash,
            "msg": current_msg,
            "files": current_files,
        })

    return commits


def map_file_to_rapid(fusion_file: str) -> str | None:
    """Map a fusion-mlx file path to its Rapid-MLX equivalent."""
    for fusion_prefix, rapid_prefix in PATH_MAPPING.items():
        if fusion_file.startswith(fusion_prefix):
            suffix = fusion_file[len(fusion_prefix):]
            return rapid_prefix + suffix
    return None


def apply_commit_to_rapid(commit_hash: str, dry_run: bool = False) -> bool:
    """Apply a single commit's changes to the Rapid-MLX fork."""
    # Get the diff for this commit
    result = run(["git", "show", commit_hash], cwd=FUSION_ROOT)
    if not result.stdout:
        return False

    # Find which files changed and map them
    changed_files = []
    for line in result.stdout.split("\n"):
        if line.startswith("diff --git "):
            parts = line.split(" ")
            if len(parts) >= 3:
                filepath = parts[2].replace("a/", "").replace("b/", "")
                rapid_path = map_file_to_rapid(filepath)
                if rapid_path:
                    changed_files.append((filepath, rapid_path))

    if not changed_files:
        return False

    if dry_run:
        print(f"  Would apply {commit_hash[:7]} to Rapid-MLX:")
        for fusion_f, rapid_f in changed_files:
            print(f"      {fusion_f} -> {rapid_f}")
        return True

    # Apply files one by one
    for fusion_file, rapid_file in changed_files:
        # Get the file content from this commit
        result = run(["git", "show", f"{commit_hash}:{fusion_file}"], cwd=FUSION_ROOT)
        if result.returncode != 0:
            continue

        # Write to Rapid-MLX
        target = RAPID_ROOT / rapid_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result.stdout)
        print(f"  Applied {fusion_file} -> {rapid_file}")

    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Sync fusion-mlx -> Rapid-MLX fork")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be synced")
    parser.add_argument("--since", default="1 week ago", help="Sync commits since this date")
    parser.add_argument("--apply", action="store_true", help="Actually apply changes (default: dry-run)")
    args = parser.parse_args()

    if not RAPID_ROOT.exists():
        print(f"Error: Rapid-MLX directory not found at {RAPID_ROOT}")
        sys.exit(1)

    commits = get_syncable_commits(args.since)
    if not commits:
        print("No syncable commits found.")
        return

    print(f"Found {len(commits)} commits with syncable changes:")
    for c in commits:
        print(f"    {c['hash'][:7]} {c['msg']} ({len(c['files'])} files)")

    if not args.apply:
        print("\nRun with --apply to actually sync changes.")
        return

    print("\nApplying to Rapid-MLX...")
    for commit in commits:
        print(f"\nProcessing {commit['hash'][:7]}: {commit['msg']}")
        apply_commit_to_rapid(commit["hash"], dry_run=False)

    print("\nSync complete. Review changes in Rapid-MLX directory.")


if __name__ == "__main__":
    main()
