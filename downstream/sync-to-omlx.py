#!/usr/bin/env python3
"""Sync fusion-mlx changes to omlx fork.

Cherry-picks relevant commits from fusion-mlx and applies them to the
omlx fork. Covers cache/, engines/, scheduler/, pool/ modules.

Usage:
    python downstream/sync-to-omlx.py --dry-run
    python downstream/sync-to-omlx.py --since "2 days ago"
    python downstream/sync-to-omlx.py --apply
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

OMLX_FORK_REMOTE = "omlx-fork"
OMLX_FORK_BRANCH = "fusion-sync"

# Maps fusion-mlx paths to omlx paths
PATH_MAPPING = {
    "fusion_mlx/cache/": "omlx/cache/",
    "fusion_mlx/engines/": "omlx/engine/",
    "fusion_mlx/scheduler.py": "omlx/scheduler.py",
    "fusion_mlx/pool/engine_pool.py": "omlx/engine_pool.py",
    "fusion_mlx/pool/memory_enforcer.py": "omlx/process_memory_enforcer.py",
    "fusion_mlx/pool/model_discovery.py": "omlx/model_discovery.py",
    "fusion_mlx/prefill_progress.py": "omlx/prefill_progress.py",
    "fusion_mlx/prefill_transient_tracker.py": "omlx/prefill_transient_tracker.py",
    "fusion_mlx/output_collector.py": "omlx/output_collector.py",
    "fusion_mlx/request.py": "omlx/request.py",
    "fusion_mlx/engine_core.py": "omlx/engine_core.py",
    "fusion_mlx/utils/": "omlx/utils/",
}

FUSION_ROOT = Path(__file__).resolve().parent.parent
OMLX_ROOT = FUSION_ROOT.parent / "omlx"


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


def map_file_to_omlx(fusion_file: str) -> str | None:
    """Map a fusion-mlx file path to its omlx equivalent."""
    for fusion_prefix, omlx_prefix in PATH_MAPPING.items():
        if fusion_file.startswith(fusion_prefix):
            suffix = fusion_file[len(fusion_prefix):]
            return omlx_prefix + suffix
    return None


def apply_commit_to_omlx(commit_hash: str, dry_run: bool = False) -> bool:
    """Apply a single commit's changes to the omlx fork."""
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
                omlx_path = map_file_to_omlx(filepath)
                if omlx_path:
                    changed_files.append((filepath, omlx_path))

    if not changed_files:
        return False

    if dry_run:
        print(f"  Would apply {commit_hash[:7]} to omlx:")
        for fusion_f, omlx_f in changed_files:
            print(f"    {fusion_f} -> {omlx_f}")
        return True

    # Apply files one by one
    for fusion_file, omlx_file in changed_files:
        # Get the file content from this commit
        result = run(["git", "show", f"{commit_hash}:{fusion_file}"], cwd=FUSION_ROOT)
        if result.returncode != 0:
            continue

        # Write to omlx
        target = OMLX_ROOT / omlx_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result.stdout)
        print(f"  Applied {fusion_file} -> {omlx_file}")

    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Sync fusion-mlx -> omlx fork")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be synced")
    parser.add_argument("--since", default="1 week ago", help="Sync commits since this date")
    parser.add_argument("--apply", action="store_true", help="Actually apply changes (default: dry-run)")
    args = parser.parse_args()

    if not OMLX_ROOT.exists():
        print(f"Error: omlx directory not found at {OMLX_ROOT}")
        sys.exit(1)

    commits = get_syncable_commits(args.since)
    if not commits:
        print("No syncable commits found.")
        return

    print(f"Found {len(commits)} commits with syncable changes:")
    for c in commits:
        print(f"  {c['hash'][:7]} {c['msg']} ({len(c['files'])} files)")

    if not args.apply:
        print("\nRun with --apply to actually sync changes.")
        return

    print("\nApplying to omlx...")
    for commit in commits:
        print(f"\nProcessing {commit['hash'][:7]}: {commit['msg']}")
        apply_commit_to_omlx(commit["hash"], dry_run=False)

    print("\nSync complete. Review changes in omlx directory.")


if __name__ == "__main__":
    main()
