#!/usr/bin/env python3
"""Unified sync tool for fusion-mlx downstream forks."""

import argparse
import datetime
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

FUSION_ROOT = Path(__file__).resolve().parent.parent
SYNC_LOG = FUSION_ROOT / "downstream" / ".sync_history.json"

# ---------------------------------------------------------------------------
# Fork definitions: path mapping + target repo root
# ---------------------------------------------------------------------------

FORKS = {
    "fusion-mlx": {
        "root": FUSION_ROOT.parent / "fusion-mlx",
        "remote": "fusion-mlx-fork",
        "branch": "fusion-sync",
        "path_mapping": {
            "fusion_mlx/cache/": "fusion-mlx/cache/",
            "fusion_mlx/engines/": "fusion-mlx/engine/",
            "fusion_mlx/scheduler.py": "fusion-mlx/scheduler.py",
            "fusion_mlx/pool/engine_pool.py": "fusion-mlx/engine_pool.py",
            "fusion_mlx/pool/memory_enforcer.py": "fusion-mlx/process_memory_enforcer.py",
            "fusion_mlx/pool/model_discovery.py": "fusion-mlx/model_discovery.py",
            "fusion_mlx/prefill_progress.py": "fusion-mlx/prefill_progress.py",
            "fusion_mlx/prefill_transient_tracker.py": "fusion-mlx/prefill_transient_tracker.py",
            "fusion_mlx/output_collector.py": "fusion-mlx/output_collector.py",
            "fusion_mlx/request.py": "fusion-mlx/request.py",
            "fusion_mlx/engine_core.py": "fusion-mlx/engine_core.py",
            "fusion_mlx/utils/": "fusion-mlx/utils/",
        },
    },
    "rapid": {
        "root": FUSION_ROOT.parent / "Rapid-MLX",
        "remote": "rapid-fork",
        "branch": "fusion-sync",
        "path_mapping": {
            "fusion_mlx/speculative/": "vllm_mlx/speculative/",
            "fusion_mlx/parsers/": "vllm_mlx/parsers/",
            "fusion_mlx/router/": "vllm_mlx/router/",
            "fusion_mlx/mllm_scheduler.py": "vllm_mlx/mllm_scheduler.py",
            "fusion_mlx/mllm_batch_generator.py": "vllm_mlx/mllm_batch_generator.py",
            "fusion_mlx/mllm_cache.py": "vllm_mlx/mllm_cache.py",
            "fusion_mlx/multimodal_processor.py": "vllm_mlx/multimodal_processor.py",
        },
    },
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], check: bool = True, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True, cwd=cwd)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def load_sync_history() -> list[dict]:
    if SYNC_LOG.exists():
        return json.loads(SYNC_LOG.read_text())
    return []


def save_sync_history(entries: list[dict]) -> None:
    SYNC_LOG.write_text(json.dumps(entries, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Commit discovery
# ---------------------------------------------------------------------------

def get_syncable_commits(fork_name: str, since: str) -> list[dict]:
    fork = FORKS[fork_name]
    syncable_paths = list(fork["path_mapping"].keys())
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
    current_files: list[str] = []

    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            if current_hash and current_files:
                commits.append({"hash": current_hash, "msg": current_msg, "files": current_files})
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
        commits.append({"hash": current_hash, "msg": current_msg, "files": current_files})

    return commits


# ---------------------------------------------------------------------------
# File mapping
# ---------------------------------------------------------------------------

def map_file(fork_name: str, fusion_file: str) -> str | None:
    fork = FORKS[fork_name]
    for fusion_prefix, target_prefix in fork["path_mapping"].items():
        if fusion_file.startswith(fusion_prefix):
            suffix = fusion_file[len(fusion_prefix):]
            return target_prefix + suffix
    return None


# ---------------------------------------------------------------------------
# Status check
# ---------------------------------------------------------------------------

def check_sync_status(fork_name: str) -> None:
    fork = FORKS[fork_name]
    target_root = fork["root"]

    if not target_root.exists():
        print(f"  Target dir missing: {target_root}")
        return

    history = load_sync_history()
    last_sync = None
    for entry in reversed(history):
        if entry.get("fork") == fork_name:
            last_sync = entry.get("time", "unknown")
            break

    print(f"Fork: {fork_name}")
    print(f"  Target: {target_root}")
    print(f"  Last sync: {last_sync or 'never'}")

    # Compare file hashes for mapped files
    mismatches = 0
    synced = 0
    missing = 0
    for fusion_prefix, target_prefix in fork["path_mapping"].items():
        if fusion_prefix.endswith("/"):
            fusion_dir = FUSION_ROOT / fusion_prefix
            if not fusion_dir.exists():
                continue
            for fp in fusion_dir.rglob("*.py"):
                rel = fp.relative_to(FUSION_ROOT)
                target_rel = map_file(fork_name, str(rel))
                if not target_rel:
                    continue
                target_fp = target_root / target_rel
                if not target_fp.exists():
                    missing += 1
                    continue
                if file_hash(fp) != file_hash(target_fp):
                    mismatches += 1
                    print(f"  DRIFT  {rel} != {target_rel}")
                else:
                    synced += 1
        else:
            fusion_fp = FUSION_ROOT / fusion_prefix
            if not fusion_fp.exists():
                continue
            target_rel = map_file(fork_name, fusion_prefix)
            if not target_rel:
                continue
            target_fp = target_root / target_rel
            if not target_fp.exists():
                missing += 1
            elif file_hash(fusion_fp) != file_hash(target_fp):
                mismatches += 1
                print(f"  DRIFT  {fusion_prefix} != {target_rel}")
            else:
                synced += 1

    print(f"  Synced: {synced} | Drifted: {mismatches} | Missing: {missing}")


# ---------------------------------------------------------------------------
# Apply changes
# ---------------------------------------------------------------------------

def apply_commit(fork_name: str, commit_hash: str, dry_run: bool = False) -> bool:
    fork = FORKS[fork_name]
    target_root = fork["root"]

    result = run(["git", "show", commit_hash], cwd=FUSION_ROOT)
    if not result.stdout:
        return False

    changed_files: list[tuple[str, str]] = []
    for line in result.stdout.split("\n"):
        if line.startswith("diff --git "):
            parts = line.split(" ")
            if len(parts) >= 3:
                filepath = parts[2].replace("a/", "").replace("b/", "")
                target_path = map_file(fork_name, filepath)
                if target_path:
                    changed_files.append((filepath, target_path))

    if not changed_files:
        return False

    if dry_run:
        print(f"  Would apply {commit_hash[:7]} to {fork_name}:")
        for fusion_f, target_f in changed_files:
            print(f"    {fusion_f} -> {target_f}")
        return True

    conflicts = 0
    applied = 0
    for fusion_file, target_file in changed_files:
        result = run(["git", "show", f"{commit_hash}:{fusion_file}"], cwd=FUSION_ROOT)
        if result.returncode != 0:
            continue

        target = target_root / target_file
        # Conflict detection: warn if target differs from source
        if target.exists():
            target_hash = file_hash(target)
            # Get hash of what we're about to write
            import tempfile
            tmp = Path(tempfile.mkdtemp()) / "tmp_content.py"
            tmp.write_text(result.stdout)
            new_hash = file_hash(tmp)
            tmp.unlink()
            if target_hash != new_hash:
                print(f"  CONFLICT {fusion_file} -> {target_file} (content differs, overwriting)")
                conflicts += 1

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result.stdout)
        applied += 1

    return applied > 0


# ---------------------------------------------------------------------------
# Sync command
# ---------------------------------------------------------------------------

def do_sync(fork_name: str, since: str, dry_run: bool, apply: bool, commit: bool) -> None:
    fork = FORKS[fork_name]
    target_root = fork["root"]

    if not target_root.exists():
        print(f"Error: target directory not found at {target_root}")
        sys.exit(1)

    commits = get_syncable_commits(fork_name, since)
    if not commits:
        print("No syncable commits found.")
        return

    print(f"Found {len(commits)} commits with syncable changes for {fork_name}:")
    for c in commits:
        print(f"  {c['hash'][:7]} {c['msg']} ({len(c['files'])} files)")

    if not apply:
        print("\nRun with --apply to actually sync changes.")
        return

    print(f"\nApplying to {fork_name}...")
    total_applied = 0
    total_conflicts = 0
    sync_entries = []

    for commit in commits:
        ch = commit["hash"]
        print(f"\nProcessing {ch[:7]}: {commit['msg']}")
        if apply_commit(fork_name, ch, dry_run=False):
            total_applied += 1
            sync_entries.append({
                "fork": fork_name,
                "commit": ch[:7],
                "msg": commit["msg"],
                "time": datetime.datetime.now().isoformat(),
            })

    # Log to history
    if sync_entries:
        history = load_sync_history()
        history.extend(sync_entries)
        # Keep last 200 entries
        if len(history) > 200:
            history = history[-200:]
        save_sync_history(history)

    # Auto-commit in target repo
    if commit and total_applied > 0:
        try:
            run(["git", "add", "."], cwd=target_root)
            run(["git", "commit", "-m",
                 f"sync: apply {total_applied} commits from fusion-mlx"],
                cwd=target_root)
            print(f"\nCommitted changes in {fork_name}.")
        except subprocess.CalledProcessError:
            print(f"\nWarning: git commit failed in {fork_name}. Changes are on disk.")

    print(f"\nSync complete: {total_applied} commits applied to {fork_name}.")


# ---------------------------------------------------------------------------
# Dep pin check
# ---------------------------------------------------------------------------

def check_dep_pins() -> None:
    import tomllib
    content = tomllib.loads((FUSION_ROOT / "pyproject.toml").read_text())
    deps = content.get("project", {}).get("dependencies", [])
    opt_deps = content.get("project", {}).get("optional-dependencies", {})

    git_deps = []
    for dep in deps:
        if "git+" in dep:
            git_deps.append(("main", dep))
    for group, items in opt_deps.items():
        for dep in items:
            if "git+" in dep:
                git_deps.append((group, dep))

    print("Pinned git dependencies:")
    for group, dep in git_deps:
        # Extract repo name and commit
        parts = dep.split("@")
        name = parts[0].strip()
        commit = parts[-1].strip() if len(parts) > 1 else "unknown"
        print(f"  [{group}] {name}@{commit[:8]}")

    # Check override-dependencies
    overrides = content.get("tool", {}).get("uv", {}).get("override-dependencies", [])
    if overrides:
        print("\nUV override-dependencies:")
        for dep in overrides:
            print(f"  {dep}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Sync fusion-mlx changes to downstream forks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # sync subcommand
    sync_p = sub.add_parser("sync", help="Sync commits to a fork")
    sync_p.add_argument("fork", choices=list(FORKS.keys()), help="Target fork name")
    sync_p.add_argument("--dry-run", action="store_true", help="Show what would be synced")
    sync_p.add_argument("--since", default="1 week ago", help="Sync commits since this date")
    sync_p.add_argument("--apply", action="store_true", help="Actually apply changes")
    sync_p.add_argument("--commit", action="store_true", help="Auto-commit in target repo after sync")

    # status subcommand
    status_p = sub.add_parser("status", help="Check sync drift between fusion-mlx and forks")
    status_p.add_argument("fork", choices=list(FORKS.keys()), nargs="?", default=None,
                          help="Fork to check (omit to check all)")

    # pins subcommand
    sub.add_parser("pins", help="Show pinned git dependency versions")

    args = parser.parse_args()

    if args.command == "sync":
        do_sync(args.fork, args.since, args.dry_run, args.apply, args.commit)
    elif args.command == "status":
        targets = [args.fork] if args.fork else list(FORKS.keys())
        for fork_name in targets:
            check_sync_status(fork_name)
            if len(targets) > 1:
                print()
    elif args.command == "pins":
        check_dep_pins()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
