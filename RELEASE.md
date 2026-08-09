# Release Process

This document fixes the fusion-mlx release flow so it is repeatable and
not tribal knowledge. Follow it end-to-end for every release.

## Prerequisites

- Push access to `dahai80/fusion-mlx` (main repo) and `dahai80/homebrew-fusion-mlx` (tap).
- Local venv: `cd fusion-mlx && source .venv/bin/activate`.
- A clean `main` (all intended changes merged).

## 1. Bump version

Edit `fusion_mlx/_version.py`:

```python
__version__ = "0.8.13"   # MAJOR.MINOR.PATCH, bump per semver
```

## 2. Update CHANGELOG.md

Add a new `## [0.8.13] - YYYY-MM-DD` section at the top. Summarize
merged PRs since the last release (one bullet per PR with `#NNN`).
Keep entries user-facing; link to the audit/issue context only when it
affects behavior.

## 3. Commit & PR

```bash
git checkout -b release/0.8.13
git add fusion_mlx/_version.py CHANGELOG.md
git commit -m "chore: bump version 0.8.12 -> 0.8.13"
# push to fusion-mlx remote (NOT origin, which is the homebrew tap)
git push -u fusion-mlx release/0.8.13
gh pr create --repo dahai80/fusion-mlx --title "release: v0.8.13" --body "..."
```

Merge the PR. **CI note**: the macOS-14 runner recurrently stalls on
the test matrix. If CI hangs, merge with `--squash --admin` (prior
releases all did this). Do not block a release on a stalled runner
once lint passes.

## 4. Tag & GitHub release

```bash
git checkout main && git pull fusion-mlx main
git tag v0.8.13
git push fusion-mlx v0.8.13
gh release create v0.8.13 --repo dahai80/fusion-mlx --title "v0.8.13" --notes-file <(gh release view v0.8.12 --repo dahai80/fusion-mlx --json body -q .body | head -1)
```

Publishing the GitHub release triggers `publish.yml`.

## 5. publish.yml (automatic)

`release: published` fires `publish.yml`, which:

1. `build` job (ubuntu): `uv build` + SHA256 checksums + upload artifacts.
2. `publish` job: uploads to **PyPI** via OIDC trusted publisher (no secret needed).
3. `update-homebrew` job: bumps the formula in `homebrew-fusion-mlx` (the `origin` remote) and opens/merges a PR to the tap.

Watch the run: https://github.com/dahai80/fusion-mlx/actions/workflows/publish.yml

### Known stalls

- The `update-homebrew` job runs on a macOS runner that recurrently
  queues. It auto-completes eventually; do not re-trigger manually.

## 6. Verify

- PyPI: https://pypi.org/project/fusion-mlx/ shows the new version.
- Homebrew: `brew install dahai80/fusion-mlx/fusion-mlx` installs it
  (or `brew upgrade`). Check the tap formula version matches.
- `pip install fusion-mlx==0.8.13` works clean.

## 7. Hotfix flow

If a shipped release has a critical bug:

1. Branch `release/0.8.14` from the `v0.8.13` tag (not main, if main
   has moved on).
2. Cherry-pick the fix.
3. Bump to the next patch, CHANGELOG entry, tag, release as above.

## Quick reference: remotes

| remote | repo | purpose |
|--------|------|---------|
| `fusion-mlx` | `git@github.com:dahai80/fusion-mlx.git` | main repo, PRs, tags, releases |
| `origin` | `git@github.com/dahai80/homebrew-fusion-mlx.git` | homebrew tap (auto-bumped by publish.yml) |
