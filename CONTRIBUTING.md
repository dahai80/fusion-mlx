# Contributing to fusion-mlx

fusion-mlx is currently a single-maintainer project. Contributions of
any size are welcome — this is the most direct way to reduce the
project's bus-factor risk. This guide gets you from clone to first PR.

## What we need most

In rough priority order (see [ROADMAP.md](ROADMAP.md) for the full plan):

1. **Test-debt cleanup** — ~301 test files are quarantined in
   `tests/unit/debt_modules.txt` (`collect_ignore`). Rescuing them
   (fixing imports, marking integration tests, or deleting truly-dead
   ones) is high-leverage and low-risk.
2. **tool_calling parser coverage** — add parsers for Gemma4 / Hermes /
   Mistral / MiniMax / ui_tars tool-call formats. One model family per
   file under a `tool_parsers/` layout (see existing `tool_calling.py`).
3. **Benchmark data** — run `scripts/benchmark_*.py` on your model and
   contribute the JSON to `benchmarks/` so the public matrix grows.
4. **Docs & examples** — modality walkthroughs (video, STS, NER),
   migration guides (Ollama → fusion-mlx).

## Setup

Requires macOS / Apple Silicon (MLX-native; no Linux/CUDA target).

```bash
git clone git@github.com:dahai80/fusion-mlx.git
cd fusion-mlx
source .venv/bin/activate     # or: python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras (only install what you use):

```bash
pip install -e ".[audio]"     # STT/TTS/STS via mlx-audio
pip install -e ".[image]"     # Flux1/Flux2 image gen via mflux-fusion
pip install -e ".[video]"     # video backends (opencv/librosa/imageio)
pip install -e ".[mcp]"       # MCP server
```

## Running the server

```bash
./start.sh start     # starts fusion-mlx (default port, see start.sh)
./start.sh stop
./start.sh status
./start.sh log
```

API auth key and port live in the config; the server listens on
`127.0.0.1` by default. Real-model tests require the server running
(see Testing below).

## Testing

```bash
# full active suite (skips quarantined + real-model tests)
pytest tests/unit -q

# a single module
pytest tests/unit/test_<name>.py -q

# real-model tests (loads actual MLX weights — slow, needs models on disk)
FUSION_MLX_REAL_MODEL_TESTS=1 pytest tests/unit -q
```

CI runs on Python 3.11 / 3.12 / 3.13 (macOS-14). The active test count
is ~377 files (301 quarantined, tracked in
`tests/unit/debt_modules.txt`). **Do not** claim a higher count in
README/badges than what `pytest --collect-only -q | tail -1` reports.

### Rule for failing tests

If you encounter a failing test — even one unrelated to your change —
locate and fix it (or file an issue). Do not leave the suite redder
than you found it.

## Lint & format

CI runs `ruff` + `black`; these must pass before merge.

```bash
ruff check fusion_mlx/ tests/
black --check fusion_mlx/ tests/
# autofix:
ruff check --fix fusion_mlx/ tests/
black fusion_mlx/ tests/
```

Notes:
- `fusion_mlx/patches/` is excluded from lint (upstream-derived vendor
  code; linting creates merge churn).
- MLX-family packages (`mlx`, `mlx_lm`, `mlx_vlm`, `mlx_embeddings`) are
  pinned as known-third-party in `[tool.ruff.lint.isort]` so isort
  classifies them deterministically across environments.
- Indentation in generated code uses multiples of 4. No docstrings in
  new code.

## Commit & PR flow

```bash
git checkout -b <type>/<short-desc>      # feat/fix/docs/chore/refactor
# ...changes...
git add <files>
git commit -m "<type>(#issue): <subject>"
git push -u fusion-mlx <branch>          # NOT origin (that's the homebrew tap)
gh pr create --repo dahai80/fusion-mlx --title "<type>(#issue): <subject>" --body "..."
```

PR checklist:
- [ ] `ruff check` + `black --check` pass
- [ ] `pytest tests/unit -q` is no redder than before
- [ ] CHANGELOG.md entry if user-facing
- [ ] README/docs updated if behavior changed

For upstream-blocking issues (mlx-lm / mlx-vlm limitations), **do not
fabricate a path** — file an issue on the upstream, link it here, and
keep a `raise` that fails visibly with a clear message.

## Code style essentials

- **Fail visibly, not silently** — loud errors over silent fallbacks.
- **Surgical changes** — touch only what your change requires; don't
  reformat adjacent code.
- **Convention beats novelty** — match the surrounding file's patterns
  even if you prefer another.
- **Logging by default** — new code should log enough to locate problems.

## Releases

See [RELEASE.md](RELEASE.md). Maintainers only.
