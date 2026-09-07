# homebrew-core Inclusion (#806)

Status: **guide + prepared formula**. The actual submission is a PR to the
external `Homebrew/homebrew-core` repository — out of scope for this repo per
the project's "only modify this project's code" constraint. This file is the
maintainer's playbook for that submission: what to file, the acceptance
checklist, and a homebrew-core-ready formula template.

## Why not in homebrew-core yet

homebrew-core acceptance criteria (https://docs.brew.sh/Acceptable-Formulae):

1. **Stable versioning** — fusion-mlx uses dynamic versioning from git tags
   (`_version.py`). A homebrew-core formula needs a stable tarball URL with a
   pinned SHA256. The PyPI sdist (`fusion-mlx-<ver>.tar.gz`) is the right
   source — it has a fixed SHA per release, unlike a GitHub archive tag.
2. **No vendored prebuilt binaries that violate policy** — the tap formula
   pulls ARM64 wheels as `resource` blocks. homebrew-core prefers building
   from source where feasible; Python wheels from PyPI are acceptable for
   `resource` blocks but each must be auditable. The slim default install
   (#805) helps: fewer resources = smaller review surface.
3. **Deterministic build** — `virtualenv_install_with_resources` is
   deterministic given pinned resource SHAs. ✅
4. **Tests pass in CI** — `test do` block runs `fusion-mlx --version`. ✅
5. **Notarized/signed provenance** — PEP 740 attestation (#804) gives
   reviewers confidence a wheel came from this repo. Staged; lands with the
   next tagged release.

## Blockers (as of this writing)

- #805 (slim extras split): **done** — improves review odds.
- #804 (Sigstore attestation): **scaffold done**, needs a real tagged release
  to mint a verifiable attestation. File the core PR *after* the next release
  ships so the reviewer can verify provenance.
- A PyPI sdist must exist for the target version (published by `publish.yml`).

## homebrew-core-ready formula template

Drop into `Homebrew/homebrew-core` as `Formula/f/fusion-mlx.rb`. Differs from
the tap formula: PyPI sdist source (not GitHub archive), `livecheck` for
`brew bump-formula-pr` automation, no `service` block on first submission
(homebrew-core review prefers minimal formula; add `service` in a follow-up).

```ruby
class FusionMlx < Formula
  desc "Unified local model serving for Apple Silicon (OpenAI-compatible API)"
  homepage "https://github.com/dahai80/fusion-mlx"
  url "https://files.pythonhosted.org/packages/source/f/fusion-mlx/fusion-mlx-<VERSION>.tar.gz"
  sha256 "<SDIST_SHA256>"
  license "Apache-2.0"

  # Apple Silicon only — MLX has no x86_64 macOS wheel.
  depends_on arch: :arm64
  depends_on "python@3.12"

  # Minimal default install (slim, #805). Heavy modalities are pip extras the
  # user adds post-install: pip install "fusion-mlx[vlm,audio,video]".
  resource "mlx" do
    url "https://files.pythonhosted.org/packages/py3/m/mlx/mlx-<MLX_VER>-cp312-cp312-macosx_14_0_arm64.whl"
    sha256 "<MLX_WHEEL_SHA256>"
  end

  resource "mlx-lm" do
    url "https://files.pythonhosted.org/packages/source/m/mlx-lm/mlx-lm-<MLX_LM_VER>.tar.gz"
    sha256 "<MLX_LM_SHA256>"
  end

  # livecheck: version follows PyPI, not git tags — enables brew bump-formula-pr.
  livecheck do
    url "https://pypi.org/pypi/fusion-mlx/json"
    regex(/"version":\s*"v?(\d+(?:\.\d+)+)"/i)
  end

  def install
    virtualenv_install_with_resources
  end

  def caveats
    <<~EOS
      fusion-mlx serves on 127.0.0.1:11434 by default.
        brew services start fusion-mlx
      or:  fusion-mlx serve <model>
      Models cached in ~/.fusion-mlx/models/
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/fusion-mlx --version")
  end
end
```

## Submission steps (maintainer, external)

```bash
# 1. Fork Homebrew/homebrew-core on GitHub.
# 2. Clone your fork inside a brew tap checkout.
brew tap --force-auto-update homebrew/core  # or the new API-tap-less workflow
cd "$(brew --repository homebrew/core)"
git remote add fork git@github.com:<you>/homebrew-core.git

# 3. Fill the template: replace <VERSION>, <SDIST_SHA256>, resource SHAs.
#    Pull SHAs from the release's SHA256SUMS.txt (publish.yml artifact).
sha256sum fusion-mlx-<VERSION>.tar.gz   # -> <SDIST_SHA256>

# 4. Lint + audit locally.
brew audit --new-formula Formula/f/fusion-mlx.rb
brew install --build-from-source Formula/f/fusion-mlx.rb
brew test fusion-mlx

# 5. File the PR.
git checkout -b fusion-mlx-<VERSION>
git add Formula/f/fusion-mlx.rb
git commit -m "fusion-mlx: new formula at <VERSION>"
git push fork fusion-mlx-<VERSION>
gh pr create --repo Homebrew/homebrew-core --title "fusion-mlx: new formula" \
  --body "New formula. Source: PyPI sdist (SHA256-pinned). Provenance: PEP 740
  attestation (#804, keyless Sigstore). Apple Silicon only (MLX arm64 wheel)."
```

## Post-acceptance: version bumps

Once accepted, bump on each release:

```bash
brew bump-formula-pr --url=<NEW_SDIST_URL> --sha256=<NEW_SHA> fusion-mlx
```

`livecheck` (above) lets `brew livecheck fusion-mlx` detect new PyPI versions
so bump automation can target them.

## What lives in THIS repo vs homebrew-core

| Artifact | Where | Why |
|---|---|---|
| Tap formula (current) | `homebrew-tap/Formula/fusion-mlx.rb` | The existing `brew tap dahai80/fusion-mlx` install path — stays. |
| Core formula (template) | this file | Reference for the external submission; not installed from here. |
| `update_checksums.sh` | `homebrew-tap/` | CI stamps the tap formula's SHA per release. |
| Submission PR | `Homebrew/homebrew-core` | External — not this repo. |
