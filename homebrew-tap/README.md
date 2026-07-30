# fusion-mlx Homebrew Tap

```bash
brew tap dahai80/fusion-mlx
brew install fusion-mlx
```

## Services

```bash
brew services start fusion-mlx
brew services stop fusion-mlx
brew services info fusion-mlx
```

## Updating SHA256 on Release

Run the helper script before tagging a new release:

```bash
bash homebrew-tap/update_checksums.sh <version>
```

This fetches real SHA256 values for all resources and updates the Formula.
