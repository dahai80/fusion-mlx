# UPSTREAM_REF - vendored mlx_vlm MiniMax M3 modules

SPDX-License-Identifier: Apache-2.0

This directory is a **vendored snapshot** of upstream `mlx_vlm` modules.
It is tracked here so the snapshot origin, local modifications, and re-sync
procedure are auditable - see fusion-mlx issue #67.

## Source
- Upstream repo: https://github.com/Blaizzy/mlx-vlm
- Current mlx-vlm pin (`pyproject.toml`): `f96138eef1f5ce7fb5d97f8dd41a664a195b5659`
- Reason this vendor exists: mlx-vlm commit `e390667` removed the out-of-tree
  MiniMax M3 model + tool parser that FusionMLX depends on. This vendor restores
  only the removed modules so FusionMLX keeps MiniMax M3 / M3-VL support on the
  newer mlx-vlm pin.

## What is vendored
Only the modules removed upstream - NOT a full `mlx_vlm` tree:
- `mlx_vlm/models/minimax_m3/`         - MiniMax M3 (text) model
- `mlx_vlm/models/minimax_m3_vl/`      - MiniMax M3-VL (multimodal) model + processor
- `mlx_vlm/tool_parsers/minimax_m3.py` - MiniMax M3 tool-call parser

13 `.py` files, ~4.4k lines total. Everything else in `mlx_vlm` is used from
the installed upstream package - this vendor only fills the gap left by
`e390667`.

## Source commit
Vendored from the mlx-vlm tree as of the commit immediately preceding
`e390667` (the last revision that still contained `models/minimax_m3/`).
To pin the exact SHA when re-syncing, diff these files against
`mlx-vlm@e390667^`.

## Local modifications
Targeted runtime patches live in the PARENT package
(`fusion_mlx/patches/mlx_vlm_minimax_m3_compat/__init__.py`), applied via
namespace `__path__` extension + `importlib.import_module` + attribute
monkey-patching of `mlx_vlm.utils` / `mlx_vlm.prompt_utils`. The vendored
files under this directory are intended to be byte-for-byte the upstream
source; any local edit MUST be recorded here.

- (none known at time of writing)

## Loading mechanism (NOT sys.path pollution)
`_install_vendor_namespace()` in the parent `__init__.py` appends the
vendored subdirectories to the installed `mlx_vlm`, `mlx_vlm.models`, and
`mlx_vlm.tool_parsers` packages' `__path__` (namespace package extension),
then imports the vendored modules via `importlib.import_module`. No global
`sys.path.insert` is performed - the vendor only augments the
already-imported `mlx_vlm` namespace.

## Sync / re-vendor procedure
1. Check upstream mlx-vlm: has MiniMax M3 been re-added? If yes, drop this
   vendor and the compat patch, use upstream directly.
2. If still removed: compare vendored files against
   `mlx-vlm@<pre-e390667>` for upstream bug/security fixes, port them here,
   and record each change in "Local modifications" above.
3. Bump the pin in `pyproject.toml` and re-run
   `tests/unit/test_mlx_vlm_minimax_m3_compat.py` + `tests/unit/test_minimax_*`
   to confirm the compat surface still holds.
