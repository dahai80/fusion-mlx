#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CI import guard for the fusion_mlx public API boundary (#615/#613).

Downstream projects (fusion-comfyui etc.) should import public symbols via
``from fusion_mlx.public_api import X`` (the stable layer whose ``__all__``
carries the stability promise). Reaching into internal submodules
(``from fusion_mlx.engine_core import ...``, ``from fusion_mlx.pool.engine_pool
import ...``) bypasses the stable layer and breaks silently when internals
refactor.

This guard scans Python source for ``from fusion_mlx.<mod> import <names>``
and ``import fusion_mlx.<mod>`` usages, then classifies each:

  - OK: the module path IS ``fusion_mlx.public_api`` or ``fusion_mlx`` top
    (the public entry points whose symbols carry the stability promise).
  - WHITELIST: an internal-module import listed in the whitelist file as a
    ``(module, symbol)`` pair (with ``*`` wildcarding the symbol). These are
    the existing downstream imports grandfathered through the migration
    window (see ``scripts/public_api_whitelist.txt``).
  - WARN (default): any OTHER internal-module import — a new boundary
    piercing. The module path is the boundary: even importing a symbol that
    happens to be in ``public_api.__all__`` via a deeper internal path is
    flagged, because the *path* (not the symbol) is what breaks on refactor.
    Reviewers should nuge these to ``from fusion_mlx.public_api import X``.

Usage:

    python scripts/check_public_api_boundary.py --root tests/
    python scripts/check_public_api_boundary.py --downstream /path/to/fusion-comfyui
    python scripts/check_public_api_boundary.py --root tests/ --fail-on-warn

Exit codes: 0 = clean (or warnings only without --fail-on-warn),
1 = violations exceeded the configured threshold or a fatal scan error.
"""

from __future__ import annotations

import argparse
import ast
import logging
import pathlib
import sys
from dataclasses import dataclass, field

logger = logging.getLogger("check_public_api_boundary")

PUBLIC_API_MODULE = "fusion_mlx.public_api"
PKG_ROOT = "fusion_mlx"


@dataclass
class Violation:
    file: pathlib.Path
    line: int
    module: str
    names: list[str]
    reason: str


@dataclass
class ScanConfig:
    whitelist: set[tuple[str, str]] = field(default_factory=set)
    fail_on_warn: bool = False


def _module_is_internal(module: str) -> bool:
    """True for ``fusion_mlx.<sub>`` paths that are NOT the public_api module
    or the bare package root. ``fusion_mlx.public_api`` and ``fusion_mlx``
    itself are the public entry points; everything else is internal."""
    if module == PKG_ROOT or module == PUBLIC_API_MODULE:
        return False
    return module.startswith(PKG_ROOT + ".")


def _load_whitelist(path: pathlib.Path | None) -> set[tuple[str, str]]:
    """Each non-empty, non-comment line: ``module:Symbol`` (``module`` is the
    full dotted ``fusion_mlx.<sub>`` path; ``Symbol`` may be ``*`` to whitelist
    every symbol imported from that module). Returns a set of (module, symbol)
    tuples. The file path of the importing source is deliberately NOT part of
    the key — downstream repos may reorganize files, so the contract is bound
    to the (module, symbol) pair, not the source location."""
    entries: set[tuple[str, str]] = set()
    if path is None or not path.exists():
        logger.debug("whitelist %s absent — no exceptions loaded", path)
        return entries
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) != 2:
            logger.warning("whitelist line malformed (want module:Symbol): %r", raw)
            continue
        entries.add((parts[0], parts[1]))
    return entries


def _is_whitelisted(cfg: ScanConfig, module: str, name: str) -> bool:
    return (module, name) in cfg.whitelist or (module, "*") in cfg.whitelist


def _classify_imports(tree: ast.AST, source_file: pathlib.Path, cfg: ScanConfig, rel_file: str) -> list[Violation]:
    """Walk the AST and yield a Violation per non-whitelisted internal-module
    import. The module path is the boundary: an import reaches an internal
    submodule even if the symbol it pulls is also in public_api.__all__ — the
    path is what breaks on refactor, so it is flagged regardless of symbol."""
    out: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if not _module_is_internal(mod):
                continue
            for alias in node.names:
                name = alias.name
                if _is_whitelisted(cfg, mod, name):
                    continue
                out.append(Violation(source_file, node.lineno, mod, [name],
                                     f"internal module import: from {mod} import {name}"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                if not _module_is_internal(mod):
                    continue
                if _is_whitelisted(cfg, mod, "*"):
                    continue
                out.append(Violation(source_file, node.lineno, mod, [],
                                     f"internal module import: import {mod}"))
    return out


def scan_tree(root: pathlib.Path, cfg: ScanConfig) -> list[Violation]:
    """Scan every ``.py`` under ``root`` (a dir or a single file) for
    internal-module imports."""
    violations: list[Violation] = []
    if not root.exists():
        logger.info("scan root %s does not exist — skipping", root)
        return violations
    if root.is_file():
        paths = [root] if root.suffix == ".py" else []
    else:
        paths = list(root.rglob("*.py"))
    for path in paths:
        if any(part in {"__pycache__", ".venv", "node_modules", ".git"} for part in path.parts):
            continue
        try:
            source = path.read_text()
            tree = ast.parse(source)
        except (OSError, SyntaxError) as exc:
            logger.warning("skip unparseable %s: %s", path, exc)
            continue
        rel = path.name if root.is_file() else str(path.relative_to(root))
        violations.extend(_classify_imports(tree, path, cfg, rel))
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=None,
                        help="tree to scan for fusion_mlx internal imports")
    parser.add_argument("--downstream", type=pathlib.Path, default=None,
                        help="downstream repo root (e.g. fusion-comfyui checkout)")
    parser.add_argument("--whitelist", type=pathlib.Path, default=None,
                        help="whitelist file (module:Symbol lines)")
    parser.add_argument("--fail-on-warn", action="store_true",
                        help="exit 1 on any non-whitelisted internal import")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")

    cfg = ScanConfig()
    cfg.whitelist = _load_whitelist(args.whitelist)
    cfg.fail_on_warn = args.fail_on_warn

    scan_roots: list[tuple[str, pathlib.Path]] = []
    if args.root is not None:
        scan_roots.append(("scan", args.root))
    if args.downstream is not None:
        scan_roots.append(("downstream", args.downstream))
    if not scan_roots:
        parser.error("specify at least one of --root or --downstream")

    total = 0
    for label, root in scan_roots:
        logger.info("scanning %s root: %s", label, root)
        violations = scan_tree(root, cfg)
        if not violations:
            logger.info("[%s] clean — no internal-module imports", label)
            continue
        for v in violations:
            names = ",".join(v.names) if v.names else "(module)"
            logger.warning("[%s] %s:%d %s (%s)", label, v.file, v.line, v.reason, names)
        total += len(violations)

    if total:
        logger.warning("total internal-module import warnings: %d", total)
        if cfg.fail_on_warn:
            logger.error("failing CI: --fail-on-warn set and %d warning(s) found", total)
            return 1
        logger.warning("not failing (no --fail-on-warn) — warnings only per #615 migration window")
        return 0
    logger.info("public API boundary clean across all scan roots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
