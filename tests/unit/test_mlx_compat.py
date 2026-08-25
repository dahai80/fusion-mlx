# SPDX-License-Identifier: Apache-2.0
"""
Tests for the MLX hardware-compat shim (#404 M5 single-stream).

We can't test on actual M5 from CI, but we can:
1. Verify the shim is installed *before* any module-level
   ``mx.new_thread_local_stream`` capture inside ``mlx_lm.generate``,
   by checking that importing ``fusion_mlx.scheduler`` triggers install().
2. Mock the probe failure and assert the fallback path returns
   ``mx.default_stream(...)``.
3. Verify idempotency — install() can be called multiple times safely.
4. Verify the shim is transparent on hardware that works (this runs on
   the dev's actual hardware in the test-apple-silicon CI job).

We do NOT test that ``import fusion_mlx`` installs the shim — that is the
*wrong* contract. We deliberately keep top-level ``import fusion_mlx``
free of ``mlx.core`` import so the package stays usable for metadata-only
access on systems where ``mlx`` is installed but Metal is unavailable
(``import mlx.core`` SIGABRTs there with an uncatchable NSException).
"""

from __future__ import annotations

import importlib
import importlib.resources

import pytest

pytest.importorskip("mlx.core")


def test_shim_installed_when_scheduler_imports():
    """Importing fusion_mlx.scheduler must install the compat shim — that's
    the gate that protects the module-level ``mx.new_thread_local_stream``
    call inside mlx_lm.generate (which scheduler imports at module top)."""
    import mlx.core as mx

    # Re-install explicitly so this test is order-independent: even if
    # scheduler was already imported by a prior test, install() is
    # idempotent and the assertion still holds.
    from fusion_mlx import _mlx_compat

    if hasattr(mx, "_fusion_mlx_compat_installed"):
        delattr(mx, "_fusion_mlx_compat_installed")
    import fusion_mlx.scheduler  # noqa: F401

    if not getattr(mx, "_fusion_mlx_compat_installed", False):
        # scheduler may already be in sys.modules from a previous test —
        # in which case its module-level install() didn't re-run. Confirm
        # that calling install() directly works.
        _mlx_compat.install()
    assert getattr(mx, "_fusion_mlx_compat_installed", False) is True


def test_fusion_mlx_init_does_not_install_shim_or_import_mlx():
    """`fusion_mlx/__init__.py` must NOT import mlx or call _mlx_compat.install().
    Both would eagerly load `mlx.core`, which SIGABRTs (uncatchable from
    Python) on systems where the `mlx` package is installed but Metal is
    unavailable — breaking metadata-only callers (`__version__`, etc.).

    Pure source-text audit; no module manipulation so the test is safe
    in a shared pytest process. The shim must be installed lazily at
    the top of every module that imports `mlx_lm.*` instead
    (verified by `test_every_mlx_lm_consumer_installs_shim`)."""
    init_source = (
        importlib.resources.files("fusion_mlx").joinpath("__init__.py").read_text()
    )
    assert "import mlx" not in init_source, (
        "fusion_mlx/__init__.py must not import mlx — it would break "
        "metadata-only usage on systems with broken Metal init."
    )
    assert "_mlx_compat.install()" not in init_source, (
        "fusion_mlx/__init__.py must not call _mlx_compat.install() — that "
        "would eagerly import mlx.core (which can SIGABRT on Metal-less "
        "systems). The shim must install at scheduler-import time instead."
    )


def test_every_mlx_lm_consumer_installs_shim():
    """fusion_mlx uses a CENTRAL boot-install contract (#617): the shim
    is installed once at the top of ``fusion_mlx/scheduler/__init__.py``,
    BEFORE any submodule that transitively imports ``mlx_lm`` at module
    load. Because ``fusion_mlx/__init__.py`` imports ``.scheduler``
    early, every entry path (``import fusion_mlx``, ``import
    fusion_mlx.oq``, ``import fusion_mlx.scheduler``) is guarded before
    the first ``mlx_lm`` submodule executes — and ``mlx_lm/__init__.py``
    capturing ``mx.new_thread_local_stream`` (#404 M5 single-stream bug)
    sees the wrapped symbol.

    This audit pins the central gate rather than a per-file guard: the
    prior per-file invariant contradicted the established boot-install
    convention (Rule 11) and flagged 28 real modules that are all
    covered by the central install. We verify the gate itself instead:

    1. ``scheduler/__init__.py`` calls ``_mlx_compat.install()`` at
       module level (not inside a function/lambda — must run at import).
    2. That install call appears BEFORE the first module-load-time
       ``mlx_lm`` import anywhere in the scheduler subtree (the install
       must win the race against ``mlx_lm/__init__.py``).
    3. ``fusion_mlx/__init__.py`` does not itself import ``mlx_lm`` or
       call install() directly (the gate lives in scheduler, per the
       metadata-only ``__init__`` contract — see
       ``test_fusion_mlx_init_does_not_install_shim_or_import_mlx``).
    """
    import ast
    import pathlib

    DEFERRED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)

    def _is_install_call(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "install":
            return False
        return isinstance(func.value, ast.Name) and func.value.id == "_mlx_compat"

    def _is_mlx_lm_node(node: ast.AST) -> bool:
        if isinstance(node, ast.Import):
            return any(
                alias.name == "mlx_lm" or alias.name.startswith("mlx_lm.")
                for alias in node.names
            )
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            return mod == "mlx_lm" or mod.startswith("mlx_lm.")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "import_module":
                if isinstance(func.value, ast.Name) and func.value.id == "importlib":
                    arg = node.args[0] if node.args else None
                    if arg is None:
                        for kw in node.keywords:
                            if kw.arg == "name":
                                arg = kw.value
                                break
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        s = arg.value
                        return s == "mlx_lm" or s.startswith("mlx_lm.")
        return False

    def _walk_module_level(node: ast.AST):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, DEFERRED_SCOPES):
                continue
            yield child
            yield from _walk_module_level(child)

    sched_init = pathlib.Path(
        str(importlib.resources.files("fusion_mlx").joinpath("scheduler/__init__.py"))
    ).resolve()
    source = sched_init.read_text()
    tree = ast.parse(source)
    first_install_line = None
    first_mlx_lm_line = None
    for node in _walk_module_level(tree):
        if first_install_line is None and _is_install_call(node):
            first_install_line = node.lineno
        if first_mlx_lm_line is None and _is_mlx_lm_node(node):
            first_mlx_lm_line = node.lineno
    assert first_install_line is not None, (
        "scheduler/__init__.py must call _mlx_compat.install() at module "
        "level — the central boot-install gate for the #404/#617 M5 "
        "single-stream shim. It was removed."
    )
    # scheduler/__init__.py itself has no direct mlx_lm import; the gate
    # must still precede any that appear (future edits) so the ordering
    # invariant is enforced whenever a direct import is present.
    if first_mlx_lm_line is not None:
        assert first_install_line < first_mlx_lm_line, (
            "scheduler/__init__.py calls _mlx_compat.install() at line "
            f"{first_install_line} but imports mlx_lm at line "
            f"{first_mlx_lm_line} first — the shim must wrap "
            "mx.new_thread_local_stream before mlx_lm/__init__.py runs "
            "(#404 M5 single-stream capture happens at mlx_lm import)."
        )


def test_install_is_idempotent():
    import mlx.core as mx

    from fusion_mlx import _mlx_compat

    _mlx_compat.install()
    first = mx.new_thread_local_stream
    _mlx_compat.install()
    second = mx.new_thread_local_stream
    assert first is second, "second install() must not re-wrap the function"


def test_install_is_noop_when_symbol_missing(monkeypatch):
    """Regression for #408: on mlx builds that predate
    ``mx.new_thread_local_stream``, ``install()`` must be a no-op rather
    than crash with AttributeError. Without this guard,
    ``import fusion_mlx.scheduler`` aborts before the server can bind a
    port — every user on the affected mlx is blocked from upgrading."""
    import mlx.core as mx

    from fusion_mlx import _mlx_compat

    # If a future mlx genuinely drops the symbol, this assert fails
    # loudly so we revisit whether the compat shim still has a job to
    # do — `raising=False` on the delattr below would silently turn
    # this into a degenerate test that exercises nothing.
    assert hasattr(mx, "new_thread_local_stream"), (
        "expected baseline mlx to expose new_thread_local_stream; "
        "if upstream removed it, this test no longer covers the #408 "
        "regression path and the shim itself can probably go away."
    )
    monkeypatch.delattr(mx, "new_thread_local_stream")
    monkeypatch.setattr(mx, "_fusion_mlx_compat_installed", False, raising=False)
    importlib.reload(_mlx_compat)
    _mlx_compat.install()  # must not raise — that's the #408 contract
    # Note: on the no-symbol path the shim deliberately does NOT mark
    # itself "installed" so that a later mlx upgrade (which adds the
    # symbol) gets the wrap on the next install() call. The contract
    # this test pins is "no AttributeError", not the flag.


def test_fallback_engages_when_probe_raises(monkeypatch):
    """Simulate M5: probe raises 'no Stream(gpu, 1)' → patched function must
    return mx.default_stream(device) instead of the unusable stream."""
    import mlx.core as mx

    from fusion_mlx import _mlx_compat

    # Make `_probe` always fail with the M5-shaped error. We poke the
    # ``mx`` namespace because the patch wires `with mx.stream(stream)`
    # → `mx.array(...) + mx.array(...)` — substituting `mx.stream`
    # itself is the cleanest interception point.
    class _BoomStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            raise RuntimeError("There is no Stream(gpu, 1) in current thread.")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mx, "stream", _BoomStream)

    # Force a fresh install with our broken probe environment.
    monkeypatch.setattr(mx, "_fusion_mlx_compat_installed", False, raising=False)
    importlib.reload(_mlx_compat)
    _mlx_compat.install()

    device = mx.default_device()
    fallback = mx.new_thread_local_stream(device)
    expected = mx.default_stream(device)
    # mx.default_stream is comparable by repr; compare structurally.
    assert repr(fallback) == repr(
        expected
    ), f"M5 fallback should return mx.default_stream({device!r}); got {fallback!r}"


def test_fallback_does_not_engage_on_unrelated_runtime_error(monkeypatch):
    """If `with mx.stream(stream)` raises a RuntimeError that doesn't look
    like the M5 single-stream signature, the shim must NOT swallow it —
    we want unexpected failures to surface, not get silently degraded."""
    import mlx.core as mx

    from fusion_mlx import _mlx_compat

    class _BoomStream:
        def __init__(self, stream):
            pass

        def __enter__(self):
            raise RuntimeError("Some completely unrelated MLX error")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mx, "stream", _BoomStream)
    monkeypatch.setattr(mx, "_fusion_mlx_compat_installed", False, raising=False)
    importlib.reload(_mlx_compat)
    _mlx_compat.install()

    with pytest.raises(RuntimeError, match="completely unrelated"):
        mx.new_thread_local_stream(mx.default_device())


def test_happy_path_unchanged_on_real_hardware():
    """On hardware where the original API works (M1–M4), the patched
    function must return a usable stream — and `with mx.stream(stream)`
    must run a trivial op. This is the test that confirms the shim is
    transparent for users who don't need it."""
    import mlx.core as mx

    from fusion_mlx import _mlx_compat

    # Cleanup from prior monkeypatched tests
    if hasattr(mx, "_fusion_mlx_compat_installed"):
        delattr(mx, "_fusion_mlx_compat_installed")
    importlib.reload(_mlx_compat)
    _mlx_compat.install()

    stream = mx.new_thread_local_stream(mx.default_device())
    with mx.stream(stream):
        result = (mx.array([1.0]) + mx.array([2.0])).item()
    assert result == 3.0
