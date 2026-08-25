# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/check_public_api_boundary.py (#615).

The guard is an AST-only source scanner — its classification functions are
pure (no mlx import needed), so these tests exercise them directly without
booting the engine. Only ``main()`` imports ``fusion_mlx.public_api``, which
we cover with a subprocess smoke test that's skipped when mlx is absent.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "check_public_api_boundary.py"
)


def _load_guard_module():
    spec = importlib.util.spec_from_file_location("check_public_api_boundary", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module: @dataclass resolves
    # ``cls.__module__`` -> sys.modules[name].__dict__ during class creation
    # (CPython dataclasses._is_type); an unregistered module makes that None
    # and crashes with AttributeError. (CPython 3.12 dataclasses.py:749)
    import sys

    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


guard = _load_guard_module()


def _classify(src: str, cfg):
    tree = ast.parse(src)
    fake = pathlib.Path("test_sample.py")
    return guard._classify_imports(tree, fake, cfg, "test_sample.py")


def test_bare_package_and_public_api_are_not_internal():
    assert guard._module_is_internal("fusion_mlx") is False
    assert guard._module_is_internal("fusion_mlx.public_api") is False


def test_submodule_paths_are_internal():
    assert guard._module_is_internal("fusion_mlx.engine_core") is True
    assert guard._module_is_internal("fusion_mlx.pool.engine_pool") is True
    assert guard._module_is_internal("fusion_mlx.video.pulid_mlx.pipeline") is True


def test_unrelated_packages_not_internal():
    assert guard._module_is_internal("os") is False
    assert guard._module_is_internal("mlx_lm") is False
    assert guard._module_is_internal("fusion_mlx_other") is False


def test_public_api_import_produces_no_violation():
    cfg = guard.ScanConfig()
    src = "from fusion_mlx.public_api import EnginePool, Server\n"
    assert _classify(src, cfg) == []


def test_bare_package_import_produces_no_violation():
    cfg = guard.ScanConfig()
    src = "from fusion_mlx import TTSEngine\nimport fusion_mlx\n"
    assert _classify(src, cfg) == []


def test_internal_module_import_flagged():
    cfg = guard.ScanConfig()
    src = "from fusion_mlx.engine_core import _something_internal\n"
    v = _classify(src, cfg)
    assert len(v) == 1
    assert v[0].module == "fusion_mlx.engine_core"
    assert v[0].names == ["_something_internal"]


def test_internal_path_flagged_even_if_symbol_in_public_all():
    # The module PATH is the boundary, not the symbol. Even pulling a symbol
    # that lives in public_api.__all__ via a deeper internal path is flagged —
    # the path breaks on refactor even if the symbol is stable.
    cfg = guard.ScanConfig()
    src = "from fusion_mlx.pool.engine_pool import EnginePool\n"
    v = _classify(src, cfg)
    assert len(v) == 1
    assert v[0].module == "fusion_mlx.pool.engine_pool"


def test_whitelist_exact_symbol_suppresses():
    cfg = guard.ScanConfig(whitelist={("fusion_mlx.pool.engine_pool", "EnginePool")})
    src = "from fusion_mlx.pool.engine_pool import EnginePool\n"
    assert _classify(src, cfg) == []


def test_whitelist_does_not_suppress_other_symbol_from_same_module():
    cfg = guard.ScanConfig(whitelist={("fusion_mlx.pool.engine_pool", "EnginePool")})
    src = "from fusion_mlx.pool.engine_pool import EnginePool, _internal_thing\n"
    v = _classify(src, cfg)
    assert len(v) == 1
    assert v[0].names == ["_internal_thing"]


def test_whitelist_wildcard_suppresses_all_symbols_from_module():
    cfg = guard.ScanConfig(whitelist={("fusion_mlx._torch_stub", "*")})
    src = "from fusion_mlx._torch_stub import install\nimport fusion_mlx._torch_stub\n"
    assert _classify(src, cfg) == []


def test_bare_import_statement_flagged():
    cfg = guard.ScanConfig()
    src = "import fusion_mlx.engine_core\n"
    v = _classify(src, cfg)
    assert len(v) == 1
    assert v[0].module == "fusion_mlx.engine_core"
    assert v[0].names == []


def test_load_whitelist_parses_module_symbol_lines(tmp_path):
    wl = tmp_path / "wl.txt"
    wl.write_text(
        "# comment\n\n"
        "fusion_mlx.pool.engine_pool:EnginePool\n"
        "fusion_mlx._torch_stub:*\n"
        "bad-line-with-no-colon\n"
        "fusion_mlx.config:get_config\n"
    )
    entries = guard._load_whitelist(wl)
    assert ("fusion_mlx.pool.engine_pool", "EnginePool") in entries
    assert ("fusion_mlx._torch_stub", "*") in entries
    assert ("fusion_mlx.config", "get_config") in entries
    assert len(entries) == 3  # malformed line skipped


def test_load_whitelist_absent_path_returns_empty():
    assert guard._load_whitelist(None) == set()
    assert guard._load_whitelist(pathlib.Path("/nonexistent/whitelist.txt")) == set()


def test_scan_tree_skips_venv_and_pycache(tmp_path):
    cfg = guard.ScanConfig()
    (tmp_path / "good.py").write_text("from fusion_mlx.engine_core import x\n")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "pkg.py").write_text("from fusion_mlx.engine_core import y\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "c.py").write_text(
        "from fusion_mlx.engine_core import z\n"
    )
    v = guard.scan_tree(tmp_path, cfg)
    modules = {x.module for x in v}
    assert modules == {"fusion_mlx.engine_core"}  # only good.py, .venv/pycache skipped


def test_comfyui_whitelist_file_present_and_nonempty():
    wl = _SCRIPT.parent / "public_api_whitelist.txt"
    assert wl.exists(), "whitelist file must ship alongside the guard"
    entries = guard._load_whitelist(wl)
    # The 13 grandfathered downstream (module, symbol) pairs.
    assert ("fusion_mlx.pool.engine_pool", "EnginePool") in entries
    assert ("fusion_mlx.video.pulid_mlx.pipeline", "PuLIDPipeline") in entries
    assert ("fusion_mlx._torch_stub", "install") in entries
    assert len(entries) >= 14


def test_main_no_roots_errors():
    with pytest.raises(SystemExit) as exc:
        guard.main([])
    assert exc.value.code != 0


@pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="main() imports fusion_mlx.public_api which needs mlx",
)
def test_main_clean_exit_on_clean_tree(tmp_path):
    (tmp_path / "clean.py").write_text("from fusion_mlx.public_api import Server\n")
    rc = guard.main(["--root", str(tmp_path)])
    assert rc == 0
