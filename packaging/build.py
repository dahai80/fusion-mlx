#!/usr/bin/env python3
# fusion-mlx packaging helper.
#
# Minimal build driver for the macOS app bundle. Supports three subcommands
# used by apps/fusion-mac/Scripts/build.sh:
#   --generate-venvstacks-toml <out>   emit resolved venvstacks.toml from pyproject
#   --write-engine-commits <pkg_dir>   write _engine_commits.json for runtime SHA display
#   --print-fingerprint                print a pyproject-derived fingerprint (freshness)
#
# venvstacks build / local-export are invoked directly (not via this script).
# Single source of truth for Python deps is pyproject.toml.

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="  %(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WHEELS_DIR = SCRIPT_DIR / "_wheels"

# pyproject sections merged into the framework-mlx-base layer, in order.
# Later sources override earlier on PEP 503 normalized name collisions.
LAYER_REQUIREMENTS_SOURCES = ["project", "grammar", "mcp"]

# Git deps shown in the admin dashboard "About" panel.
ENGINE_COMMIT_REPOS = {
    "mlx-lm": "https://github.com/ml-explore/mlx-lm",
    "mlx-vlm": "https://github.com/Blaizzy/mlx-vlm",
    "mlx-embeddings": "https://github.com/Blaizzy/mlx-embeddings",
    "dflash-mlx": "https://github.com/bstnxbt/dflash-mlx",
}

# Pin transitive deps to the versions validated in the dev .venv. Without an
# exclude-newer cutoff, venvstacks resolves latest-compatible, which drifted
# transformers to 5.13.0 — its AutoTokenizer.register() requires a config
# CLASS, but mlx_lm 0.31.3 passes the string "NewlineTokenizer", crashing at
# import. 5.0.0 is the tested-working version. Add more pins here as drift
# surfaces.
LAYER_PIN_OVERRIDES = {
    "transformers": "transformers==5.0.0",
}


def _read_pyproject():
    import tomllib
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)


def _read_layer_requirements():
    data = _read_pyproject()
    proj = data.get("project", {})
    out = {"project": list(proj.get("dependencies", []))}
    for name, reqs in proj.get("optional-dependencies", {}).items():
        out[name] = list(reqs)
    return out


def _normalize(req):
    key = req.split("@", 1)[0].split(";", 1)[0]
    key = re.split(r"[<>=!~]", key, maxsplit=1)[0].strip()
    return key.split("[", 1)[0].lower()


def _merge_requirements():
    sources = _read_layer_requirements()
    merged = {}
    for src in LAYER_REQUIREMENTS_SOURCES:
        if src not in sources:
            print(f"  ✗ pyproject section {src!r} not declared", file=sys.stderr)
            sys.exit(1)
        for req in sources[src]:
            merged[_normalize(req)] = req
    for name, pin in LAYER_PIN_OVERRIDES.items():
        merged[name] = pin
    return list(merged.values())


def _parse_git_requirements():
    data = _read_pyproject()
    proj = data.get("project", {})
    git_reqs = []
    for req in proj.get("dependencies", []):
        m = re.search(r"git\+https?://\S+@\S+", req)
        if m:
            git_reqs.append((req, m.group(0)))
    for reqs in proj.get("optional-dependencies", {}).values():
        for req in reqs:
            m = re.search(r"git\+https?://\S+@\S+", req)
            if m:
                git_reqs.append((req, m.group(0)))
    return git_reqs


def _find_wheel_python():
    # Pure-Python git deps build as py3-none-any wheels, but we still prefer a
    # 3.11 interpreter (matches the venvstacks target) and require pip.
    candidates = [os.environ.get("WHEEL_PYTHON"), shutil.which("python3.11"),
                  shutil.which("python3"), sys.executable]
    for p in candidates:
        if not p or not Path(p).exists():
            continue
        check = subprocess.run([p, "-c", "import pip"], capture_output=True)
        if check.returncode == 0:
            return p
    print("  ✗ no python with pip found (set WHEEL_PYTHON)", file=sys.stderr)
    sys.exit(1)


def _patch_wheel_metadata(whl_path, patches):
    # Rewrite a wheel's *.dist-info/METADATA in place, applying substring
    # replacements. Used to relax over-conservative pins on git deps whose
    # declared floor is higher than the version validated in the dev .venv.
    import zipfile
    tmp_path = whl_path.with_suffix(".whl.tmp")
    applied = False
    with zipfile.ZipFile(whl_path, "r") as zin:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename.endswith(".dist-info/METADATA"):
                    text = data.decode("utf-8")
                    for old, new in patches:
                        if old in text:
                            text = text.replace(old, new)
                            applied = True
                            print(f"    ~ {whl_path.name}: {old} -> {new}")
                    data = text.encode("utf-8")
                zout.writestr(item, data)
    if applied:
        whl_path.unlink()
        tmp_path.rename(whl_path)
    else:
        tmp_path.unlink()
    return applied


# Relax git-dep pins that are tighter than what the dev .venv validates.
# mlx-vlm 0.5.0 declares transformers>=5.5.0, but the dev .venv runs it on
# transformers 5.0.0 (5.5+ breaks mlx_lm 0.31.3's AutoTokenizer.register call).
# mlx-audio 0.4.3 (git) pins mlx-lm==0.31.1 exactly, but the dev .venv runs it
# on mlx-lm 0.31.3 (mlx-vlm needs >=0.31.3). Relax to a floor pin.
WHEEL_METADATA_PATCHES = {
    "mlx-vlm": [("transformers>=5.5.0", "transformers>=5.0.0")],
    "mlx-audio": [("mlx-lm==0.31.1", "mlx-lm>=0.31.1")],
}


def _reuse_existing_wheels():
    git_reqs = _parse_git_requirements()
    if not git_reqs:
        return {}
    git_pkg_names = {
        full_req.split("@")[0].strip().split("[")[0].lower().replace("-", "_")
        for full_req, _ in git_reqs
    }
    wheel_map = {}
    if WHEELS_DIR.exists():
        for whl in WHEELS_DIR.glob("*.whl"):
            name = whl.stem.split("-")[0].replace("_", "-").lower()
            if name.replace("-", "_") in git_pkg_names:
                wheel_map[name] = whl.resolve().as_uri()
                print(f"    + {name} -> {whl.name} (existing)")
    return wheel_map


def _build_local_wheels():
    # venvstacks locks with --require-hashes; git/VCS URLs cannot be hashed,
    # so pre-build wheels for git deps and reference them as file:// URLs.
    git_reqs = _parse_git_requirements()
    if not git_reqs:
        return {}
    existing = _reuse_existing_wheels()
    if existing and len(existing) >= len(git_reqs):
        print(f"  reusing {len(existing)} existing wheels (skip rebuild)")
        return existing
    if WHEELS_DIR.exists():
        shutil.rmtree(WHEELS_DIR)
    WHEELS_DIR.mkdir(parents=True)
    # Strip extras ([tts,stt,...]) and version markers so mlx-audio[tts,stt,sts]
    # normalizes to mlx_audio — matches the wheel-stem key built below.
    git_pkg_names = {
        full_req.split("@")[0].strip().split("[")[0].lower().replace("-", "_")
        for full_req, _ in git_reqs
    }
    wheel_python = _find_wheel_python()
    print(f"  wheel python: {wheel_python}")
    for full_req, git_url in git_reqs:
        pkg = full_req.split("@")[0].strip()
        print(f"  building wheel for {pkg} ...")
        r = subprocess.run(
            [wheel_python, "-m", "pip", "wheel", git_url,
             "--no-deps", "-w", str(WHEELS_DIR)]
        )
        if r.returncode != 0:
            print(f"  ✗ wheel build failed for {pkg}", file=sys.stderr)
            sys.exit(1)
    for whl in WHEELS_DIR.glob("*.whl"):
        name = whl.stem.split("-")[0].replace("_", "-").lower()
        if name in WHEEL_METADATA_PATCHES:
            print(f"  patching {name} wheel metadata...")
            _patch_wheel_metadata(whl, WHEEL_METADATA_PATCHES[name])
    wheel_map = {}
    for whl in WHEELS_DIR.glob("*.whl"):
        name = whl.stem.split("-")[0].replace("_", "-").lower()
        if name.replace("-", "_") in git_pkg_names:
            wheel_map[name] = whl.resolve().as_uri()
            print(f"    + {name} -> {whl.name}")
    return wheel_map


def prepare_venvstacks_toml(out_path):
    wheel_map = _build_local_wheels()
    reqs = _merge_requirements()
    lines = []
    emitted = set()
    for req in reqs:
        norm = _normalize(req)
        emitted.add(norm)
        if norm in wheel_map:
            lines.append(f'    "{norm} @ {wheel_map[norm]}",')
        else:
            lines.append(f'    "{req}",')
    # Emit git-built wheels not in the layer's direct requirements but needed
    # transitively with a looser pin than the PyPI release. mlx-audio is pulled
    # by mlx-vlm; PyPI 0.4.4 pins transformers>=5.5.0, but the git build (0.4.3)
    # allows >=5.0.0 — matching the dev .venv. Including it forces venvstacks to
    # use this wheel for the transitive resolution.
    for name, uri in wheel_map.items():
        if name not in emitted:
            # No inline comment: the requirements array is emitted on one line,
            # so a trailing '#' would eat the closing bracket.
            lines.append(f'    "{name} @ {uri}",')
            emitted.add(name)
            print(f"    + {name} (transitive, git wheel)")
    req_block = "".join(lines)
    toml = f'''# fusion-mlx venvstacks layer spec (generated from pyproject.toml).
# Git deps pre-built as local wheels (venvstacks requires hashable URLs).
# Source of truth for Python deps: pyproject.toml [project] + [grammar] + [mcp].

[[runtimes]]
name = "cpython-3.11"
python_implementation = "cpython@3.11.10"
requirements = []
platforms = [
    "macosx_arm64",
]

[[frameworks]]
name = "mlx-base"
runtime = "cpython-3.11"
requirements = [
{req_block}]
platforms = [
    "macosx_arm64",
]
# cv2 is pulled transitively; its bundled dylibs duplicate PIL's. Exclude
# cv2's copies so venvstacks picks a single provider (matches omlx layer).
dynlib_exclude = [
    "cv2/.dylibs/libavif*",
    "cv2/.dylibs/liblcms2*",
    "cv2/.dylibs/libpng16*",
    "cv2/.dylibs/libtiff*",
    "cv2/.dylibs/libfreetype*",
    "cv2/.dylibs/libharfbuzz*",
    "cv2/.dylibs/liblzma*",
    "cv2/.dylibs/libxcb*",
    "cv2/.dylibs/libXau*",
    # macOS debug-symbol bundles (.dSYM) sit next to the real loadable dylib
    # and shadow it; exclude files inside any .dSYM so the real dylib wins.
    "**/*.dSYM/**",
    # libomp (OpenMP runtime) is bundled by both sklearn/.dylibs and torch/lib.
    # Keep torch's copy (canonical ML provider), drop sklearn's bundled dup.
    "sklearn/.dylibs/libomp*",
]

[tool.uv]
environments = [
    "sys_platform == 'darwin' and platform_machine == 'arm64'",
]
'''
    out = Path(out_path)
    out.write_text(toml)
    print(f"  + wrote {out} ({len(reqs)} requirements, {len(wheel_map)} local wheels)")


def generate_venvstacks_toml(out_path):
    reqs = _merge_requirements()
    req_lines = "".join(f'    "{r}",\n' for r in reqs)
    toml = f'''# fusion-mlx venvstacks layer spec (generated from pyproject.toml).
# Source of truth for Python deps: pyproject.toml [project] + [grammar] + [mcp].

[[runtimes]]
name = "cpython-3.11"
python_implementation = "cpython@3.11.10"
requirements = []
platforms = [
    "macosx_arm64",
]

[[frameworks]]
name = "mlx-base"
runtime = "cpython-3.11"
requirements = [
{req_lines}]
platforms = [
    "macosx_arm64",
]

[tool.uv]
environments = [
    "sys_platform == 'darwin' and platform_machine == 'arm64'",
]
'''
    out = Path(out_path)
    out.write_text(toml)
    print(f"  + wrote {out} ({len(reqs)} requirements)")


def write_engine_commits(pkg_dir):
    commits = {}
    for full_req, git_url in _parse_git_requirements():
        pkg_name = full_req.split("@")[0].strip().lower().split("[", 1)[0]
        if "@" in git_url and pkg_name in ENGINE_COMMIT_REPOS:
            commit = git_url.rsplit("@", 1)[1]
            commits[pkg_name] = {
                "commit": commit,
                "url": ENGINE_COMMIT_REPOS[pkg_name],
            }
    if commits:
        commits_file = Path(pkg_dir) / "_engine_commits.json"
        commits_file.write_text(json.dumps(commits, indent=2) + "\n")
        print(f"  + wrote {commits_file} ({list(commits.keys())})")


def print_fingerprint():
    raw = json.dumps(_read_layer_requirements(), sort_keys=True)
    print(hashlib.sha256(raw.encode()).hexdigest())


def venvstacks_only():
    logging.info("Step 1/4: generating venvstacks.toml …")
    toml_path = SCRIPT_DIR / "venvstacks.toml"
    prepare_venvstacks_toml(str(toml_path))
    logging.info("Step 2/4: venvstacks build …")
    subprocess.check_call(
        ["venvstacks", "build", str(toml_path)],
        cwd=str(SCRIPT_DIR),
    )
    logging.info("Step 3/4: venvstacks local-export …")
    subprocess.check_call(
        ["venvstacks", "local-export", str(toml_path)],
        cwd=str(SCRIPT_DIR),
    )
    export_dir = SCRIPT_DIR / "_export"
    if not export_dir.exists():
        print(f"  ✗ export dir missing: {export_dir}", file=sys.stderr)
        sys.exit(1)
    logging.info("Step 4/4: writing fingerprint …")
    fp = _read_layer_requirements()
    raw = json.dumps(fp, sort_keys=True)
    (export_dir / ".fingerprint").write_text(hashlib.sha256(raw.encode()).hexdigest() + "\n")
    print(f"  ✓ venvstacks export ready at {export_dir}")


def main():
    args = sys.argv[1:]
    if not args:
        print("usage: build.py --prepare-venvstacks <out> | "
              "--generate-venvstacks-toml <out> | "
              "--venvstacks-only | "
              "--write-engine-commits <dir> | --print-fingerprint", file=sys.stderr)
        sys.exit(2)
    cmd = args[0]
    if cmd == "--prepare-venvstacks" and len(args) == 2:
        prepare_venvstacks_toml(args[1])
    elif cmd == "--generate-venvstacks-toml" and len(args) == 2:
        generate_venvstacks_toml(args[1])
    elif cmd == "--venvstacks-only":
        venvstacks_only()
    elif cmd == "--write-engine-commits" and len(args) == 2:
        write_engine_commits(args[1])
    elif cmd == "--print-fingerprint":
        print_fingerprint()
    else:
        print(f"unknown args: {args}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
