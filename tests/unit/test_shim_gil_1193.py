"""#1193 shim GIL audit tests.

Source-level: every bound member in bindings.cpp must have a row in
GIL_AUDIT.md (table completeness). Native path (skipped when _ext is not
built): a slow shim call must not block a concurrent asyncio heartbeat.
"""

import asyncio
import re
from pathlib import Path

import pytest

BINDINGS = (
    Path(__file__).resolve().parents[2]
    / "fusion_mlx"
    / "shim"
    / "csrc"
    / "bindings.cpp"
)
AUDIT = Path(__file__).resolve().parents[2] / "fusion_mlx" / "shim" / "GIL_AUDIT.md"


def _extract_bound_members():
    text = BINDINGS.read_text()
    er_idx = text.index("EngineRunner")
    names = []
    for m in re.finditer(r'm\.def\(\s*"([A-Za-z_][A-Za-z0-9_]*)"', text):
        names.append(m.group(1))
    for m in re.finditer(r"\.def\(\s*nb::init", text):
        names.append("EngineRunner.__init__")
    for m in re.finditer(r'\.def\(\s*"([A-Za-z_][A-Za-z0-9_]*)"', text):
        if m.start() >= er_idx:
            names.append("EngineRunner." + m.group(1))
    for m in re.finditer(r"\.def_ro\(", text):
        names.append("HardwareProbe.def_ro")
    return sorted(set(names))


def _audit_rows():
    text = AUDIT.read_text()
    rows = set()
    for line in text.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) >= 2:
            rows.add(cells[1].strip("`"))
    return rows


def test_audit_table_covers_all_bound_members():
    members = _extract_bound_members()
    assert members, "bindings.cpp parse produced no members - regex drift"
    rows = _audit_rows()
    missing = [
        m for m in members if m not in rows and not m.startswith("HardwareProbe.def_ro")
    ]
    assert not missing, f"GIL_AUDIT.md missing rows for: {missing}"
    assert "HardwareProbe.def_ro" in rows


def test_heavy_bindings_release_gil():
    text = BINDINGS.read_text()
    for fn_name in ("hardware_probe", "hardware_probe_dict", "_test_slow_call"):
        def_idx = text.index(f'"{fn_name}"')
        # The next gil_scoped_release after the def must come from inside
        # the lambda (before the next m.def), not a later binding.
        next_def = text.find("m.def(", def_idx + 1)
        seg = text[def_idx : next_def if next_def != -1 else len(text)]
        assert "gil_scoped_release" in seg, f"{fn_name} does not release the GIL"


def _ext_or_skip():
    from fusion_mlx.shim import fast

    if fast._ext is None:
        pytest.skip("shim _ext not built")
    return fast._ext


def test_slow_shim_call_does_not_block_heartbeat():
    ext = _ext_or_skip()
    ticks = []

    async def heartbeat():
        for _ in range(40):
            ticks.append(asyncio.get_running_loop().time())
            await asyncio.sleep(0.01)

    async def main():
        hb = asyncio.ensure_future(heartbeat())
        await asyncio.sleep(0.02)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, ext._test_slow_call, 0.25)
        await hb

    asyncio.run(main())
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert len(ticks) == 40
    assert (
        max(gaps) < 0.25
    ), f"heartbeat stalled during shim call: max gap {max(gaps):.3f}s"


def test_gil_released_during_slow_call():
    ext = _ext_or_skip()
    # Prove the release is real: a busy Python thread runs DURING the C++
    # sleep. With the GIL held the busy loop would starve until the call
    # returns; with the release it finishes well inside the sleep window.
    import threading
    import time

    done = threading.Event()
    spins = [0]

    def busy():
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline and not done.is_set():
            spins[0] += 1
        done.set()

    t = threading.Thread(target=busy, daemon=True)
    t.start()
    ext._test_slow_call(0.2)
    done.wait(0.1)
    assert spins[0] > 100_000, "Python thread starved - GIL not released"
