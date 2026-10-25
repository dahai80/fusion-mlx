# SPDX-License-Identifier: Apache-2.0
# Regression tests for the CLI -> Config fidelity story (#400): a flag that
# argparse accepts but the SchedulerConfig construction site never forwards
# silently keeps the dataclass default.
#
# The original version of this file imported `vllm_mlx.cli` / `vllm_mlx.server`
# and a `scripts/audit_cli_config_fidelity.py` that no longer exist after the
# rapid-mlx migration. The invariant is still worth pinning, so the audit lives
# here as a small AST helper over the current module layout.

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SERVE_COMMAND_SOURCE = REPO_ROOT / "fusion_mlx" / "cli_serve" / "serve_command.py"
CONFIG_SOURCE = REPO_ROOT / "fusion_mlx" / "config.py"


def _scheduler_config_fields() -> set[str]:
    # Only attributes that name a real SchedulerConfig field can be
    # dropped by the construction site; everything else serve_command
    # reads from args goes to ServerConfig / load_model.
    tree = ast.parse(CONFIG_SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SchedulerConfig":
            return {
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            }
    raise AssertionError("SchedulerConfig not found in config.py")


def _function_scheduler_config_kwargs(source: str, func_name: str) -> set[str]:
    tree = (
        ast.parse(source)
        if isinstance(source, str)
        else ast.parse(Path(source).read_text())
    )
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name == func_name):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "SchedulerConfig"
            ):
                return {kw.arg for kw in call.keywords if kw.arg}
        raise AssertionError(f"{func_name} exists but never constructs SchedulerConfig")
    raise AssertionError(f"function {func_name} not found")


def _audit_dropped_kwargs(
    source: str, func_name: str, fields: set[str] | None = None
) -> list[str]:
    # An args.X read that names a SchedulerConfig field but is missing from
    # the SchedulerConfig(...) call leaves the dataclass default in place —
    # the #400 bug shape.
    if fields is None:
        fields = _scheduler_config_fields()
    tree = ast.parse(source)
    func = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == func_name
        ),
        None,
    )
    if func is None:
        return []
    read_attrs = {
        node.attr
        for node in ast.walk(func)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "args"
    }
    passed = set()
    for call in ast.walk(func):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "SchedulerConfig"
        ):
            passed = {kw.arg for kw in call.keywords if kw.arg}
    return sorted(attr for attr in read_attrs if attr in fields and attr not in passed)


def test_prefill_step_size_is_plumbed_in_serve_command():
    kwargs = _function_scheduler_config_kwargs(
        SERVE_COMMAND_SOURCE.read_text(), "serve_command"
    )
    assert "prefill_step_size" in kwargs, (
        "regression: SchedulerConfig in serve_command no longer receives "
        "prefill_step_size — see #400. Add "
        "`prefill_step_size=args.prefill_step_size` to the construction."
    )


def test_audit_detects_synthetic_drift():
    source = textwrap.dedent("""
        def serve_command(args):
            print(args.prefill_step_size)
            cfg = SchedulerConfig(max_num_seqs=args.max_num_seqs)
            return cfg
        """)
    issues = _audit_dropped_kwargs(source, "serve_command")
    assert "prefill_step_size" in issues, issues
    assert "max_num_seqs" not in issues, issues


def test_audit_no_false_positive_when_field_unused():
    source = textwrap.dedent("""
        def benchmark_command(args):
            cfg = SchedulerConfig(max_num_seqs=args.max_num_seqs)
            return cfg
        """)
    assert _audit_dropped_kwargs(source, "benchmark_command") == []


def test_audit_is_clean_on_the_real_serve_command():
    # The real construction site forwards every args attribute it reads, so
    # the detector must stay quiet (otherwise it is noise, not a gate).
    assert _audit_dropped_kwargs(SERVE_COMMAND_SOURCE.read_text(), "serve_command") == (
        []
    )
