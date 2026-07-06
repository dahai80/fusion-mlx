# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Defect 1 indexer gate.

Migrated from Rapid-MLX. The ``deepseek_v32_indexer_gate`` patch module
has NOT been migrated to fusion-mlx as a standalone module. The indexer
gate logic is integrated differently in ``fusion_mlx.patches.glm_moe_dsa``
which handles indexer_types natively in the model layer rather than via
monkey-patching. All tests are skipped with a clear reason.
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)

_SKIP_REASON = (
    "deepseek_v32_indexer_gate has not been migrated as a standalone "
    "patch module to fusion_mlx.patches. The indexer_types handling is "
    "integrated into fusion_mlx.patches.glm_moe_dsa model code directly "
    "rather than via monkey-patching. Re-enable when a compatible gate "
    "module is added or when the glm_moe_dsa patches gain equivalent "
    "test coverage."
)


@pytest.mark.skip(reason=_SKIP_REASON)
def test_upstream_without_gate_fails_with_missing_indexer_keys():
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_gate_loads_mixed_full_shared_config():
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_gate_is_noop_when_indexer_types_absent():
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_gate_rejects_all_shared_indexer_types():
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_decode_step_after_prefill_keeps_in_call_reuse():
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_empty_local_layer_slice_delegates_to_upstream():
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_reuse_path_runs_for_run_of_consecutive_shared_layers():
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_pp_shard_with_oversized_num_layers_raises_clear_error():
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_pp_shard_starting_on_shared_layer_raises_clear_error():
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_uninstall_restores_originals_across_module_reload():
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_gate_rejects_shared_at_index_zero():
    pass


@pytest.mark.skip(reason=_SKIP_REASON)
def test_install_fires_on_real_serve_import_path():
    pass
