# SPDX-License-Identifier: Apache-2.0
"""Tests for fatal process-exit helpers."""

from unittest.mock import patch

from fusion_mlx.utils.fatal import FATAL_EXIT_CODE, fatal_exit


def test_fatal_exit_dumps_traceback_and_exits():
    with (
        patch("fusion_mlx.utils.fatal.faulthandler.dump_traceback") as dump_traceback,
        patch("fusion_mlx.utils.fatal.os._exit") as exit_process,
    ):
        fatal_exit("fatal test")

    dump_traceback.assert_called_once_with(all_threads=True)
    exit_process.assert_called_once_with(FATAL_EXIT_CODE)
