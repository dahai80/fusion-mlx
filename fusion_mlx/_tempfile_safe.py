# SPDX-License-Identifier: Apache-2.0
"""Tempfile safety utilities stub."""

import logging
import tempfile

logger = logging.getLogger(__name__)


import contextlib


def safe_tempdir(*args, **kwargs):
    return tempfile.TemporaryDirectory(*args, **kwargs)


@contextlib.contextmanager
def managed_tempfile_path(*args, **kwargs):
    with tempfile.TemporaryDirectory(*args, **kwargs) as td:
        yield td
