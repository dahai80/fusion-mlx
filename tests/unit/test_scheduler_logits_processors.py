import pytest

from fusion_mlx.scheduler.sched_batch import Request, RequestOutput


def test_request_creation():
    assert Request is not None


def test_request_output_creation():
    assert RequestOutput is not None
