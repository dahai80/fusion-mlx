# SPDX-License-Identifier: Apache-2.0
"""Request types for fusion-mlx. Stub module."""

from enum import Enum


class RequestStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    EVICTED = "evicted"


class Request:
    def __init__(self, request_id=""):
        self.request_id = request_id
        self.status = RequestStatus.WAITING


class SamplingParams:
    def __init__(self):
        self.temperature = 1.0
        self.top_p = 1.0
        self.max_tokens = 256
