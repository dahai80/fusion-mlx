# SPDX-License-Identifier: Apache-2.0
"""Distributed pipeline-parallelism primitives (#621).

First-version layer-slicing / activation-transfer surface for
fusion-multi-node Pipeline Parallelism. A shard holds a loaded mlx_lm
LanguageModel and runs forward over a contiguous layer range, returning
the hidden states for the next shard to consume. Activation tensors are
serialized via mx.save (.npy) + base64 so every mlx dtype (incl. bfloat16)
round-trips bit-exactly without a numpy detour (numpy cannot host bf16).

Transport (FMP binary protocol, AES-GCM, compression) is the scheduler
layer's job — these endpoints only expose the forward step.
"""

from .shard import (
    ShardError,
    ShardManager,
    deserialize_activation,
    serialize_activation,
)

__all__ = [
    "ShardManager",
    "serialize_activation",
    "deserialize_activation",
    "ShardError",
]
