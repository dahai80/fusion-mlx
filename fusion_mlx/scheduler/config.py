from dataclasses import dataclass, field
from typing import Any

# The canonical SchedulerConfig + SchedulingPolicy live in fusion_mlx.config
# (the merged fusion-mlx model_settings + Rapid-MLX runtime config). Re-exported here
# so legacy ``from fusion_mlx.scheduler.config import SchedulerConfig`` keeps
# resolving to the single merged class.
#
# Why: the rapid-MLX migration carved a minimal runtime SchedulerConfig (this
# file) out of the released v0.4.0 SchedulerConfig (fusion_mlx/config.py) and
# made ``from .scheduler import SchedulerConfig`` resolve to the minimal one.
# The single-model ``serve --model`` / ``bench`` CLI paths build the rich
# config (~40 kwargs) but imported the minimal one, so they raised TypeError
# on the first rich kwarg. Merging the runtime fields into the released rich
# class and re-exporting it here restores the v0.4.0 contract: one class that
# carries both the CLI knobs and the runtime field reads. See
# fusion_mlx/config.py:SchedulerConfig for the field set.
from fusion_mlx.config import SchedulerConfig, SchedulingPolicy  # noqa: F401


@dataclass
class SchedulerOutput:
    """Output from a scheduling step.

    Contains information about what was scheduled and results.
    """

    # Requests scheduled in this step
    scheduled_request_ids: list[str] = field(default_factory=list)
    # Total tokens scheduled
    num_scheduled_tokens: int = 0
    # Requests that finished in this step
    finished_request_ids: set[str] = field(default_factory=set)
    # Request outputs (tokens generated)
    outputs: list = field(default_factory=list)
    # Whether any work was done
    has_work: bool = False
    # Prefill eviction request for memory management (fusion-mlx compat)
    prefill_eviction_request: Any = None
