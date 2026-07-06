# SPDX-License-Identifier: Apache-2.0
import bisect
import logging
import threading
from collections import deque

logger = logging.getLogger(__name__)

COST_EWMA_ALPHA = 0.3
COST_CLAMP_FRACTION = 0.25
ACCEPTANCE_EWMA_ALPHA = 0.1
ACCEPTANCE_MIN_SAMPLES = 10
DEPTH_PROBE_INTERVAL = 4
DEPTH_PROBE_INTERVAL_MAX = 512
DEFAULT_MAX_K = 3
COST_SEED_MIN_SAMPLES = 4
STARVATION_PROBE_INTERVAL = 4


class CostModel:
    __slots__ = ("_ewma", "_depths", "_visits")

    def __init__(self) -> None:
        self._ewma: dict[int, float] = {}
        self._depths: list[int] = []
        self._visits: dict[int, int] = {}

    def observe(self, drafts: int, wall_ms: float) -> None:
        if drafts < 0 or wall_ms <= 0.0:
            return
        prev = self._ewma.get(drafts)
        if prev is None:
            self._ewma[drafts] = wall_ms
            bisect.insort(self._depths, drafts)
        else:
            limit = COST_CLAMP_FRACTION * prev
            innovation = wall_ms - prev
            if innovation > limit:
                innovation = limit
            elif innovation < -limit:
                innovation = -limit
            self._ewma[drafts] = prev + COST_EWMA_ALPHA * innovation
        self._visits[drafts] = self._visits.get(drafts, 0) + 1

    def ready(self) -> bool:
        return len(self._ewma) >= 2

    def sampled(self, drafts: int) -> bool:
        return drafts in self._ewma

    def visits(self, drafts: int) -> int:
        return self._visits.get(drafts, 0)

    def cost(self, drafts: int) -> float:
        ds = self._depths
        if not ds:
            return 0.0
        if drafts <= ds[0]:
            return self._ewma[ds[0]]
        if drafts >= ds[-1]:
            return self._ewma[ds[-1]]
        for i in range(1, len(ds)):
            hi = ds[i]
            if drafts <= hi:
                lo = ds[i - 1]
                t = (drafts - lo) / (hi - lo)
                return self._ewma[lo] + t * (self._ewma[hi] - self._ewma[lo])
        return self._ewma[ds[-1]]

    def sample_string(self) -> str:
        return " ".join(f"{d}:{self._ewma[d]:.0f}ms" for d in self._depths)


class AcceptanceModel:
    __slots__ = ("_rate", "_seen")

    def __init__(self) -> None:
        self._rate: list[float] = [0.0]
        self._seen: list[int] = [0]

    def _grow(self, i: int) -> None:
        while len(self._seen) <= i:
            self._rate.append(0.0)
            self._seen.append(0)

    def observe(self, drafted: int, accepted: int) -> None:
        for i in range(1, drafted + 1):
            if accepted < i - 1:
                break
            self._grow(i)
            outcome = 1.0 if accepted >= i else 0.0
            if self._seen[i] == 0:
                self._rate[i] = outcome
            else:
                self._rate[i] += ACCEPTANCE_EWMA_ALPHA * (outcome - self._rate[i])
            self._seen[i] += 1

    def acceptance(self, i: int) -> float:
        if 1 <= i < len(self._seen) and self._seen[i] >= ACCEPTANCE_MIN_SAMPLES:
            return self._rate[i]
        for j in range(i - 1, 0, -1):
            if j < len(self._seen) and self._seen[j] >= ACCEPTANCE_MIN_SAMPLES:
                return self._rate[j]
        return 1.0

    def expected_committed(self, n: int) -> float:
        total = 1.0
        prod = 1.0
        for k in range(1, n + 1):
            prod *= self.acceptance(k)
            total += prod
        return total

    def frontier(self) -> int:
        f = 0
        for i in range(1, len(self._seen)):
            if self._seen[i] >= ACCEPTANCE_MIN_SAMPLES:
                f = i
            else:
                break
        return f


class DepthController:
    def __init__(self, max_k: int = DEFAULT_MAX_K) -> None:
        self.cost = CostModel()
        self.acc = AcceptanceModel()
        self._probed = False
        self.scheduled = 0
        self._probe_interval = DEPTH_PROBE_INTERVAL
        self._probe_since = 0
        self._last_selected = 0
        self.max_k = max(0, max_k)
        self._round_probe_counter = 0
        self._round_probe_interval = STARVATION_PROBE_INTERVAL
        self._round_probe_last_sel = 0
        self._recent_k_used: deque[int] = deque(maxlen=DEPTH_PROBE_INTERVAL_MAX)
        self.starvation_probe_count = 0
        self.park_count = 0
        self.round_count = 0
        self.k_histogram: dict[int, int] = {}

    def frontier(self) -> int:
        return self.acc.frontier()

    def pick_k(self) -> int:
        sel = self._selected()
        if sel != self._last_selected:
            self._probe_interval = DEPTH_PROBE_INTERVAL
            self._probe_since = 0
            self._last_selected = sel
        elif self._probed:
            self._probe_interval = min(
                self._probe_interval * 2, DEPTH_PROBE_INTERVAL_MAX
            )
        self._probed = False
        self._probe_since += 1
        depth: int
        outward_probe_fired = False
        if self._probe_since >= self._probe_interval:
            self._probe_since = 0
            probe = min(sel + 1, self.frontier() + 1, self.max_k)
            if probe != sel:
                self._probed = True
                depth = probe
                outward_probe_fired = True
        if not outward_probe_fired:
            seed = self._cost_seed_depth()
            if seed >= 0:
                depth = seed
            else:
                depth = sel
            depth = min(depth, self.max_k)
        depth = self._apply_starvation_probe(sel, depth)
        self.scheduled = depth
        return depth

    def _apply_starvation_probe(self, sel: int, depth: int) -> int:
        if sel != self._round_probe_last_sel:
            self._round_probe_interval = STARVATION_PROBE_INTERVAL
            self._round_probe_counter = 0
            self._round_probe_last_sel = sel
        self._round_probe_counter += 1
        if self._round_probe_counter < self._round_probe_interval:
            return depth
        self._round_probe_counter = 0
        limit = min(self.frontier() + 1, self.max_k)
        if limit <= 0:
            self._round_probe_interval = min(
                self._round_probe_interval * 2, DEPTH_PROBE_INTERVAL_MAX
            )
            return depth
        window_size = self._round_probe_interval
        if len(self._recent_k_used) > 0:
            if len(self._recent_k_used) <= window_size:
                window = list(self._recent_k_used)
            else:
                window = list(self._recent_k_used)[-window_size:]
        else:
            window = []
        counts: dict[int, int] = {n: 0 for n in range(0, limit + 1)}
        for k in window:
            if k in counts:
                counts[k] += 1
        probe_k = min(counts.keys(), key=lambda n: (counts[n], n))
        self._round_probe_interval = min(
            self._round_probe_interval * 2, DEPTH_PROBE_INTERVAL_MAX
        )
        if probe_k != depth:
            self.starvation_probe_count += 1
            return probe_k
        return depth

    def record(
        self,
        k_used: int,
        wall_ms: float,
        accepts: list[bool] | None = None,
    ) -> None:
        self.round_count += 1
        if k_used == 0:
            self.park_count += 1
        self.k_histogram[k_used] = self.k_histogram.get(k_used, 0) + 1
        self._recent_k_used.append(k_used)
        if wall_ms > 0.0:
            self.cost.observe(k_used, wall_ms)
        if accepts:
            accepted = 0
            for outcome in accepts:
                if outcome:
                    accepted += 1
                else:
                    break
            self.acc.observe(len(accepts), accepted)

    def _selected(self) -> int:
        if not self.cost.ready():
            return 0
        limit = min(self.frontier() + 1, self.max_k)
        best = 0
        best_ev = self.acc.expected_committed(0) / self.cost.cost(0)
        for n in range(1, limit + 1):
            cost_n = self.cost.cost(n)
            if cost_n <= 0.0:
                continue
            ev = self.acc.expected_committed(n) / cost_n
            if ev > best_ev:
                best = n
                best_ev = ev
        return best

    def _cost_seed_depth(self) -> int:
        limit = min(self.frontier() + 1, self.max_k)
        for n in range(0, limit + 1):
            if self.cost.visits(n) < COST_SEED_MIN_SAMPLES:
                return n
        return -1

    def diagnostics(self) -> str:
        return (
            f"K_scheduled={self.scheduled} frontier={self.frontier()} "
            f"max_k={self.max_k} rounds={self.round_count} "
            f"parks={self.park_count} "
            f"cost=[{self.cost.sample_string()}] "
            f"probe_interval={self._probe_interval} "
            f"starve_interval={self._round_probe_interval} "
            f"starve_probes={self.starvation_probe_count}"
        )


_controllers: dict[str, DepthController] = {}
_lock = threading.Lock()


def get_or_create_controller(
    model_id: str,
    max_k: int = DEFAULT_MAX_K,
) -> DepthController:
    with _lock:
        ctrl = _controllers.get(model_id)
        if ctrl is None:
            ctrl = DepthController(max_k=max_k)
            _controllers[model_id] = ctrl
            logger.info(
                "[MTP-controller] created DepthController for model_id=%r max_k=%d",
                model_id,
                max_k,
            )
        elif ctrl.max_k != max_k:
            logger.warning(
                "[MTP-controller] ignoring max_k=%d for existing controller "
                "model_id=%r (already max_k=%d)",
                max_k,
                model_id,
                ctrl.max_k,
            )
        return ctrl


def reset_controllers() -> None:
    with _lock:
        _controllers.clear()


def get_controller_snapshot() -> dict[str, str]:
    with _lock:
        return {k: v.diagnostics() for k, v in _controllers.items()}


def sum_across_controllers() -> tuple[int, int, dict[int, int]]:
    with _lock:
        round_total = 0
        park_total = 0
        hist: dict[int, int] = {}
        for ctrl in _controllers.values():
            round_total += ctrl.round_count
            park_total += ctrl.park_count
            for k, count in ctrl.k_histogram.items():
                hist[k] = hist.get(k, 0) + count
        return round_total, park_total, hist
