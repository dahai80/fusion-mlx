import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import mlx.core as mx

logger = logging.getLogger(__name__)

ENV_FLAG = "FUSION_SPECULATIVE_DENOISE"
ENV_K = "FUSION_SPEC_K"
ENV_EPSILON = "FUSION_SPEC_EPSILON"
ENV_DRAFT_BLOCKS = "FUSION_SPEC_DRAFT_BLOCKS"
ENV_DRAFT_STRATEGY = "FUSION_SPEC_DRAFT_STRATEGY"
ENV_EVAL_STEPS = "FUSION_SPEC_EVAL_STEPS"
ENV_ASYNC_FLAG = "FUSION_ASYNC_DENOISE"


def speculative_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "0").lower() in ("1", "true", "yes", "on")


def async_denoise_enabled() -> bool:
    # issue #180: Metal 异步派发 (双缓冲去噪). 默认关 - 生产同步路径字节不变.
    # 开启时每步 mx.async_eval 入队 (非阻塞) + 末尾 mx.synchronize 排空.
    return os.environ.get(ENV_ASYNC_FLAG, "0").lower() in ("1", "true", "yes", "on")


@dataclass
class SpeculativeConfig:
    K: int = 4
    epsilon: float = 0.1
    relative: bool = True
    eval_steps: bool = True
    draft_strategy: str = "extrapolation"

    @classmethod
    def from_env(cls) -> "SpeculativeConfig":
        def _env_int(name, default):
            try:
                return int(os.environ.get(name, str(default)))
            except ValueError:
                logger.warning(
                    "spec-denoise: invalid int for %s, fallback %d", name, default
                )
                return default

        def _env_float(name, default):
            try:
                return float(os.environ.get(name, str(default)))
            except ValueError:
                logger.warning(
                    "spec-denoise: invalid float for %s, fallback %f", name, default
                )
                return default

        def _env_bool(name, default):
            return os.environ.get(name, "1" if default else "0").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )

        strategy = os.environ.get(ENV_DRAFT_STRATEGY, "extrapolation").lower().strip()
        if strategy not in ("extrapolation", "layer_pruned"):
            logger.warning(
                "spec-denoise: unknown draft_strategy=%s, fallback extrapolation",
                strategy,
            )
            strategy = "extrapolation"

        return cls(
            K=max(1, _env_int(ENV_K, 4)),
            epsilon=_env_float(ENV_EPSILON, 0.1),
            relative=True,
            eval_steps=_env_bool(ENV_EVAL_STEPS, True),
            draft_strategy=strategy,
        )


VelocityFn = Callable[[mx.array, mx.array], mx.array]


class DraftDiTMixin:
    """Layer-pruned draft contract for a block-list DiT (wan2/SkyReels family).

    A DiT adopting this mixin implements forward_partial to run only the first
    n_blocks transformer blocks plus the shared output head, reusing the full
    model's patch embed / time embed / cross-KV. The draft is cheaper per
    forward (n_blocks / num_layers of the compute) and co-loads from the same
    weights - no separate draft checkpoint is needed.
    """

    def forward_partial(self, latent_input, t, n_blocks, **kwargs):
        raise NotImplementedError


class LayerPrunedDraft:
    """Adapts a forward_partial DiT into a batched VelocityFn draft."""

    def __init__(self, dit, n_blocks, **call_kwargs):
        self.dit = dit
        self.n_blocks = n_blocks
        self.call_kwargs = call_kwargs

    def __call__(self, x_batch: mx.array, t_batch: mx.array) -> mx.array:
        return self.dit.forward_partial(
            x_batch, t_batch, n_blocks=self.n_blocks, **self.call_kwargs
        )


class ExtrapolationDraft:
    """Zero-cost velocity draft using polynomial extrapolation from history.

    Instead of a separate model forward, predicts velocity from previously
    verified (full-model) velocities. This is an ODE predictor-corrector at
    the speculative level: linear or quadratic extrapolation predicts K steps,
    full model batch-verifies. Even modest acceptance rates yield speedup
    because draft cost is ~0 (no model forward).

    History is populated by the speculative loop after each full-verify pass:
    the verified velocities are fed back via record_verified().

    Modes:
      "linear"  — 1st-order: v_{i+1} = v_i  (constant velocity assumption)
      "quadratic" — 2nd-order: uses v_{i-1}, v_i to extrapolate via
                   linear velocity trend: v_{i+1} = 2*v_i - v_{i-1}
    """

    def __init__(self, mode: str = "linear"):
        self.mode = mode
        self._history: list[mx.array] = []
        self._t_history: list[float] = []

    def record_verified(self, v: mx.array, t: float) -> None:
        self._history.append(v)
        self._t_history.append(t)

    def __call__(self, x_batch: mx.array, t_batch: mx.array) -> mx.array:
        k = t_batch.shape[0]
        if not self._history:
            return mx.zeros_like(x_batch)

        last_v = self._history[-1]

        if self.mode == "quadratic" and len(self._history) >= 2:
            prev_v = self._history[-2]
            dv = last_v - prev_v
            batch = []
            for j in range(k):
                v_pred = last_v + dv * (j + 1)
                batch.append(v_pred)
            return mx.stack(batch, axis=0)
        else:
            batch = []
            for j in range(k):
                batch.append(last_v)
            return mx.stack(batch, axis=0)

    def reset(self) -> None:
        self._history.clear()
        self._t_history.clear()


def _velocity_error(
    v_draft: mx.array, v_full: mx.array, relative: bool = True
) -> float:
    diff = v_draft - v_full
    err = mx.sqrt(mx.sum(diff * diff))
    if not relative:
        return float(err)
    denom = mx.sqrt(mx.sum(v_full * v_full)) + 1e-8
    return float(err / denom)


def _euler_step(x: mx.array, v: mx.array, t_cur: float, t_next: float) -> mx.array:
    return x + (t_next - t_cur) * v


@dataclass
class SpecStats:
    macro_steps: int = 0
    accepted: list[int] = field(default_factory=list)
    full_forwards: int = 0
    draft_forwards: int = 0
    baseline_steps: int = 0
    draft_strategy: str = "extrapolation"

    @property
    def avg_accept(self) -> float:
        if not self.accepted:
            return 0.0
        return sum(self.accepted) / len(self.accepted)

    @property
    def speedup(self) -> float:
        if self.full_forwards == 0:
            return 0.0
        return self.baseline_steps / self.full_forwards

    def to_dict(self) -> dict:
        # issue #177 Phase-3: plain-dict serialization for the denoise-stats
        # stage surface (VideoBackend.last_denoise_stats / GET /denoise-stats).
        # Adds nothing to the denoise hot path; only called by the poll route.
        return {
            "macro_steps": self.macro_steps,
            "accepted": list(self.accepted),
            "avg_accept": self.avg_accept,
            "full_forwards": self.full_forwards,
            "draft_forwards": self.draft_forwards,
            "baseline_steps": self.baseline_steps,
            "speedup": self.speedup,
            "draft_strategy": self.draft_strategy,
        }


def speculative_denoise(
    full_velocity: VelocityFn,
    draft_velocity: VelocityFn | None,
    latents: mx.array,
    timesteps: mx.array,
    config: SpeculativeConfig | None = None,
) -> tuple:
    if config is None:
        config = SpeculativeConfig()

    # If caller provides draft_velocity but config says extrapolation,
    # auto-switch to layer_pruned (caller explicitly opted in to model-based draft).
    if draft_velocity is not None and config.draft_strategy == "extrapolation":
        config = SpeculativeConfig(
            K=config.K,
            epsilon=config.epsilon,
            relative=config.relative,
            eval_steps=config.eval_steps,
            draft_strategy="layer_pruned",
        )
        logger.info(
            "spec-denoise: draft_velocity provided, auto-switching strategy to layer_pruned"
        )

    timesteps = mx.array(timesteps)
    N = timesteps.shape[0]
    if N < 2:
        logger.warning("spec-denoise: need >=2 timesteps, got %d; returning input", N)
        return latents, SpecStats()

    K = config.K
    eps = config.epsilon
    use_extrap = config.draft_strategy == "extrapolation"
    stats = SpecStats(baseline_steps=N - 1, draft_strategy=config.draft_strategy)
    ts = [float(timesteps[k]) for k in range(N)]

    extrap_draft = None
    if use_extrap:
        _extrap_mode = os.environ.get("FUSION_SPEC_EXTRAP_MODE", "linear").lower()
        if _extrap_mode not in ("linear", "quadratic"):
            logger.warning("FUSION_SPEC_EXTRAP_MODE=%s unknown, using linear", _extrap_mode)
            _extrap_mode = "linear"
        extrap_draft = ExtrapolationDraft(mode=_extrap_mode)

    logger.info(
        "spec-denoise start: N=%d K=%d eps=%g relative=%s strategy=%s",
        N,
        K,
        eps,
        config.relative,
        config.draft_strategy,
    )

    i = 0
    x = latents
    while i < N - 1:
        K_eff = min(K, N - 1 - i)

        # First macro step with no history: run full model to bootstrap
        if use_extrap and not extrap_draft._history:
            v_f = full_velocity(mx.expand_dims(x, 0), mx.array([ts[i]]))[0]
            stats.full_forwards += 1
            if config.eval_steps:
                mx.eval(v_f)
            x = _euler_step(x, v_f, ts[i], ts[i + 1])
            if config.eval_steps:
                mx.eval(x)
            extrap_draft.record_verified(v_f, ts[i])
            stats.accepted.append(0)
            stats.macro_steps += 1
            i += 1
            logger.info(
                "spec-denoise macro %d: bootstrap (no history) -> step %d/%d",
                stats.macro_steps,
                i,
                N - 1,
            )
            continue

        xs = [x]
        vs_d = []
        cur = x
        for j in range(K_eff):
            t_j = ts[i + j]
            t_n = ts[i + j + 1]
            if use_extrap:
                v_d = extrap_draft(mx.expand_dims(cur, 0), mx.array([t_j]))[0]
            else:
                v_d = draft_velocity(mx.expand_dims(cur, 0), mx.array([t_j]))[0]
                stats.draft_forwards += 1
            if config.eval_steps:
                mx.eval(v_d)
            vs_d.append(v_d)
            cur = _euler_step(cur, v_d, t_j, t_n)
            if config.eval_steps:
                mx.eval(cur)
            if j < K_eff - 1:
                xs.append(cur)

        x_batch = mx.stack(xs, axis=0)
        t_batch = mx.array([ts[i + j] for j in range(K_eff)])
        vs_f = full_velocity(x_batch, t_batch)
        stats.full_forwards += 1
        mx.eval(vs_f)

        a = 0
        while a < K_eff:
            err = _velocity_error(vs_d[a], vs_f[a], relative=config.relative)
            if err < eps:
                a += 1
            else:
                logger.debug(
                    "spec-denoise macro %d: diverge at j=%d err=%g (eps=%g)",
                    stats.macro_steps + 1,
                    a,
                    err,
                    eps,
                )
                break

        # Feed verified velocities back to extrapolation draft
        if use_extrap:
            for j in range(a + 1):
                extrap_draft.record_verified(vs_f[j], ts[i + j])

        if a < K_eff:
            x = _euler_step(xs[a], vs_f[a], ts[i + a], ts[i + a + 1])
            i = i + a + 1
        else:
            x = cur
            i = i + K_eff

        if config.eval_steps:
            mx.eval(x)

        stats.accepted.append(a)
        stats.macro_steps += 1
        logger.info(
            "spec-denoise macro %d: accepted=%d/%d -> step %d/%d",
            stats.macro_steps,
            a,
            K_eff,
            i,
            N - 1,
        )

    logger.info(
        "spec-denoise done: macro=%d avg_accept=%.2f full_fwds=%d draft_fwds=%d speedup=%.2fx strategy=%s",
        stats.macro_steps,
        stats.avg_accept,
        stats.full_forwards,
        stats.draft_forwards,
        stats.speedup,
        config.draft_strategy,
    )
    return x, stats


def baseline_euler(
    full_velocity: VelocityFn,
    latents: mx.array,
    timesteps: mx.array,
) -> mx.array:
    timesteps = mx.array(timesteps)
    N = timesteps.shape[0]
    ts = [float(timesteps[k]) for k in range(N)]
    x = latents
    for i in range(N - 1):
        v = full_velocity(mx.expand_dims(x, 0), mx.array([ts[i]]))[0]
        x = _euler_step(x, v, ts[i], ts[i + 1])
        mx.eval(x)
    return x


def create_extrap_draft(mode: str = "linear") -> ExtrapolationDraft:
    return ExtrapolationDraft(mode=mode)
