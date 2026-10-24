# SPDX-License-Identifier: Apache-2.0
# Video generation engine for fusion-mlx.
# Thin delegator over a registered VideoBackend (see video_backends/). The
# engine owns engine-level lifecycle (start/stop), activity tracking, and the
# public generate() surface; the backend owns the mlx-video / direct-MLX
# generate path and per-backend constraints. Phase 0 ships LTX2Backend only;
# adding backends does not require touching this file.

import logging
import time
from typing import Any

import mlx.core as mx

from ..engine_core import (
    get_video_gen_timeout,
    is_video_executor_poisoned,
    poison_executor,
)
from ._progress import StepCallback
from .base import BaseNonStreamingEngine
from .video_backends import VideoGenParams, resolve_backend

logger = logging.getLogger(__name__)


# #1335: video activation memory scales linearly with latent frames and
# resolution (see video_unified_scheduler's per-step attention note), not with
# a constant fraction of the weights. 64 B per pixel-frame is a conservative
# envelope for latents + DiT intermediates at the common 768x512 / 121-frame
# workload (~3GB); the cap keeps the total projection at or below the old
# weights * 1.4 constant for oversized workloads so safety never regresses.
_VIDEO_ACTIVATION_BYTES_PER_PIXEL_FRAME = 64
_VIDEO_ACTIVATION_FLOOR_BYTES = 512 * 1024 * 1024
_VIDEO_ACTIVATION_WEIGHT_FRACTION_CAP = 0.40


def _entry_estimated_bytes(entry) -> int:
    # On-disk/discovery estimate already held by the pool entry: knowable
    # without a load, a disk scan or a repo resolution/download.
    if entry is None:
        return 0
    for attr in ("last_observed_size", "estimated_size"):
        value = getattr(entry, attr, 0)
        if isinstance(value, int) and value > 0:
            return int(value)
    return 0


def _pool_entry_estimated_bytes(model_name: str) -> int:
    try:
        from ..server import _server_state

        pool = getattr(_server_state, "engine_pool", None)
        entry = pool.get_entry(model_name) if pool is not None else None
    except Exception:  # noqa: BLE001
        return 0
    return _entry_estimated_bytes(entry)


def _resolve_video_weight_bytes(model_name: str) -> int:
    # Prefer the pool's discovery-time estimate (#1335) — the pre-load
    # admission path runs before get_engine and must not resolve/download the
    # repo. Fall back to the on-disk scan for engine-layer callers without a
    # pool entry. Returns 0 when unresolvable so the gate degrades to a no-op.
    pooled = _pool_entry_estimated_bytes(model_name)
    if pooled > 0:
        return pooled
    from pathlib import Path

    try:
        # ltx2_5.utils.get_model_path resolves HF repo id → local snapshot
        # (works for any repo cached under ~/.fusion-mlx/models); other
        # backends that don't use this layout degrade to 0 (no-op).
        from ..pool.model_discovery import estimate_model_size
        from ..video.ltx2_5.utils import get_model_path

        return int(estimate_model_size(Path(get_model_path(model_name))))
    except Exception:  # noqa: BLE001
        return 0


def video_activation_budget_bytes(
    num_frames: int,
    width: int,
    height: int,
    n: int = 1,
    weight_bytes: int = 0,
) -> int:
    # Per-request transient footprint: latent frames x pixels. n generates n
    # videos in the same call, so it scales the same way (the cap bounds it).
    px = max(1, int(width)) * max(1, int(height))
    frames = max(1, int(num_frames)) * max(1, int(n))
    activation = px * frames * _VIDEO_ACTIVATION_BYTES_PER_PIXEL_FRAME
    weight_bytes = max(0, int(weight_bytes))
    if weight_bytes > 0:
        activation = min(
            activation, int(weight_bytes * _VIDEO_ACTIVATION_WEIGHT_FRACTION_CAP)
        )
    return max(_VIDEO_ACTIVATION_FLOOR_BYTES, int(activation))


def video_admission_budget_bytes(
    weight_bytes: int,
    num_frames: int,
    width: int,
    height: int,
    n: int = 1,
    include_weights: bool = True,
) -> int:
    # include_weights=True (pre-load projection): resident does not hold the
    # weights yet, so the budget is weights + activation. include_weights=False
    # (post-load / engine gate): the weights are already counted in resident,
    # so adding them again double-counts the model (#1335).
    weight_bytes = max(0, int(weight_bytes))
    if include_weights and weight_bytes <= 0:
        return 0
    activation = video_activation_budget_bytes(
        num_frames, width, height, n, weight_bytes
    )
    return weight_bytes + activation if include_weights else activation


def _fps_signal(state: dict) -> None:
    # #1197 stage 2: feed the pacing controller a diffusion FPS quality
    # ratio (current fps / learned solo baseline). Cheap: one monotonic
    # read per denoise step; all failures are swallowed (a signal glitch
    # must never break a render).
    try:
        import time as _t

        now = _t.monotonic()
        last = state.get("last")
        state["last"] = now
        if last is None or now <= last:
            return
        fps = 1.0 / (now - last)
        baseline = max(float(state.get("baseline") or 0.0), fps)
        state["baseline"] = baseline
        self_ref = state.get("owner")
        if self_ref is not None:
            self_ref._fps_baseline = baseline
        if baseline < 0.05:  # cold start: 20s/step floor before signaling
            return
        from ..pacing import get_pacing_gate

        get_pacing_gate().update_signal("video", fps / baseline)
    except Exception:
        pass


class VideoGenEngine(BaseNonStreamingEngine):
    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self._model_name = model_name
        backend_name = kwargs.pop("backend", None)
        self._backend = resolve_backend(model_name, explicit=backend_name, **kwargs)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def _loaded(self) -> bool:
        return self._backend._loaded

    async def start(self) -> None:
        logger.info("Starting VideoGen engine: %s", self._model_name)
        await self._backend.start(self._model_name)
        logger.info("VideoGen engine ready: %s", self._model_name)

    async def stop(self) -> None:
        await self._backend.stop()

    async def generate(
        self,
        prompt: str,
        num_frames: int = 97,
        width: int = 768,
        height: int = 512,
        fps: int = 24,
        seed: int | None = None,
        n: int = 1,
        on_step: StepCallback | None = None,
        output_format: str = "mp4",
        **kwargs,
    ) -> list[bytes] | list[Any]:
        # Return type: list[bytes] for MP4, list[Any] for raw numpy arrays.
        # Callers: openai_routes.py video endpoint, ComfyUI stage callbacks.
        if not self._loaded:
            raise RuntimeError("VideoGen engine not started.")
        # R-3 (#811): if a prior generation hung and poisoned the video
        # executor, fast-fail loudly instead of silently queuing behind the
        # dead worker. The operator must restart fusion-mlx to reclaim the
        # stuck thread's memory; a fresh worker has already been spawned but
        # it cannot cancel the hung job.
        if is_video_executor_poisoned():
            raise RuntimeError(
                "video subsystem unavailable: a prior generation hung and "
                "poisoned the worker thread (#811 R-3). Restart fusion-mlx "
                "to restore video generation. "
                "Set FUSION_VIDEO_GEN_TIMEOUT to adjust the hang deadline."
            )
        # #1197: video submission pacing gate — per-job at the engine
        # boundary (per-step pacing would need backend hooks; the job-level
        # token already stops a spike of concurrent diffusions from piling
        # onto the Metal queue). Throttled = explicit error to the route.
        from ..pacing import pace_acquire_async

        if not await pace_acquire_async("video", 0.5):
            raise RuntimeError(
                "video generation throttled by submission pacing gate (#1197)"
            )

        params = VideoGenParams(
            prompt=prompt,
            n=n,
            num_frames=num_frames,
            width=width,
            height=height,
            fps=fps,
            seed=seed,
            negative_prompt=kwargs.get("negative_prompt"),
            image=kwargs.get("image"),
            # MiniMax-H3 l2va/fl2va (issue #687)：last_frame_image=末帧 keyframe
            # conditioning。Backend reads params.last_frame_image (minimax_h3.py)
            # and forwards to generate_video。HTTP route sets gen_kwargs directly
            # so it worked；engine-layer callers (ComfyUI/SDK) had it silently
            # dropped here → 一直 first-frame-only。补齐转发，None 保持向后兼容。
            last_frame_image=kwargs.get("last_frame_image"),
            image_strength=kwargs.get("image_strength", 1.0),
            num_inference_steps=kwargs.get("num_inference_steps"),
            cfg_scale=kwargs.get("cfg_scale"),
            guide_scale=kwargs.get("guide_scale"),
            pipeline=kwargs.get("pipeline"),
            scheduler=kwargs.get("scheduler"),
            shift=kwargs.get("shift"),
            tiling=kwargs.get("tiling"),
            no_compile=kwargs.get("no_compile"),
            enhance_prompt=kwargs.get("enhance_prompt"),
            # #1212 Turbo/LoRA adapter path (MiniMax-H3 few-step sampling):
            # True=auto-detect bundled turbo_lora.safetensors, str=explicit
            # path, None=off. Forwarded to backend params (minimax_h3).
            lora=kwargs.get("lora"),
            on_step=on_step,
            session_id=kwargs.get("session_id"),
            output_format=output_format,
            # VACE / camera control kwargs — forward to backend params so the
            # Wan2 generate_video path receives them (control_video,
            # control_mask, reference_images, camera_conditions). Without this
            # the caller's VACE reference_images were silently dropped here.
            control_video=kwargs.get("control_video"),
            control_mask=kwargs.get("control_mask"),
            reference_images=kwargs.get("reference_images"),
            camera_conditions=kwargs.get("camera_conditions"),
            quantize=kwargs.get("quantize", "none"),
            # MiniMax-H3 native audio (issue #588)：API 路由把 audio 放入 gen_kwargs，
            # 但此处未转发到 VideoGenParams → backend 收到默认 False → 永远 video-only，
            # 音频被静默丢弃。补齐转发，False 保持向后兼容。
            audio=kwargs.get("audio", False),
            audio_frozen=kwargs.get("audio_frozen", False),
            # LTX-2.5 (#982): audio VAE weights path for A/V decode chain。
            audio_vae_weights=kwargs.get("audio_vae_weights"),
        )

        # #1187/#1335: unified heavy-admission gate BEFORE begin_task acquires
        # the video mutex — a reject must not hold the mutex. generate() only
        # runs on a started engine, so the weights are already counted in
        # resident; charging them again double-counted the model and made the
        # projection duration-independent (weights * 1.4). The pre-load
        # projection (weights + activation) is done by the HTTP route before
        # get_engine. Unknown weights (0) skip the gate.
        from ..admission import get_heavy_admission_gate

        _weights_1187 = _resolve_video_weight_bytes(self._model_name)
        if _weights_1187 > 0:
            _budget_1187 = video_admission_budget_bytes(
                _weights_1187,
                num_frames,
                width,
                height,
                n,
                include_weights=False,
            )
            _gate = get_heavy_admission_gate()
            _decision = _gate.admit_media("video", _budget_1187)
            if not _decision.admitted:
                from ..scheduler.video_unified_scheduler import (
                    VideoAdmissionRejected,
                )

                raise VideoAdmissionRejected(_decision.reason)
        else:
            logger.debug(
                "video admission: weight estimate unavailable; unified gate skipped"
            )

        t0 = time.monotonic()
        activity_id = self._begin_activity(
            "generating video",
            metadata={"prompt_len": len(prompt), "num_frames": num_frames, "n": n},
        )

        # PRD v1 §4.1: every video task flows through the unified scheduler
        # so the 3-level memory circuit breaker (90/95/98GB) + dual-model
        # mutex + NF4 dequant cache bracket the backend generate. The
        # scheduler may degrade params (reduce steps / drop res / disable
        # audio) before the backend sees them. begin_task acquires the
        # video mutex (blocks if another video model is resident).
        from fusion_mlx.scheduler.video_unified_scheduler import get_video_scheduler

        _sched = get_video_scheduler()
        _backend_name = getattr(self._backend, "name", "") or self._model_name
        params = _sched.begin_task(_backend_name, params)

        # OP-901 (defect 1) parity with image_gen: register a media
        # reservation for the admitted video job so the memory enforcer's
        # hard watermark is raised to the full ceiling while the job runs.
        # Video pipelines (LTX-2.5 A/V two-stage @ 1088p+) legitimately push
        # Metal usage past the 95% hard watermark in late denoise/VAE stages
        # (#976 follow-up); without the raise the enforcer aborts a healthy
        # job. Single-slot — video jobs are serialized by the scheduler.
        _enforcer = None
        try:
            from ..server import _server_state

            _enforcer = getattr(
                _server_state.engine_pool, "process_memory_enforcer", None
            )
            if _enforcer is not None:
                # Estimate = on-disk weight bytes + 40% headroom for
                # activations/cache (LTX-2.5 A/V late-stage peaks are mostly
                # Metal cache reuse, bounded by the 64GB metal cache cap).
                _weights = _resolve_video_weight_bytes(self._model_name)
                _enforcer.register_media_reservation(
                    int(_weights * 1.4) if _weights > 0 else 0
                )
        except Exception:  # noqa: BLE001
            logger.debug(
                "video admission: reservation unavailable; skipping", exc_info=True
            )
        _vid_watchdog = None
        try:
            # #1189: video executor heartbeat watchdog. on_hang →
            # poison_executor("video") (same as the #811 R-3 TimeoutError
            # path). Heartbeat advances via the wrapped on_step callback so
            # a steadily-progressing render never trips the window.
            from ..executor_watchdog import get_executor_watchdog

            _vid_watchdog = get_executor_watchdog(
                "video", on_hang=lambda _name: poison_executor("video")
            )
            # #1197 stage 2: FPS quality signal — the pacing controller
            # shrinks AGGRESSOR budgets when diffusion FPS drops vs the
            # engine's own learned solo baseline (running max FPS; only
            # reported once the baseline clears the cold-start floor).
            _fps_state = {
                "last": None,
                "baseline": getattr(self, "_fps_baseline", 0.0),
                "owner": self,
            }

            # Wrap ALWAYS, but keep the #171 contract: when the caller
            # supplied no callback we inject a heartbeat-only wrapper —
            # purely internal (advances the #1189 watchdog, emits nothing),
            # so no user-visible callback semantics are injected. Without
            # it, a long-but-progressing render on the plain HTTP route
            # (wan2/hunyuanvideo 1280x704 > 600s) was falsely poisoned at
            # the staleness window while still denoising (#1216 E2E hit).
            # Stage-2 FPS signaling still fires only for engine-layer
            # callers (ComfyUI/SDK paths, which pass on_step); the plain
            # HTTP route keeps the stage-1 static pacing gate only. See
            # docs/pacing-gate.md.
            if params.on_step is not None:
                _user_step_cb = params.on_step

                async def _wd_step_cb(step: int, total: int) -> None:
                    # Advance the watchdog heartbeat on every denoise step.
                    _vid_watchdog.heartbeat()
                    _fps_signal(_fps_state)
                    await _user_step_cb(step, total)

                params.on_step = _wd_step_cb
            else:

                async def _wd_hb_only(step: int, total: int) -> None:
                    _vid_watchdog.heartbeat()

                params.on_step = _wd_hb_only
            _vid_watchdog.begin_job("video_generate")
            try:
                result = await self._backend.generate(params)
            except TimeoutError:
                # R-3 (#811): the backend's per-generation deadline fired but
                # the worker thread keeps running the hung pipeline (Python
                # cannot cancel a running thread). Poison + replace the
                # executor so later requests fast-fail loudly instead of
                # silently queuing behind the dead worker forever.
                poison_executor("video")
                logger.error(
                    "VideoGen generation timed out after %.0fs and poisoned "
                    "the video worker (#811 R-3). Subsequent video requests "
                    "will be rejected until fusion-mlx restart.",
                    get_video_gen_timeout(),
                )
                raise RuntimeError(
                    "video generation exceeded the hang deadline "
                    f"({get_video_gen_timeout():.0f}s) and the worker thread "
                    "could not be cancelled (#811 R-3). The video subsystem "
                    "is now poisoned; restart fusion-mlx to recover."
                )
            elapsed = time.monotonic() - t0
            self._update_activity(activity_id, elapsed_seconds=elapsed)
            logger.info("VideoGen generated %d video(s) in %.2fs", len(result), elapsed)
            return result
        finally:
            # #1189: end the video watchdog job. On the TimeoutError path the
            # handler above already poisoned the executor (#811 R-3).
            try:
                if _vid_watchdog is not None:
                    _vid_watchdog.end_job()
            except Exception:  # noqa: BLE001
                logger.debug("video watchdog end_job failed", exc_info=True)
            if _enforcer is not None:
                try:
                    _enforcer.unregister_media_reservation()
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "video admission: reservation release failed", exc_info=True
                    )
            _sched.end_task(_backend_name)
            await self._finish_activity(activity_id)

    # ------------------------------------------------------------------
    # Pipeline stage API (issue #170). Thin delegation to the backend.
    # Backends override VideoBackend stage methods to expose individual
    # stages for Fusion-ComfyUI sequential offload. Video latents are 5D
    # (batch, c, num_frames, h, w); denoise carries num_frames. Backends
    # that have not implemented a stage raise NotImplementedError.
    # ------------------------------------------------------------------
    async def load_text_encoder(self) -> None:
        await self._backend.load_text_encoder()

    async def encode_text(self, prompt: str) -> dict:
        return await self._backend.encode_text(prompt)

    async def unload_text_encoder(self) -> None:
        await self._backend.unload_text_encoder()

    async def load_dit(self) -> None:
        await self._backend.load_dit()

    async def denoise(
        self,
        latent: mx.array,
        pos_embed: mx.array,
        neg_embed: mx.array | None,
        steps: int,
        cfg: float,
        seed: int,
        num_frames: int,
        control=None,
        inpaint_mask=None,
        init_latent=None,
    ) -> mx.array:
        return await self._backend.denoise(
            latent,
            pos_embed,
            neg_embed,
            steps,
            cfg,
            seed,
            num_frames,
            control,
            inpaint_mask=inpaint_mask,
            init_latent=init_latent,
        )

    async def unload_dit(self) -> None:
        await self._backend.unload_dit()

    async def load_vae_encoder(self) -> None:
        await self._backend.load_vae_encoder()

    async def encode_control(self, **kwargs):
        return await self._backend.encode_control(**kwargs)

    async def unload_vae_encoder(self) -> None:
        await self._backend.unload_vae_encoder()

    async def load_vae(self) -> None:
        await self._backend.load_vae()

    async def decode(self, latent: mx.array) -> mx.array:
        return await self._backend.decode(latent)

    async def encode(self, pixels: mx.array) -> mx.array:
        return await self._backend.encode(pixels)

    async def decode_tiled(self, latent: mx.array, tile_size: int = 256) -> mx.array:
        return await self._backend.decode_tiled(latent, tile_size)

    async def unload_vae(self) -> None:
        await self._backend.unload_vae()

    def get_stats(self) -> dict[str, Any]:
        return {"model_name": self._model_name, "loaded": self._loaded}

    def __repr__(self) -> str:
        status = "running" if self._loaded else "stopped"
        return f"<VideoGenEngine model={self._model_name} status={status}>"
