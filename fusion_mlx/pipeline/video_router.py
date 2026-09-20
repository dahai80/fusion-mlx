# SPDX-License-Identifier: Apache-2.0
"""Unified video router (PRD v1 §7.1).

Smart routing layer: general-purpose UX/demo/high-res → LTX-2.3;
short-drama / character-stable / dialogue → MiniMax H3. Upper business
layer + fusion-autotest call one entry point, the router picks the
backend. Manual override supported.

This wraps the existing VideoGenEngine backend registry rather than
duplicating it — the actual generate path stays in
engines/video_backends/{ltx2_5,minimax_h3}.py. The router only decides
WHICH backend + applies the unified scheduler's degradation plan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fusion_mlx.scheduler.video_unified_scheduler import get_video_scheduler

logger = logging.getLogger(__name__)

_LTX_KEYWORDS = (
    "ui",
    "demo",
    "interface",
    "scroll",
    "page",
    "popup",
    "interaction",
    "ux",
    "animation",
    "high-res",
    "1080p",
    "通用",
    "演示",
    "界面",
    "滚动",
)
_H3_KEYWORDS = (
    "drama",
    "short drama",
    "短剧",
    "character",
    "dialogue",
    "dubbing",
    "嘴型",
    "配音",
    "人物",
    "剧情",
    "分镜",
    "story",
    "scene cut",
)


@dataclass
class VideoRouteResult:
    backend: str
    reason: str
    degraded: bool
    plan: dict


class VideoRouter:
    def route(
        self,
        prompt: str,
        *,
        scene: str | None = None,
        model_hint: str | None = None,
        audio: bool = False,
    ) -> str:
        if model_hint:
            hint = model_hint.lower()
            if "h3" in hint or "minimax" in hint:
                return "minimax_h3"
            if "ltx" in hint:
                return "ltx2_5"
        if scene == "drama" or scene == "短剧":
            return "minimax_h3"
        if scene == "general" or scene == "通用":
            return "ltx2_5"
        p = (prompt or "").lower()
        h3_score = sum(1 for k in _H3_KEYWORDS if k in p)
        ltx_score = sum(1 for k in _LTX_KEYWORDS if k in p)
        if h3_score > ltx_score:
            return "minimax_h3"
        if ltx_score > h3_score:
            return "ltx2_5"
        return "ltx2_5" if not audio else "minimax_h3"

    def dispatch(
        self, params: Any, *, model_hint: str | None = None
    ) -> VideoRouteResult:
        sched = get_video_scheduler()
        backend = self.route(
            getattr(params, "prompt", ""),
            scene=(
                getattr(params, "extra", {}).get("scene")
                if hasattr(params, "extra")
                else None
            ),
            model_hint=model_hint,
            audio=getattr(params, "audio", False),
        )
        before = sched.snapshot()
        params = sched.begin_task(backend, params)
        after = sched.snapshot()
        degraded = after["level"] != "OK"
        logger.info(
            "router -> %s (degraded=%s peak=%.1fGB)",
            backend,
            degraded,
            after["peak_gb"],
        )
        return VideoRouteResult(
            backend=backend,
            reason="keyword/hint routing",
            degraded=degraded,
            plan=after,
        )


_router: VideoRouter | None = None


def get_video_router() -> VideoRouter:
    global _router
    if _router is None:
        _router = VideoRouter()
    return _router
