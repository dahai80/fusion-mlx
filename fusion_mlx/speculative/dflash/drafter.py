# SPDX-License-Identifier: Apache-2.0
import logging
from collections.abc import Generator
from typing import Any, Protocol

import mlx.core as mx

logger = logging.getLogger(__name__)


class BlockDiffusionDrafter(Protocol):
    block_size: int

    def draft_block(
        self,
        prefix_tokens: mx.array,
        current_position: int,
    ) -> mx.array: ...

    def reset(self) -> None: ...


class StubBlockDiffusionDrafter:
    def __init__(
        self,
        scripted_blocks: list[list[int]],
        block_size: int = 16,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be >= 1; got {block_size}")
        for i, block in enumerate(scripted_blocks):
            if len(block) != block_size:
                raise ValueError(
                    f"scripted_blocks[{i}] has length {len(block)}; "
                    f"expected block_size={block_size}"
                )
        self.block_size = block_size
        self._script: list[list[int]] = [list(b) for b in scripted_blocks]
        self._cursor = 0
        self._draft_calls = 0
        self._reset_calls = 0

    def draft_block(
        self,
        prefix_tokens: mx.array,
        current_position: int,
    ) -> mx.array:
        if self._cursor >= len(self._script):
            raise IndexError(
                f"StubBlockDiffusionDrafter script exhausted after "
                f"{self._cursor} blocks"
            )
        block = self._script[self._cursor]
        self._cursor += 1
        self._draft_calls += 1
        _ = prefix_tokens
        _ = current_position
        return mx.array(block, dtype=mx.uint32)

    def reset(self) -> None:
        self._cursor = 0
        self._draft_calls = 0
        self._reset_calls += 1


_RUNTIME_REQUIRED_ATTRS: tuple[str, ...] = (
    "drafter",
    "kind",
    "reset_accept_lens",
)


def _validate_runtime_contract(runtime: Any) -> None:
    missing = [attr for attr in _RUNTIME_REQUIRED_ATTRS if not hasattr(runtime, attr)]
    if missing:
        raise TypeError(
            f"runtime is missing required attribute(s) {missing}. "
            "Expected a DFlashRuntime instance or compatible shim."
        )


class MlxVlmDFlashDriver:
    def __init__(
        self,
        target_repo: str,
        drafter_repo: str,
        *,
        block_size: int | None = None,
    ) -> None:
        if not target_repo:
            raise ValueError("target_repo must be a non-empty string")
        if not drafter_repo:
            raise ValueError("drafter_repo must be a non-empty string")
        if block_size is not None and block_size <= 0:
            raise ValueError(f"block_size must be >= 1 or None; got {block_size}")
        self.target_repo = target_repo
        self.drafter_repo = drafter_repo
        self.block_size = block_size
        self._target: Any | None = None
        self._processor: Any | None = None
        self._runtime: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._target is not None and self._runtime is not None

    @property
    def target(self) -> Any:
        if self._target is None:
            raise RuntimeError("MlxVlmDFlashDriver.load() must be called first")
        return self._target

    @property
    def processor(self) -> Any:
        if self._processor is None:
            raise RuntimeError("MlxVlmDFlashDriver.load() must be called first")
        return self._processor

    @property
    def runtime(self) -> Any:
        if self._runtime is None:
            raise RuntimeError("MlxVlmDFlashDriver.load() must be called first")
        return self._runtime

    def load(self) -> None:
        if self.loaded:
            return
        from mlx_vlm import load as _mlx_vlm_load

        from .runtime import load_runtime

        logger.info("[dflash.driver] Loading target via mlx-vlm: %s", self.target_repo)
        self._target, self._processor = _mlx_vlm_load(self.target_repo)
        logger.info(
            "[dflash.driver] Loading DFlash drafter: %s (block_size=%r)",
            self.drafter_repo,
            self.block_size,
        )
        rt = load_runtime(self.drafter_repo)
        _validate_runtime_contract(rt)
        self._runtime = rt

    def adopt(
        self,
        *,
        target: Any,
        processor: Any,
        runtime: Any,
    ) -> None:
        if self.loaded:
            raise ValueError(
                "MlxVlmDFlashDriver is already loaded; construct a new driver."
            )
        if target is None or processor is None or runtime is None:
            raise ValueError("adopt() requires non-None target, processor, runtime")
        _validate_runtime_contract(runtime)
        self._target = target
        self._processor = processor
        self._runtime = runtime

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Generator[Any, None, None]:
        if not self.loaded:
            raise RuntimeError(
                "MlxVlmDFlashDriver.generate() requires load() to be called first"
            )
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1; got {max_tokens}")
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0.0, 1.0]; got {top_p}")
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0; got {temperature}")
        self.runtime.reset_accept_lens()
        drafter = self.runtime.drafter
        if hasattr(drafter, "draft_lens") and isinstance(drafter.draft_lens, list):
            drafter.draft_lens.clear()

        from mlx_vlm import stream_generate

        gen_kwargs: dict[str, Any] = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            draft_model=drafter,
            draft_kind=self.runtime.kind,
        )
        if self.block_size is not None:
            gen_kwargs["draft_block_size"] = self.block_size
        upstream = stream_generate(self.target, self.processor, prompt, **gen_kwargs)
        try:
            for chunk in upstream:
                yield chunk
        finally:
            close = getattr(upstream, "close", None)
            if callable(close):
                close()

    def accept_stats(self) -> dict[str, Any]:
        if self._runtime is None:
            return {
                "attempts": 0,
                "accepted_tokens": 0,
                "drafted_tokens": 0,
                "accept_rate": 0.0,
                "mean_accepted_per_attempt": 0.0,
                "accept_lens": [],
                "draft_lens": [],
            }
        drafter = self._runtime.drafter
        accept_lens = list(getattr(drafter, "accept_lens", []) or [])
        draft_lens = list(getattr(drafter, "draft_lens", []) or [])
        attempts = len(accept_lens)
        accepted = int(sum(accept_lens))
        drafted = int(sum(draft_lens))
        accept_rate = accepted / drafted if drafted > 0 else 0.0
        mean_accept = accepted / attempts if attempts > 0 else 0.0
        return {
            "attempts": attempts,
            "accepted_tokens": accepted,
            "drafted_tokens": drafted,
            "accept_rate": accept_rate,
            "mean_accepted_per_attempt": mean_accept,
            "accept_lens": accept_lens,
            "draft_lens": draft_lens,
        }
