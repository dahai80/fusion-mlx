import asyncio
from unittest.mock import MagicMock, patch
import pytest


class TestAsyncEngineCore:
     @pytest.mark.asyncio
     async def test_start_stores_task_reference(self):
        with patch("fusion_mlx.engine_core.EngineCore") as MockEngine, \
             patch("fusion_mlx.engine_core.mx") as mock_mx:
            from fusion_mlx.engine_core import AsyncEngineCore

            mock_engine = MagicMock()
            mock_engine.start = MagicMock(return_value=asyncio.sleep(0))
            MockEngine.return_value = mock_engine

            core = AsyncEngineCore(MagicMock(), MagicMock())
            assert core._start_task is None

            core.start()
            assert core._start_task is not None
            assert isinstance(core._start_task, asyncio.Task)
