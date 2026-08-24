from __future__ import annotations

import fusion_mlx
import fusion_mlx.public_api as public_api


class TestPublicApiStable:
    def test_all_symbols_importable(self):
        for name in public_api.__all__:
            assert hasattr(public_api, name), f"public_api.__all__ 声明 {name} 但不可 import"

    def test_public_api_reexports_match_internal(self):
        from fusion_mlx.config import MemoryConfig, MemoryTier, ServerConfig, get_config
        from fusion_mlx.engines import (
            EmbeddingEngine,
            ImageGenEngine,
            RerankerEngine,
            STSEngine,
            STTEngine,
            TTSEngine,
            VideoGenEngine,
        )
        from fusion_mlx.model_registry import get_registry
        from fusion_mlx.server import Server, create_app
        from fusion_mlx.video.latentsync_mlx.pipeline import LipsyncPipelineMLX
        from fusion_mlx.video.pulid_mlx.pipeline import PuLIDPipeline

        assert public_api.TTSEngine is TTSEngine
        assert public_api.ImageGenEngine is ImageGenEngine
        assert public_api.VideoGenEngine is VideoGenEngine
        assert public_api.STTEngine is STTEngine
        assert public_api.STSEngine is STSEngine
        assert public_api.EmbeddingEngine is EmbeddingEngine
        assert public_api.RerankerEngine is RerankerEngine
        assert public_api.get_config is get_config
        assert public_api.get_registry is get_registry
        assert public_api.ServerConfig is ServerConfig
        assert public_api.MemoryConfig is MemoryConfig
        assert public_api.MemoryTier is MemoryTier
        assert public_api.Server is Server
        assert public_api.create_app is create_app
        assert public_api.LipsyncPipelineMLX is LipsyncPipelineMLX
        assert public_api.PuLIDPipeline is PuLIDPipeline

    def test_version_exposed(self):
        assert public_api.__version__ == fusion_mlx.__version__
        assert isinstance(public_api.__version__, str)

    def test_toplevel_reexports_videogenengine(self):
        assert hasattr(fusion_mlx, "VideoGenEngine")
        from fusion_mlx.engines import VideoGenEngine
        assert fusion_mlx.VideoGenEngine is VideoGenEngine
