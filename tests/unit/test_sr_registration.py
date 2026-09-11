import logging

logger = logging.getLogger(__name__)


def test_public_api_reexports_sr():
    from fusion_mlx.public_api import (
        RealESRGANConfig,
        RRDBNet,
        super_resolve,
    )

    assert callable(super_resolve)
    assert callable(RealESRGANConfig)
    assert isinstance(RRDBNet, type)
    logger.info("public_api re-exports SR symbols OK")


def _walk_routes(app):
    for r in app.routes:
        if hasattr(r, "original_router"):
            yield from r.original_router.routes
        else:
            yield r


def test_sr_route_registered():
    from fusion_mlx.config import ServerConfig
    from fusion_mlx.server import create_app

    # R-7: SR route requires image modality — use full profile.
    cfg = ServerConfig()
    cfg.profile = "full"
    app = create_app(cfg)
    paths = {getattr(r, "path", None) for r in _walk_routes(app)}
    assert "/v1/images/super-resolution" in paths
    logger.info("SR route registered in create_app() OK")
