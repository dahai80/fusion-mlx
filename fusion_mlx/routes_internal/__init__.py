"""Internal route handlers called by ``api/*_routes.py`` FastAPI routers.

These modules contain request-processing logic (validation, engine dispatch,
response formatting) but do NOT define ``APIRouter`` instances — those live
in ``fusion_mlx.api/``. The split keeps router definitions (URLs, deps,
status codes) separate from handler internals."""
