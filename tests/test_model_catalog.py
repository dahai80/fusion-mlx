import json
import logging
from pathlib import Path

import pytest
from fusion_model_hub.catalog.schema import CatalogEntry
from pydantic import ValidationError

from fusion_mlx.api.openai._common import _catalog_file_candidates, _model_catalog_dir
from fusion_mlx.launch.cli import _resolve_default_model
from fusion_mlx.model_aliases import list_profiles

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.skipif(
    _model_catalog_dir() is None,
    reason="model catalog dir not present in this checkout",
)


def _catalog_files() -> list[Path]:
    return sorted(Path(_model_catalog_dir()).glob("*.json"))


def test_catalog_present_and_nonempty():
    files = _catalog_files()
    logger.info("model catalog: %d entries in %s", len(files), _model_catalog_dir())
    assert files, "catalog dir exists but holds no entries — CI gate would be vacuous"


def test_all_catalog_entries_validate_against_schema():
    errors = []
    for path in _catalog_files():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            CatalogEntry.model_validate(data)
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    assert not errors, "catalog schema drift detected:\n" + "\n".join(errors)


def test_no_duplicate_hub_ids():
    seen: dict[str, str] = {}
    for path in _catalog_files():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        hub_id = data.get("hubId", "")
        if hub_id:
            assert hub_id not in seen, (
                f"duplicate hubId {hub_id!r} in {path.name} and {seen[hub_id]}"
            )
            seen[hub_id] = path.name


def test_default_model_is_cataloged():
    default = _resolve_default_model()
    profile = list_profiles().get(default)
    assert profile is not None, f"default alias {default!r} missing from alias table"
    catalog_dir = _model_catalog_dir()
    matches = [
        name
        for name in _catalog_file_candidates(profile.hf_path)
        if (Path(catalog_dir) / f"{name}.json").is_file()
    ]
    assert matches, (
        f"served default model {default!r} ({profile.hf_path}) has no catalog "
        f"entry in {catalog_dir} — add one or the /v1/models listing degrades"
    )


def test_unknown_fields_are_rejected():
    files = _catalog_files()
    with open(files[0], encoding="utf-8") as f:
        data = json.load(f)
    drifted = dict(data)
    drifted["bogusFutureField"] = 1
    with pytest.raises(ValidationError):
        CatalogEntry.model_validate(drifted)
