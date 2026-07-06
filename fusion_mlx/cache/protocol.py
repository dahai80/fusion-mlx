# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "1"
MANIFEST_FILENAME = "manifest.json"

_DEFAULT_EXPORT_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "fusion-mlx", "cache_exports"
)
_EXPORT_DIR_ENV = "FUSION_MLX_CACHE_EXPORT_DIR"


class InvalidExportPathError(ValueError):
    pass


class ManifestNotFoundError(FileNotFoundError):
    pass


class MalformedManifestError(ValueError):
    pass


class ManifestMismatchError(ValueError):
    def __init__(self, field: str, expected: str, actual: str) -> None:
        super().__init__(
            f"manifest {field} mismatch: expected {expected!r}, got {actual!r}"
        )
        self.field = field
        self.expected = expected
        self.actual = actual


_FIELD_TYPES: dict[str, type] = {
    "protocol_version": str,
    "model_id": str,
    "quantization": str,
    "paged_cache": bool,
    "turboquant_kv": bool,
    "index_format_version": int,
    "entries": int,
    "total_bytes": int,
    "fusion_mlx_version": str,
    "created_at": str,
    "extra": dict,
}


def _is_expected_type(value: object, expected: type) -> bool:
    if expected is int and isinstance(value, bool):
        return False
    return isinstance(value, expected)


@dataclass
class Manifest:
    protocol_version: str = PROTOCOL_VERSION
    model_id: str = ""
    quantization: str = ""
    paged_cache: bool = False
    turboquant_kv: bool = False
    index_format_version: int = 0
    entries: int = 0
    total_bytes: int = 0
    fusion_mlx_version: str = ""
    created_at: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        filtered = {}
        for key, value in data.items():
            if key not in _FIELD_TYPES:
                continue
            expected = _FIELD_TYPES[key]
            if not _is_expected_type(value, expected):
                raise MalformedManifestError(
                    f"manifest field {key!r}: expected {expected.__name__}, "
                    f"got {type(value).__name__}"
                )
            filtered[key] = value
        return cls(**filtered)


def write_manifest(root: Path, manifest: Manifest) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    target = root / MANIFEST_FILENAME
    fd, tmp_name = tempfile.mkstemp(
        prefix=".manifest-", suffix=".json", dir=str(root)
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return target


def read_manifest(root: Path) -> Manifest:
    target = root / MANIFEST_FILENAME
    if not target.is_file():
        exc = ManifestNotFoundError("manifest.json not found")
        exc.filename = str(target)
        raise exc
    try:
        data = json.loads(target.read_text())
    except json.JSONDecodeError as exc:
        raise MalformedManifestError(
            f"manifest.json is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise MalformedManifestError(
            f"manifest.json must decode to a JSON object, got {type(data).__name__}"
        )
    return Manifest.from_dict(data)


def default_export_root() -> Path:
    raw = os.environ.get(_EXPORT_DIR_ENV) or _DEFAULT_EXPORT_DIR
    return Path(os.path.realpath(os.path.expanduser(raw)))


def resolve_cache_dir(caller_path: str | None) -> Path:
    root = default_export_root()
    root.mkdir(parents=True, exist_ok=True)

    if caller_path is None or caller_path == "":
        return root

    if ".." in Path(caller_path).parts:
        raise InvalidExportPathError(
            f"path component '..' is not allowed: {caller_path!r}"
        )

    candidate = Path(caller_path)
    if not candidate.is_absolute():
        candidate = root / candidate

    resolved = Path(os.path.realpath(candidate))

    try:
        common = Path(os.path.commonpath([str(root), str(resolved)]))
    except ValueError as exc:
        raise InvalidExportPathError(
            f"path {caller_path!r} could not be compared to sandbox root"
        ) from exc

    if common != root:
        raise InvalidExportPathError(
            f"path {caller_path!r} resolves outside sandbox {root}"
        )

    return resolved
