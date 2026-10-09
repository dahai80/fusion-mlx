from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_GRAMMARS_DIR = Path(__file__).resolve().parent.parent / "grammars"

_GRAMMAR_ALIASES: dict[str, dict[str, str]] = {
    "bnup-socratic": {
        "file": "socratic_dsl_v3.gbnf",
        "format": "gbnf",
        "description": "BNUP Socratic DSL v3.0 - <thinking>+<dsl> constrained output",
    },
}


def list_grammar_aliases() -> list[dict[str, str]]:
    return [
        {"alias": name, "format": info["format"], "description": info["description"]}
        for name, info in _GRAMMAR_ALIASES.items()
    ]


def resolve_grammar_alias(name: str) -> dict | None:
    info = _GRAMMAR_ALIASES.get(name)
    if info is None:
        return None
    path = _GRAMMARS_DIR / info["file"]
    if not path.exists():
        logger.error("grammar alias %r: file not found: %s", name, path)
        return None
    try:
        content = path.read_text(encoding="utf-8")
        logger.info(
            "grammar alias %r resolved (%s, %d bytes)",
            name,
            info["format"],
            len(content),
        )
        return {"grammar": content, "format": info["format"]}
    except Exception as e:
        logger.error("grammar alias %r: failed to read %s: %s", name, path, e)
        return None


def is_grammar_alias(name: str) -> bool:
    return name in _GRAMMAR_ALIASES
