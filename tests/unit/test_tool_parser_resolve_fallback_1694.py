from unittest.mock import MagicMock

from fusion_mlx.api.openai import streaming


def test_resolve_streaming_tool_parser_falls_back_to_auto(monkeypatch):
    calls = {}

    def boom(model_name, profile_tool, profile_reasoning):
        calls["model"] = model_name
        raise RuntimeError("profile lookup IO error")

    class FakeAutoParser:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    monkeypatch.setattr(
        "fusion_mlx.routes_internal.models.effective_parsers_for",
        boom,
    )
    from fusion_mlx.tool_parsers.abstract_tool_parser import ToolParserManager

    monkeypatch.setattr(
        ToolParserManager,
        "get_tool_parser",
        classmethod(lambda cls, name: FakeAutoParser if name == "auto" else None),
    )
    engine = MagicMock()
    engine._tokenizer = "tok"
    parser = streaming._resolve_streaming_tool_parser(engine, "mymodel")
    assert isinstance(parser, FakeAutoParser)
    assert parser.tokenizer == "tok"
    assert calls["model"] == "mymodel"


def test_resolve_streaming_tool_parser_double_failure_returns_none(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(
        "fusion_mlx.routes_internal.models.effective_parsers_for",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("first boom")),
    )

    def always_raise(name):
        raise KeyError(f"parser '{name}' not found")

    from fusion_mlx.tool_parsers.abstract_tool_parser import ToolParserManager

    monkeypatch.setattr(
        ToolParserManager,
        "get_tool_parser",
        classmethod(lambda cls, name: always_raise(name)),
    )
    with caplog.at_level(logging.WARNING, logger=streaming.logger.name):
        parser = streaming._resolve_streaming_tool_parser(MagicMock(), "mymodel")
    assert parser is None
    assert any("#1694" in r.message for r in caplog.records)


def test_resolve_streaming_tool_parser_happy_path(monkeypatch):
    class FakeHermes:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

    monkeypatch.setattr(
        "fusion_mlx.routes_internal.models.effective_parsers_for",
        lambda *a, **kw: ("hermes", None),
    )
    from fusion_mlx.tool_parsers.abstract_tool_parser import ToolParserManager

    monkeypatch.setattr(
        ToolParserManager,
        "get_tool_parser",
        classmethod(lambda cls, name: FakeHermes),
    )
    engine = MagicMock()
    engine._tokenizer = "tok"
    parser = streaming._resolve_streaming_tool_parser(engine, "mymodel")
    assert isinstance(parser, FakeHermes)
