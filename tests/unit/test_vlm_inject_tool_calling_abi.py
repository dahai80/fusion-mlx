from fusion_mlx.engines.vlm import VLMBatchedEngine


class _FakeTokenizer:
    def __init__(self, chat_template):
        self.chat_template = chat_template
        self.has_tool_calling = False
        self._vocab = {}

    def get_vocab(self):
        return self._vocab


_MINIMAX_TEMPLATE = "helpers <minimax:tool_call> invoke"


def _make_engine():
    return VLMBatchedEngine.__new__(VLMBatchedEngine)


def _load_module_tool_tokens(monkeypatch):
    from mlx_vlm import tool_parsers

    module = tool_parsers.load_tool_module("minimax_m2")
    return module.tool_call_start, module.tool_call_end


def test_inject_accepts_tokenizer_object_abi(monkeypatch):
    start, end = _load_module_tool_tokens(monkeypatch)
    tok = _FakeTokenizer(_MINIMAX_TEMPLATE)
    tok._vocab = {start: 1, end: 2}
    engine = _make_engine()
    engine._inject_tool_calling(tok)
    assert tok.has_tool_calling is True
    assert tok.tool_parser is not None


def test_inject_falls_back_to_string_abi(monkeypatch):
    # simulate an older mlx-lm whose _infer_tool_parser takes the template
    # string: passing a tokenizer object raises TypeError on `in` checks
    from mlx_vlm import tool_parsers

    def old_abi(chat_template):
        if "<minimax:tool_call>" in chat_template:
            return "minimax_m2"
        return None

    monkeypatch.setattr(tool_parsers, "_infer_tool_parser", old_abi)
    start, end = _load_module_tool_tokens(monkeypatch)
    tok = _FakeTokenizer(_MINIMAX_TEMPLATE)
    tok._vocab = {start: 1, end: 2}
    engine = _make_engine()
    engine._inject_tool_calling(tok)
    assert tok.has_tool_calling is True


def test_inject_noop_without_chat_template():
    tok = _FakeTokenizer(None)
    engine = _make_engine()
    engine._inject_tool_calling(tok)
    assert tok.has_tool_calling is False
