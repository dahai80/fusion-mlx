# LSA context-limit resolution: the hard 400 preflight
# (enforce_context_length -> get_model_max_context) must see the LSA
# memory-budget context instead of the tokenizer's native
# max_position_embeddings, or every 1M-token LSA request is rejected
# before prefill (262144 tokenizer cap on a 2.5M-token budget).
import pytest

from fusion_mlx.service.helpers import get_model_max_context


class _DictArgsModel:
    def __init__(self, args):
        self.args = args


class _Engine:
    def __init__(self, model=None, lsa_max_context_tokens=0):
        if model is not None:
            self._model = model
        if lsa_max_context_tokens:
            self._lsa_max_context_tokens = lsa_max_context_tokens
        self._tokenizer = type(
            "_Tok", (), {"model_max_length": 262144}
        )()


def test_get_model_max_context_reads_dict_args_text_config():
    # Live Qwen3.8-27B exposes model.args as a plain dict with a dict
    # text_config; getattr-based probing missed both and fell through to
    # the tokenizer cap.
    model = _DictArgsModel(
        {"text_config": {"max_position_embeddings": 262144}}
    )
    assert get_model_max_context(_Engine(model=model)) == 262144


def test_get_model_max_context_prefers_lsa_budget():
    engine = _Engine(
        model=_DictArgsModel(
            {"text_config": {"max_position_embeddings": 262144}}
        ),
        lsa_max_context_tokens=2538048,
    )
    assert get_model_max_context(engine) == 2538048


def test_get_model_max_context_lsa_zero_falls_through():
    engine = _Engine(lsa_max_context_tokens=0)
    assert get_model_max_context(engine) == 262144
