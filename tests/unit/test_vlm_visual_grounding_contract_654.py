# SPDX-License-Identifier: Apache-2.0
"""Visual-grounding VLM contract — regression for issue #654.

fusion-browser's T3.4 visual-grounding fallback screenshots a stale DOM
node via WKWebView, encodes the PNG as a base64 data URI, and asks a VLM
loaded in fusion-mlx to predict ``{"x":<int>,"y":<int>}`` click coords
against ``POST /v1/chat/completions``. The smoke test already passed
against a real VLM; this file locks the contract so a future fusion-mlx
change does not silently break visual grounding.

Three load-bearing promises pinned here:

1. **Base64 data-URI ``image_url`` is decoded to a real PIL image.**
   ``load_image`` accepts ``data:image/png;base64,...`` (not just remote
   URLs, which are refused for SSRF). ``extract_images_from_messages``
   pulls the ``.url`` out of an ``image_url`` part and round-trips it
   through ``load_image`` to a loaded PIL ``Image``. fusion-browser
   inlines WKWebView snapshots as data URIs and does NOT host them over
   HTTP, so data-URI support is the hard requirement.

2. **Remote image URLs are refused (SSRF), not silently fetched.**
   A ``data:`` URI or local path is the only accepted form; an
   ``http(s)://`` ``image_url`` raises and the part is skipped. This is
   the security boundary fusion-browser relies on staying intact.

3. **``response_format: {type: "json_object"}`` is honored on the VLM
   path.** ``_compile_grammar_for_request`` is called unconditionally
   (no ``is_mllm`` bypass) and the VLM engine exposes a
   ``grammar_compiler``, so a json_object request compiles a real
   grammar that constrains decoding — letting fusion-browser switch to
   strict JSON mode for coordinate output.

No real model is loaded — these are contract/unit tests, run in CI
without an Apple Silicon GPU.
"""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from fusion_mlx.api.openai_routes import _compile_grammar_for_request
from fusion_mlx.utils.image import extract_images_from_messages, load_image


def _png_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


# --- Promise 1: base64 data-URI image_url decodes to a real PIL image ---


def test_extract_images_decodes_base64_data_uri_png():
    """extract_images_from_messages turns a data-URI image_url part into a
    loaded PIL Image — the path fusion-browser's WKWebView snapshot takes."""
    img = Image.new("RGB", (8, 8), color=(10, 20, 30))
    uri = _png_data_uri(img)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Predict the click point as JSON."},
                {"type": "image_url", "image_url": {"url": uri}},
            ],
        }
    ]

    text_msgs, images, videos, audio = extract_images_from_messages(messages)

    assert len(images) == 1, f"expected 1 decoded image, got {len(images)}"
    decoded = images[0]
    assert isinstance(decoded, Image.Image)
    assert decoded.size == (8, 8)
    assert decoded.mode == "RGB"

    # Media part stripped from the text message the model sees.
    assert len(text_msgs) == 1
    assert "Predict the click point" in text_msgs[0]["content"]
    assert "[image]" not in text_msgs[0]["content"]
    assert videos == []
    assert audio == []


def test_extract_images_decodes_data_uri_without_media_type():
    """A ``data:;base64,...`` URI (no media type) still decodes — some
    WKWebView snapshot paths emit the bare form."""
    img = Image.new("RGB", (4, 4), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    uri = f"data:;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": uri}}],
        }
    ]

    _, images, _, _ = extract_images_from_messages(messages)
    assert len(images) == 1
    assert images[0].size == (4, 4)


# --- Promise 2: remote URLs refused (SSRF), part skipped not fetched ---


def test_extract_images_skips_remote_url_part():
    """An http(s) image_url is refused by load_image (SSRF guard) and the
    part is skipped — fusion-browser must inline as data: URI, never rely on
    the server fetching a remote screenshot."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/shot.png"},
                },
            ],
        }
    ]

    _, images, _, _ = extract_images_from_messages(messages)
    assert images == [], "remote URL must not be fetched or decoded"


def test_load_image_refuses_remote_url():
    with pytest.raises(ValueError, match="Remote image URLs"):
        load_image("https://example.com/shot.png")


# --- Promise 3: response_format json_object honored on the VLM path ---


class _FakeGrammarCompiler:
    """Stand-in for engine.grammar_compiler. Records the schema it was
    asked to compile and returns a sentinel so the test can assert the
    json_object branch ran — without needing a real xgrammar install or
    tokenizer (keeps the test deterministic in any CI env)."""

    def __init__(self):
        self.compiled: list[str] = []
        self.sentinel = object()

    def compile_json_schema(self, schema: str):
        self.compiled.append(schema)
        return self.sentinel


class _StubVLMEngine:
    """Minimal VLM engine shape: is_mllm=True + grammar_compiler present,
    so _compile_grammar_for_request takes the xgrammar branch."""

    is_mllm = True

    def __init__(self):
        self.grammar_compiler = _FakeGrammarCompiler()


def _request(response_format, grammar_backend="xgrammar"):
    """Build a ChatCompletionRequest-ish namespace with only the fields
    _compile_grammar_for_request reads (structured_outputs /
    response_format / grammar_backend)."""
    return SimpleNamespace(
        structured_outputs=None,
        response_format=response_format,
        grammar_backend=grammar_backend,
    )


def test_json_object_compiles_grammar_for_vlm_engine():
    """response_format:json_object must compile a non-None grammar on a
    VLM engine (is_mllm=True, grammar_compiler present). This is the
    contract fusion-browser relies on to switch visual-grounding
    coordinate output to strict JSON mode.

    _compile_grammar_for_request (openai_routes.py:383) is called
    unconditionally on the chat path — no is_mllm bypass — so the VLM
    branch is the same code path as the text-LLM branch. Pin: the json
    object branch (openai_routes.py:420-421) turns the request into
    grammar_spec={"json_schema": "{}"} and the xgrammar branch
    (:451-463) calls engine.grammar_compiler.compile_json_schema("{}").
    """
    engine = _StubVLMEngine()
    req = _request({"type": "json_object"})

    compiled = _compile_grammar_for_request(engine, req)

    assert (
        compiled is engine.grammar_compiler.sentinel
    ), "json_object did not compile a grammar for the VLM engine"
    assert engine.grammar_compiler.compiled == [
        "{}"
    ], f"expected compile_json_schema('{{}}'), got {engine.grammar_compiler.compiled!r}"


def test_json_object_compiles_on_vlm_same_path_as_text():
    """The VLM path must NOT bypass grammar compilation. An engine with
    is_mllm=True still receives a compiled grammar for json_object —
    proving there is no is_mllm guard that skips VLM requests."""
    vlm_engine = _StubVLMEngine()
    req = _request({"type": "json_object"})

    compiled = _compile_grammar_for_request(vlm_engine, req)

    assert (
        compiled is not None
    ), "VLM (is_mllm=True) request must compile grammar, not bypass"


def test_no_response_format_returns_none_for_vlm():
    """A VLM request without response_format compiles nothing — the
    grammar path is opt-in, never forced on plain visual chat."""
    engine = _StubVLMEngine()
    req = _request(None)

    assert _compile_grammar_for_request(engine, req) is None
    assert engine.grammar_compiler.compiled == []
