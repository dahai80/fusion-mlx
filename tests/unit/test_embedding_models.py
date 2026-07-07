# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fusion_mlx.api.embedding_models — Pydantic schemas.

Covers EmbeddingInputItem validation (text/image required, extra forbidden),
EmbeddingRequest input-source mutual exclusion (input vs items), EmbeddingData
defaults, EmbeddingUsage fields, EmbeddingResponse shape. Aims at ≥90% line
coverage of embedding_models.py.
"""

from __future__ import annotations

import pytest

from fusion_mlx.api.embedding_models import (
    EmbeddingData,
    EmbeddingInputItem,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)


class TestEmbeddingInputItem:
    def test_text_only_ok(self):
        i = EmbeddingInputItem(text="hello")
        assert i.text == "hello"
        assert i.image is None

    def test_image_only_ok(self):
        i = EmbeddingInputItem(image="http://x/a.png")
        assert i.image == "http://x/a.png"
        assert i.text is None

    def test_text_and_image_ok(self):
        i = EmbeddingInputItem(text="hello", image="http://x/a.png")
        assert i.text == "hello"
        assert i.image == "http://x/a.png"

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="must include text or image"):
            EmbeddingInputItem()

    def test_extra_field_forbidden(self):
        with pytest.raises(Exception):
            EmbeddingInputItem(text="x", audio="y")


class TestEmbeddingRequest:
    def test_input_string_ok(self):
        r = EmbeddingRequest(input="hello", model="m")
        assert r.input == "hello"
        assert r.items is None
        assert r.encoding_format == "float"
        assert r.dimensions is None

    def test_input_list_ok(self):
        r = EmbeddingRequest(input=["a", "b"], model="m")
        assert r.input == ["a", "b"]

    def test_items_ok(self):
        r = EmbeddingRequest(
            items=[EmbeddingInputItem(text="a"), EmbeddingInputItem(text="b")],
            model="m",
        )
        assert r.items is not None
        assert len(r.items) == 2

    def test_neither_input_nor_items_raises(self):
        with pytest.raises(ValueError, match="Either input or items"):
            EmbeddingRequest(model="m")

    def test_both_input_and_items_raises(self):
        with pytest.raises(ValueError, match="cannot be provided together"):
            EmbeddingRequest(
                input="x",
                items=[EmbeddingInputItem(text="a")],
                model="m",
            )

    def test_empty_items_raises(self):
        with pytest.raises(ValueError, match="items cannot be empty"):
            EmbeddingRequest(items=[], model="m")

    def test_encoding_format_base64(self):
        r = EmbeddingRequest(input="x", model="m", encoding_format="base64")
        assert r.encoding_format == "base64"

    def test_dimensions(self):
        r = EmbeddingRequest(input="x", model="m", dimensions=128)
        assert r.dimensions == 128


class TestEmbeddingData:
    def test_defaults(self):
        d = EmbeddingData(index=0, embedding=[0.1, 0.2])
        assert d.object == "embedding"
        assert d.index == 0
        assert d.embedding == [0.1, 0.2]

    def test_base64_embedding_str(self):
        d = EmbeddingData(index=1, embedding="base64str")
        assert d.embedding == "base64str"


class TestEmbeddingUsage:
    def test_fields(self):
        u = EmbeddingUsage(prompt_tokens=5, total_tokens=5)
        assert u.prompt_tokens == 5
        assert u.total_tokens == 5


class TestEmbeddingResponse:
    def test_shape(self):
        r = EmbeddingResponse(
            data=[EmbeddingData(index=0, embedding=[0.1])],
            model="m",
            usage=EmbeddingUsage(prompt_tokens=1, total_tokens=1),
        )
        assert r.object == "list"
        assert len(r.data) == 1
        assert r.model == "m"
        assert r.usage.prompt_tokens == 1
