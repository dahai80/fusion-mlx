# SPDX-License-Identifier: Apache-2.0
"""Grammar compilation helpers extracted from chat.py."""

from __future__ import annotations

from ...engines.base import GenerationOutput
from ..adapters.base import InternalResponse
from ..grammar import GrammarBackend, resolve_grammar_backend
from ..openai_models import ChatCompletionRequest
from ._common import logger


def _extract_strict_json_schema(req: ChatCompletionRequest):
    # #514: extract the JSON schema dict the post-generate validator
    # should validate against, but ONLY when the request asked for a
    # STRICT json_schema (strict=true). Returns None for non-strict /
    # json_object / absent response_format so the R12-4 postgen path
    # does not run on loose-schema requests (constrained decoding or
    # unconstrained generation is the intended behavior there).
    # Mirrors the schema extraction in _compile_grammar_for_request.
    from ..tool_calling import is_strict_json_schema

    rf = getattr(req, "response_format", None)
    if rf is None:
        return None
    if not is_strict_json_schema(rf):
        return None
    if isinstance(rf, dict):
        schema = rf.get("json_schema", {})
        if isinstance(schema, dict) and "schema" in schema:
            return schema["schema"]
        return schema
    if hasattr(rf, "type") and rf.type == "json_schema":
        inner = getattr(rf, "json_schema", None)
        if inner and hasattr(inner, "schema_"):
            return inner.schema_
        if inner and hasattr(inner, "schema"):
            return inner.schema
    return None


def _compile_grammar_for_request(engine, req: ChatCompletionRequest):
    """Compile grammar from request's structured_outputs / response_format.

    Returns a backend-specific compiled grammar object (xgrammar CompiledGrammar
    or llguidance LLMatcher), or None if no grammar constraint is requested.
    """
    so = getattr(req, "structured_outputs", None)
    grammar_backend_str = getattr(req, "grammar_backend", None)
    if so is None and getattr(req, "response_format", None) is None:
        return None

    backend = resolve_grammar_backend(grammar_backend_str)
    grammar_spec = None

    if so is not None:
        if isinstance(so, dict):
            grammar_spec = so
        else:
            grammar_spec = {}
            if so.json_schema is not None:
                grammar_spec["json_schema"] = so.json_schema
            if so.regex is not None:
                grammar_spec["regex"] = so.regex
            if so.choice is not None:
                grammar_spec["choice"] = so.choice
            if so.grammar is not None:
                grammar_spec["grammar"] = so.grammar

    rf = getattr(req, "response_format", None)
    if rf is not None and grammar_spec is None:
        if isinstance(rf, dict):
            if rf.get("type") == "json_schema":
                schema = rf.get("json_schema", {})
                if isinstance(schema, dict) and "schema" in schema:
                    grammar_spec = {"json_schema": schema["schema"]}
                else:
                    grammar_spec = {"json_schema": schema}
            elif rf.get("type") == "json_object":
                grammar_spec = {"json_schema": "{}"}
        elif hasattr(rf, "type"):
            if rf.type == "json_schema":
                inner = getattr(rf, "json_schema", None)
                if inner and hasattr(inner, "schema_"):
                    grammar_spec = {"json_schema": inner.schema_}
                elif inner and hasattr(inner, "schema"):
                    grammar_spec = {"json_schema": inner.schema}
            elif rf.type == "json_object":
                grammar_spec = {"json_schema": "{}"}

    if grammar_spec is None:
        return None

    if backend == GrammarBackend.LLGUIDANCE:
        from ..grammar import create_llguidance_matcher

        vocab_size = None
        if hasattr(engine, "_model"):
            from ...utils.tokenizer import resolve_vocab_size

            vocab_size = resolve_vocab_size(engine._model)
        matcher = create_llguidance_matcher(
            engine._tokenizer, grammar_spec, vocab_size=vocab_size
        )
        if matcher is not None:
            logger.info("compiled grammar via llguidance for request")
            return matcher
        logger.warning("llguidance compilation failed, trying xgrammar fallback")

    if backend in (GrammarBackend.XGRAMMAR, GrammarBackend.LLGUIDANCE):
        compiler = getattr(engine, "grammar_compiler", None)
        if compiler is None:
            logger.debug("no grammar_compiler on engine, skipping grammar compilation")
            return None
        try:
            if "json_schema" in grammar_spec:
                import json

                schema = grammar_spec["json_schema"]
                if isinstance(schema, dict):
                    schema = json.dumps(schema)
                return compiler.compile_json_schema(schema)
            if "regex" in grammar_spec:
                return compiler.compile_regex(grammar_spec["regex"])
            if "choice" in grammar_spec:
                import json

                return compiler.compile_json_schema(
                    json.dumps(
                        {
                            "type": "string",
                            "enum": grammar_spec["choice"],
                        }
                    )
                )
            if "grammar" in grammar_spec:
                return compiler.compile_grammar(grammar_spec["grammar"])
        except Exception as exc:
            logger.warning("xgrammar compilation failed: %s", exc)
    return None


def _gen_to_internal(
    gen: GenerationOutput, model: str, request_id: str
) -> InternalResponse:
    """Convert GenerationOutput to InternalResponse for the adapter."""
    return InternalResponse(
        text=gen.text,
        finish_reason=gen.finish_reason,
        prompt_tokens=gen.prompt_tokens,
        completion_tokens=gen.completion_tokens,
        cached_tokens=gen.cached_tokens,
        tool_calls=gen.tool_calls,
        request_id=request_id,
        model=model,
        logprobs=getattr(gen, "logprobs", None),
        model_load_duration=getattr(gen, "model_load_duration", None),
        time_to_first_token=getattr(gen, "time_to_first_token", None),
        generation_tokens_per_second=(
            gen.generation_tokens_per_second
            if getattr(gen, "generation_tokens_per_second", None) is not None
            else (gen.generation_tps or None)
        ),
    )
