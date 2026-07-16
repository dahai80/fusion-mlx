from __future__ import annotations

# The os.system / __import__ strings below are INERT injection payloads: they
# live inside string literals passed as graph field values. _generate_python_script
# escapes them into a JSON literal and compile() only parses (never executes),
# so no shell ever runs. They exist precisely to assert the injection is blocked.
from fusion_mlx.api.agent_routes import _generate_python_script


def _graph(name: str = "t", **node_fields) -> dict:
    node = {"type": "llm", "model": "qwen3.5-9b"}
    node.update(node_fields)
    return {"name": name, "nodes": {"n1": node}, "edges": []}


def test_temperature_injection_is_coerced():
    payload = '0.7, __import__("os").system("id")'
    script = _generate_python_script(_graph(temperature=payload))
    compile(script, "<gen>", "exec")
    assert "__import__" not in script
    assert '"temperature": 0.7' in script


def test_name_with_quote_and_newline_no_breakout():
    name = 'x"\nimport os; os.system("id")'
    script = _generate_python_script(_graph(name=name))
    compile(script, "<gen>", "exec")
    assert "\nimport os" not in script


def test_system_prompt_with_quote_no_breakout():
    prompt = '"}]; import os; os.system("id"); #'
    script = _generate_python_script(_graph(system_prompt=prompt))
    compile(script, "<gen>", "exec")


def test_model_with_quote_no_breakout():
    model = 'm"; import os; os.system("id"); #'
    script = _generate_python_script(_graph(model=model))
    compile(script, "<gen>", "exec")


def test_temperature_out_of_range_clamped():
    script = _generate_python_script(_graph(temperature=5.0))
    compile(script, "<gen>", "exec")
    assert '"temperature": 0.7' in script


def test_temperature_non_numeric_clamped():
    script = _generate_python_script(_graph(temperature="hot"))
    compile(script, "<gen>", "exec")
    assert '"temperature": 0.7' in script


def test_normal_values_embedded():
    script = _generate_python_script(
        _graph(
            name="my-agent",
            model="qwen3.5-9b",
            system_prompt="hi",
            temperature=0.5,
        )
    )
    compile(script, "<gen>", "exec")
    assert '"name": "my-agent"' in script
    assert '"model": "qwen3.5-9b"' in script
    assert '"system_prompt": "hi"' in script
    assert '"temperature": 0.5' in script


def test_generated_script_uses_config_literal():
    script = _generate_python_script(_graph())
    assert "_CONFIG" in script
    assert "input(" in script
