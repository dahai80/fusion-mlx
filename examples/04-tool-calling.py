"""Tool calling — define functions and let the model call them."""
import json
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }
]

payload = {
    "model": "Qwen2.5-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": tools,
    "max_tokens": 256,
}

req = urllib.request.Request(
    URL,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())
    choice = result["choices"][0]["message"]

    if choice.get("tool_calls"):
        for tc in choice["tool_calls"]:
            print(f"Tool: {tc['function']['name']}")
            print(f"Args: {tc['function']['arguments']}")
    else:
        print(choice["content"])
