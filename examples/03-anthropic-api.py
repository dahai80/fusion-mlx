"""Anthropic Messages API — fusion-mlx is Anthropic-compatible."""
import json
import urllib.request

URL = "http://localhost:8000/v1/messages"

payload = {
    "model": "Qwen2.5-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Name three prime numbers."}],
    "max_tokens": 100,
}

req = urllib.request.Request(
    URL,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())
    for block in result.get("content", []):
        if block["type"] == "text":
            print(block["text"])
    print(f"\nTokens: {result['usage']['input_tokens']} in, {result['usage']['output_tokens']} out")
