"""Streaming chat completion — reads SSE tokens as they arrive."""
import json
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"

payload = {
    "model": "Qwen2.5-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Count from 1 to 10."}],
    "max_tokens": 100,
    "stream": True,
}

req = urllib.request.Request(
    URL,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)

print("Streaming response: ", end="", flush=True)
with urllib.request.urlopen(req) as resp:
    for line in resp:
        line = line.decode().strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                content = obj["choices"][0]["delta"].get("content", "")
                if content:
                    print(content, end="", flush=True)
            except json.JSONDecodeError:
                pass

print()  # final newline
