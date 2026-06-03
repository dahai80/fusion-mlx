"""Basic non-streaming chat completion via curl-equivalent HTTP request."""
import json
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"

payload = {
    "model": "Qwen2.5-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "max_tokens": 10,
}

req = urllib.request.Request(
    URL,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())
    print(result["choices"][0]["message"]["content"])
