"""Concurrent requests to multiple models simultaneously."""
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://localhost:8000/v1/chat/completions"

def query_model(model_name, prompt):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 20,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"]

models = [
    ("Qwen2.5-3B-Instruct-4bit", "What is the capital of France? One word."),
    ("Qwen3.6-27B-mxfp8", "What is the capital of Japan? One word."),
]

with ThreadPoolExecutor(max_workers=len(models)) as pool:
    futures = [pool.submit(query_model, name, prompt) for name, prompt in models]
    for (name, _), fut in zip(models, futures):
        print(f"{name}: {fut.result()}")
