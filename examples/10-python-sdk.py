"""Use the OpenAI Python SDK with fusion-mlx as a local backend.

Requires: pip install openai
"""
try:
    from openai import OpenAI
except ImportError:
    print("Install openai first: pip install openai")
    raise

# Point OpenAI client at fusion-mlx
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="local",  # fusion-mlx doesn't require auth by default
)

# Non-streaming chat
response = client.chat.completions.create(
    model="Qwen2.5-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "What is 3*7? One number."}],
    max_tokens=10,
)
print(f"Answer: {response.choices[0].message.content}")
print(f"Tokens: {response.usage.prompt_tokens} prompt, {response.usage.completion_tokens} completion")

# Streaming chat
print("\nStreaming:")
stream = client.chat.completions.create(
    model="Qwen2.5-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "List three colors."}],
    max_tokens=50,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()

# List models
models = client.models.list()
print(f"\nAvailable models: {[m.id for m in models.data]}")
