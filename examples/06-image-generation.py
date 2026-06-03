"""Image generation with Flux 2 via the OpenAI-compatible images API."""
import base64
import json
import urllib.request

URL = "http://localhost:8000/v1/images/generate"

payload = {
    "prompt": "A serene mountain lake at golden hour, photorealistic",
    "n": 1,
    "width": 1024,
    "height": 1024,
    "steps": 20,
    "guidance": 7.5,
}

req = urllib.request.Request(
    URL,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())
    for img in result["data"]:
        if img.get("b64_json"):
            # Save as PNG
            with open("generated.png", "wb") as f:
                f.write(base64.b64decode(img["b64_json"]))
            print("Saved to generated.png")
        elif img.get("url"):
            print(f"Image URL: {img['url']}")
