"""Text-to-Speech (TTS) — generate WAV audio from text."""
import json
import urllib.request

URL = "http://localhost:8000/v1/audio/speech"

payload = {
    "model": "kokoro",
    "input": "Hello! This is a text-to-speech demonstration powered by fusion-mlx.",
    "voice": "default",
    "speed": 1.0,
    "response_format": "wav",
}

req = urllib.request.Request(
    URL,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(req) as resp:
    with open("output.wav", "wb") as f:
        f.write(resp.read())
    print("Saved to output.wav")
