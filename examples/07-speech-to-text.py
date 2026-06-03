"""Speech-to-Text (STT) — transcribe an audio file via multipart upload."""
import json
import mimetypes
import uuid

BOUNDARY = f"boundary-{uuid.uuid4().hex}"
URL = "http://localhost:8000/v1/audio/transcriptions"

# Path to your audio file
AUDIO_FILE = "sample.wav"
MODEL = "whisper-large"

def build_multipart(audio_path, model_name, boundary):
    body = b""
    # model field
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="model"\r\n\r\n'
    body += model_name.encode() + b"\r\n"
    # file field
    body += f"--{boundary}\r\n".encode()
    mime, _ = mimetypes.guess_type(audio_path)
    mime = mime or "audio/wav"
    body += f'Content-Disposition: form-data; name="file"; filename="{audio_path}"\r\n'.encode()
    body += f"Content-Type: {mime}\r\n\r\n".encode()
    with open(audio_path, "rb") as f:
        body += f.read()
    body += b"\r\n" + f"--{boundary}--\r\n".encode()
    return body

body = build_multipart(AUDIO_FILE, MODEL, BOUNDARY)
req = urllib.request.Request(
    URL,
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
)

with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read())
    print(f"Transcription: {result['text']}")
    print(f"Language: {result.get('language', 'unknown')}")
