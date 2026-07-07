# Video Input

Fusion-MLX's VLM engine (`VLMBatchedEngine`) accepts video as a first-class
input alongside images, following the OpenAI-compatible `video_url` content
part convention. This document covers how to enable video, the request format,
frame-extraction parameters, the two execution paths, limits, and pitfalls.

## Prerequisites

Video frame extraction depends on OpenCV, which is **not** installed by the
base package. Install the `vision` extra:

```bash
pip install "fusion-mlx[vision]"
```

The `vision` extra pulls `opencv-python>=4.8.0`, `torch>=2.3.0`, and
`torchvision>=0.18.0`. Without it, the frame-extraction fallback path raises
`ImportError` at request time. (The Qwen native video path does not need
OpenCV, but installing the extra is still recommended so the fallback works
for non-Qwen models.)

## Request format

Send video as a `video_url` content part inside a user message. The `url`
field accepts either an `https://` URL or a base64 data URL:

```json
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe what happens in this video."},
      {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}}
    ]
  }]
}
```

Base64 data URL form (useful for local files without a public URL):

```json
{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAAIGZ0eXBpc..."}}
```

Multiple videos and mixed image/video content are supported in a single
message; each `video_url` part is extracted and passed to the model as a
sequence of frames.

## Frame-extraction parameters

Two optional request fields control how many frames are sampled:

| Field              | Type   | Default | Description                                            |
|--------------------|--------|---------|--------------------------------------------------------|
| `video_fps`        | float  | `2.0`   | Target sampling rate (frames per second).              |
| `video_max_frames` | int    | `128`   | Hard cap on sampled frames.                            |

The frame count is computed by `smart_nframes()`, which aligns to
`FRAME_FACTOR = 2` (required by Qwen-VL's vision encoder) and clamps to
`[MIN_FRAMES = 4, MAX_FRAMES = 128]`. Concretely, for a video of duration
`D` seconds at the requested `fps`:

```
nframes = clamp(ceil(D * fps, factor=2), min=4, max=128)
```

Override either field per request:

```bash
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "video_fps": 1.0,
    "video_max_frames": 32,
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Summarize this clip."},
        {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}}
      ]
    }]
  }'
```

## Two execution paths

Fusion-MLX selects the path per model, not per request:

### 1. Qwen native video fast path

For Qwen-family VL models whose config exposes `video_token_id` or
`video_token_index`, the engine uses `mlx_vlm.video_generate.load_video`
directly. This lets the model's native video tokenizer handle temporal
compression rather than treating frames as independent images.

Detection: `_is_native_video_model()` in `engines/vlm.py` checks
`config.video_token_id` / `config.video_token_index`.

### 2. Frame-extraction fallback (universal)

For all other VL models (e.g. Llama-3.2-Vision, Pixtral), the engine falls
back to OpenCV-based frame extraction:

1. `process_video_input()` resolves the source — URL download, base64 decode,
   or local path — to a temporary file.
2. `extract_video_frames_smart()` opens the file with `cv2.VideoCapture`,
   samples `nframes` frames via `np.linspace` over the frame index range, and
   returns them as a list of `np.ndarray`.
3. `save_frames_to_temp()` encodes each frame to JPEG via Pillow and writes
   to a temp file managed by `TempFileManager` (cleaned up at process exit
   via `atexit`).

The extracted frames are then fed to the model exactly like image inputs.

## Size limits

| Source         | Constant                  | Limit   |
|----------------|---------------------------|---------|
| URL download   | `MAX_VIDEO_SIZE`          | 500 MB  |
| Base64 data URL| `MAX_BASE64_VIDEO_LENGTH` | 700 MB  |

URLs are HEAD-checked before download; exceeding the limit raises
`FileSizeExceededError` and the request fails fast rather than downloading
hundreds of megabytes first. Downloads stream to a temp file with a 120s
timeout.

## Python example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11435/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens in this video?"},
            {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}},
        ],
    }],
    video_fps=1.0,
    video_max_frames=32,
)
print(resp.choices[0].message.content)
```

> The OpenAI Python SDK accepts arbitrary extra fields, so `video_fps` and
> `video_max_frames` pass through to the server even though they are not part
> of the upstream spec.

## Pitfalls

- **`[vision]` extra required.** A bare `pip install fusion-mlx` does not
  install OpenCV. Video requests against non-Qwen models will raise
  `ImportError: opencv-python is required`. Install `fusion-mlx[vision]`.
- **Native path is Qwen-only.** Only models exposing `video_token_id` /
  `video_token_index` take the native path. Other models always use
  frame-extraction, which is slower and loses temporal modeling.
- **Large videos hit temp disk.** URL downloads and base64 decodes write to
  temp files under the OS temp dir. `TempFileManager` cleans these at exit,
  but a long-running server processing many large videos should ensure
  adequate temp disk space.
- **Frame count is factor-2 aligned.** `smart_nframes` rounds up to a
  multiple of 2. Requesting `video_max_frames=31` still yields up to 32
  frames (clamped then aligned). Set `video_fps` lower to reduce the actual
  count for long videos.
- **Videos count toward context.** Each extracted frame consumes image
  tokens; 128 frames on a 7B VL model can dominate the context window.
  Prefer low `video_fps` (1.0) and a tight `video_max_frames` for long clips.
