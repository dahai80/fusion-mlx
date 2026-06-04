"""ComfyUI workflow example — generate images via fusion-mlx backend.

Prerequisites:
1. fusion-mlx server running with a Flux model loaded
2. ComfyUI custom node installed via: fusion-mlx launch comfyui

This script demonstrates calling the ComfyUI API to trigger image
generation through the fusion-mlx custom node.
"""
import json
import time
import urllib.request

COMFYUI_URL = "http://127.0.0.1:8188"


def submit_workflow(prompt: str, width: int = 1024, height: int = 1024) -> str:
     """Submit an image generation workflow to ComfyUI.

    Returns the workflow ID (prompt ID).
     """
    workflow = {
        "3": {
             "class_type": "SaveImage",
             "inputs": {"images": ["4", 0]},
         },
         "4": {
             "class_type": "FusionMlxImageGenerate",
             "inputs": {
                 "prompt": prompt,
                 "width": width,
                 "height": height,
                 "steps": 20,
                 "guidance": 7.5,
                 "seed": 0,
                 "n_images": 1,
             },
         },
     }

    payload = {"prompt": workflow}
    req = urllib.request.Request(
         f"{COMFYUI_URL}/prompt",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
     )

    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    print(f"Workflow submitted, ID: {result['prompt_id']}")
    return result["prompt_id"]


def get_history(prompt_id: str) -> dict:
     """Get workflow output from ComfyUI history."""
    req = urllib.request.Request(
         f"{COMFYUI_URL}/history/{prompt_id}"
     )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def wait_for_completion(prompt_id: str, timeout: int = 600) -> dict:
     """Poll until the workflow completes."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        history = get_history(prompt_id)
        job = history.get(prompt_id, {})
        outputs = job.get("outputs", {})
        if outputs:
            return outputs
        time.sleep(2)
    raise TimeoutError(f"Workflow {prompt_id} did not complete within {timeout}s")


if __name__ == "__main__":
    prompt_id = submit_workflow("A serene mountain lake at golden hour, photorealistic")
    outputs = wait_for_completion(prompt_id)

     # Show output image paths
    for node_id, node_output in outputs.items():
        images = node_output.get("images", [])
        for img in images:
            url = f"{COMFYUI_URL}/view?filename={img['filename']}&subfolder={img.get('subfolder', '')}"
            print(f"Image: {url}")
