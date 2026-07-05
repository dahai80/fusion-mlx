"""ComfyUI integration — routes image generation through fusion-mlx as custom backend."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

from .base import Integration

DEFAULT_COMFYUI_PORT = 8188


class ComfyUIIntegration(Integration):
    """ComfyUI integration that registers fusion-mlx as an image generation backend.

    Strategy:
    1. Write a custom ComfyUI node (`fusion_mlx_loader.py`) into ComfyUI's
       `custom_nodes/` directory. The node calls fusion-mlx's /v1/images/generate
       endpoint instead of running local diffusion.
    2. Generate a default workflow JSON that uses the custom node.
    3. Launch ComfyUI with the right flags.
    """

    CUSTOM_NODE_TEMPLATE = '''\
"""ComfyUI custom node — Image generation via fusion-mlx backend.

Drop this node into any ComfyUI workflow to offload image generation to a
local fusion-mlx server running Flux 2 on Apple Silicon.
"""

import json
import urllib.request


class FusionMlxImageGenerate:
    """Generate images using fusion-mlx as the backend."""

    @classmethod
    def define(cls, prompt) -> dict:
        """Return node definition for the ComfyUI frontend."""
        return {
            "input": {
                "required": {
                    "prompt": ("STRING", {"default": "", "multiline": True}),
                    "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                    "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 50}),
                    "guidance": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                    "n_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                },
            },
            "output": ["IMAGE"],
            "name": "FusionMLXImageGen",
            "display_name": "Fusion MLX Image Generate",
            "description": "Generate images via fusion-mlx Flux 2 backend",
            "category": "image",
            "documentation": "https://github.com/dahai80/fusion-mlx",
        }

    @classmethod
    def execute(
        cls,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        guidance: float = 7.5,
        seed: int = 0,
        n_images: int = 1,
    ):
        import numpy as np
        from PIL import Image

        mlx_url = os.environ.get("FUSION_MLX_URL", "http://127.0.0.1:8000")
        url = f"{mlx_url}/v1/images/generate"

        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
            "seed": seed if seed > 0 else None,
            "n": n_images,
            "response_format": "b64_json",
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"fusion-mlx image generation failed: {e.code} {e.read().decode()}")

        images = []
        for item in result.get("data", []):
            b64 = item.get("b64_json", "")
            if not b64:
                continue
            import base64
            img = Image.open(__import__("io").BytesIO(base64.b64decode(b64)))
            array = np.array(img).astype(np.float32) / 255.0
            # ComfyUI expects shape (H, W, C) with C=3 or 4
            if array.shape[2] == 4:
                array = array[:, :, :3]  # RGBA -> RGB
            images.append(array)

        if not images:
            raise RuntimeError("No images returned from fusion-mlx")

        return {"result": images}


NODE_CLASS_NAME = "FusionMlxImageGenerate"
NODE_DISPLAY_NAME = "Fusion MLX Image Generate"
'''

    def __init__(self):
        super().__init__(
            name="comfyui",
            display_name="ComfyUI",
            type="config_file",
            install_check="comfyui",
            install_hint="pip install comfyui  or  git clone https://github.com/comfyanonymous/ComfyUI",
        )
        self._comfyui_dir = self._find_comfyui_dir()
        self._config_path = Path.home() / ".fusion-mlx" / "comfyui.json"

    def _find_comfyui_dir(self) -> Path | None:
        """Find the ComfyUI installation directory."""
        # Check common locations
        candidates = [
            Path.home() / "ComfyUI",
            Path("/opt/ComfyUI"),
        ]
        for p in candidates:
            if (p / "main.py").exists():
                return p
        return None

    def get_command(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> str:
        return (
            f"FUSION_MLX_URL=http://{host}:{port} "
            f"comfyui --listen 127.0.0.1 --port {DEFAULT_COMFYUI_PORT}"
        )

    def configure(
        self,
        port: int,
        api_key: str,
        model: str,
        host: str = "127.0.0.1",
    ) -> None:
        """Install the custom node and generate default workflow."""
        if not self._comfyui_dir:
            print(
                "Warning: ComfyUI directory not found. Installing node to ~/.fusion-mlx/comfyui-node/"
            )
            node_dir = Path.home() / ".fusion-mlx" / "comfyui-node"
            node_dir.mkdir(parents=True, exist_ok=True)
        else:
            node_dir = self._comfyui_dir / "custom_nodes" / "fusion_mlx_node"
            node_dir.mkdir(parents=True, exist_ok=True)

        # Write custom node
        node_file = node_dir / "fusion_mlx_loader.py"
        node_file.write_text(self.CUSTOM_NODE_TEMPLATE, encoding="utf-8")
        print(f"Custom node installed: {node_file}")

        # Write config
        config = {
            "fusion_mlx_url": f"http://{host}:{port}",
            "image_model": model or "flux-2",
            "comfyui_port": DEFAULT_COMFYUI_PORT,
        }
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Config written: {self._config_path}")

        # Generate default workflow
        self._generate_workflow(node_dir.parent)

    def _generate_workflow(self, target_dir: Path) -> None:
        """Generate a ComfyUI workflow JSON using the fusion-mlx custom node."""
        workflow = {
            "last_node_id": 3,
            "last_link_id": 2,
            "nodes": [
                {
                    "id": 1,
                    "type": "FusionMlxImageGenerate",
                    "pos": [80, 200],
                    "size": [315, 270],
                    "flags": {},
                    "order": 0,
                    "mode": 0,
                    "inputs": [],
                    "outputs": [
                        {
                            "name": "IMAGE",
                            "type": "IMAGE",
                            "links": [1],
                            "slot_index": 0,
                        }
                    ],
                    "properties": {
                        "Node name for S&R": "FusionMlxImageGenerate",
                    },
                    "widgets_values": [
                        "A beautiful sunset over mountains, photorealistic, 4K",
                        1024,
                        1024,
                        20,
                        7.5,
                        0,
                        1,
                    ],
                },
                {
                    "id": 2,
                    "type": "SaveImage",
                    "pos": [500, 200],
                    "size": [315, 270],
                    "flags": {},
                    "order": 1,
                    "mode": 0,
                    "inputs": [
                        {
                            "name": "images",
                            "type": "IMAGE",
                            "link": 1,
                        }
                    ],
                    "outputs": [],
                    "properties": {
                        "Node name for S&R": "SaveImage",
                    },
                    "widgets_values": "fusion_mlx_output",
                },
            ],
            "links": [
                [1, 1, 0, 2, 0, "IMAGE"],
            ],
            "groups": [],
            "version": 0.4,
        }

        wf_path = target_dir / "fusion_mlx_default_workflow.json"
        wf_path.write_text(
            json.dumps(workflow, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Default workflow saved: {wf_path}")
        print(
            f"  Load in ComfyUI: drag-drop {wf_path.name} "
            f"or use Manage -> Load Workflow"
        )

    def _wait_for_port(self, host: str, port: int, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=0.5):
                    return True
            except OSError:
                time.sleep(0.25)
        return False

    def launch(
        self,
        port: int,
        api_key: str,
        model: str,
        host: str = "127.0.0.1",
        **kwargs,
    ) -> None:
        """Configure and launch ComfyUI with fusion-mlx as image backend."""
        self.configure(port, api_key, model, host=host)

        if not self._comfyui_dir:
            print("\nComfyUI not found at standard locations.")
            print("To run manually:")
            print(f"  FUSION_MLX_URL=http://{host}:{port} ")
            print(f"  comfyui --listen 127.0.0.1 --port {DEFAULT_COMFYUI_PORT}")
            print("\nOr clone: git clone https://github.com/comfyanonymous/ComfyUI")
            return

        env = self._scrubbed_env()
        env["FUSION_MLX_URL"] = f"http://{host}:{port}"

        print(f"Launching ComfyUI from {self._comfyui_dir}...")
        print(f"  Image backend: fusion-mlx at http://{host}:{port}")
        print(f"  Model: {model or 'flux-2'}")

        proc = subprocess.Popen(
            [
                sys.executable,
                str(self._comfyui_dir / "main.py"),
                "--listen",
                "127.0.0.1",
                "--port",
                str(DEFAULT_COMFYUI_PORT),
                "--custom-nodes-dir",
                str(self._comfyui_dir / "custom_nodes"),
            ],
            env=env,
        )

        if self._wait_for_port("127.0.0.1", DEFAULT_COMFYUI_PORT):
            print(f"ComfyUI running at http://127.0.0.1:{DEFAULT_COMFYUI_PORT}")
        else:
            print("Warning: ComfyUI did not start within 30s")

        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
