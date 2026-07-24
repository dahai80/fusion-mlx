"""PuLID pipeline — identity-preserving image generation via Flux DiT.

Orchestrates: insightface detection -> EVA-CLIP vision encoding ->
IDFormer embedding -> Flux DiT attention injection.

Uses CPU-only insightface (ONNX Runtime) for face detection, avoiding
PyTorch dependency entirely.

Pure MLX port of pulid/pipeline_flux.py.
"""
import logging
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np

from .encoders import IDFormer
from .attention import PerceiverAttentionCA, IDAttnProcessor

logger = logging.getLogger(__name__)


def _crop_face(image, face_box, scale=2.5):
    """Crop and align face region from image."""
    x1, y1, x2, y2 = face_box.astype(int)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale

    img_h, img_w = image.shape[:2]
    x1 = max(0, int(cx - w / 2))
    y1 = max(0, int(cy - h / 2))
    x2 = min(img_w, int(cx + w / 2))
    y2 = min(img_h, int(cy + h / 2))

    crop = image[y1:y2, x1:x2]
    crop = cv2.resize(crop, (336, 336))
    crop = crop.astype(np.float32) / 255.0
    crop = (crop - [0.48145466, 0.4578275, 0.40821073]) / [0.26862954, 0.26130258, 0.27577711]
    return crop.transpose(2, 0, 1)


class PuLIDPipeline:
    """Identity-preserving generation pipeline.

    Injects face identity into Flux DiT during denoising via IDFormer
    embeddings and PerceiverAttentionCA cross-attention hooks.
    """

    def __init__(self, id_former, eva_clip=None, face_app=None, dtype=mx.float16):
        self.id_former = id_former
        self.eva_clip = eva_clip
        self.face_app = face_app
        self.dtype = dtype
        self.attn_processors = {}
        self._id_embedding = None

    @classmethod
    def from_pretrained(cls, model_dir, dtype=mx.float16):
        """Load pipeline from pretrained weights directory.

        Expected structure:
            model_dir/
                id_former/  — IDFormer safetensors
                eva_clip/   — EVA-CLIP safetensors (optional)
        """
        model_dir = Path(model_dir)
        logger.info(f"Loading PuLID pipeline from {model_dir}")

        id_former = IDFormer()
        former_dir = model_dir / "id_former"
        if former_dir.exists():
            weights = mx.load(str(former_dir / "weights.safetensors"))
            id_former.load_weights(list(weights.items()))
        id_former.eval()

        eva_clip = None
        eva_dir = model_dir / "eva_clip"
        if eva_dir.exists():
            try:
                from .eva_clip import EVACLIPEncoder
                eva_clip = EVACLIPEncoder.from_pretrained(str(eva_dir), dtype=dtype)
                logger.info("EVA-CLIP loaded")
            except ImportError:
                logger.warning("EVA-CLIP not available, face image encoding disabled")

        face_app = None
        try:
            import insightface
            face_app = insightface.app.FaceAnalysis(
                name="antelopev2",
                root=str(model_dir / "insightface"),
                providers=["CPUExecutionProvider"],
            )
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("insightface antelopev2 loaded")
        except Exception as e:
            logger.warning(f"insightface not available: {e}")

        pipeline = cls(id_former=id_former, eva_clip=eva_clip, face_app=face_app, dtype=dtype)
        return pipeline

    def extract_id_embedding(self, image):
        """Extract identity embedding from a face image.

        Args:
            image: (H, W, 3) uint8 numpy array (BGR from cv2)
        Returns:
            (1, num_queries, 2048) mx.array ID embedding, or None
        """
        if self.face_app is None:
            logger.error("insightface not loaded, cannot extract ID embedding")
            return None

        faces = self.face_app.get(image)
        if not faces:
            logger.warning("No face detected in image")
            return None

        face = faces[0]
        arcface_emb = face.normed_embedding.reshape(1, -1)
        arcface_emb = mx.array(arcface_emb.astype(np.float32))

        id_cond = arcface_emb

        vit_hidden = []
        if self.eva_clip is not None:
            face_crop = _crop_face(image, face.bbox)
            face_tensor = mx.array(face_crop[np.newaxis]).astype(self.dtype)
            clip_out = self.eva_clip(face_tensor)
            if isinstance(clip_out, (list, tuple)):
                for h in clip_out:
                    vit_hidden.append(h.astype(self.dtype))
                clip_cls = clip_out[-1][:, 0]
                id_cond = mx.concatenate([arcface_emb, clip_cls], axis=-1).astype(self.dtype)
            else:
                for _ in range(5):
                    vit_hidden.append(mx.zeros((1, 1, 1024), dtype=self.dtype))
        else:
            for _ in range(5):
                vit_hidden.append(mx.zeros((1, 1, 1024), dtype=self.dtype))

        while len(vit_hidden) < 5:
            vit_hidden.append(mx.zeros((1, 1, 1024), dtype=self.dtype))

        id_embedding = self.id_former(id_cond, vit_hidden)
        logger.info(f"ID embedding shape: {id_embedding.shape}")
        return id_embedding

    def setup_attn_processors(self, dit_model, double_interval=2, single_interval=4):
        """Install IDAttnProcessor hooks into Flux DiT attention layers.

        Args:
            dit_model: Flux DiT model with .double_blocks and .single_blocks
            double_interval: inject into every N-th double block
            single_interval: inject into every N-th single block
        """
        self.attn_processors = {}

        if hasattr(dit_model, "double_blocks"):
            for idx, block in enumerate(dit_model.double_blocks):
                if idx % double_interval == 0:
                    proc = IDAttnProcessor(
                        dim=3072, dim_head=128, heads=16, kv_dim=2048,
                        ortho_mode="ortho_v2", scale=1.0,
                    )
                    self.attn_processors[f"double_blocks.{idx}"] = proc
                    if hasattr(block, "processor"):
                        block.processor = proc

        if hasattr(dit_model, "single_blocks"):
            for idx, block in enumerate(dit_model.single_blocks):
                if idx % single_interval == 0:
                    proc = IDAttnProcessor(
                        dim=3072, dim_head=128, heads=16, kv_dim=2048,
                        ortho_mode="ortho_v2", scale=1.0,
                    )
                    self.attn_processors[f"single_blocks.{idx}"] = proc
                    if hasattr(block, "processor"):
                        block.processor = proc

        logger.info(f"Installed {len(self.attn_processors)} ID attention processors")

    def inject_id(self, id_embedding):
        """Set ID embedding on all attention processors for current step."""
        self._id_embedding = id_embedding
        for proc in self.attn_processors.values():
            proc.set_id_embedding(id_embedding)

    def clear_id(self):
        """Clear ID embedding from all attention processors."""
        self._id_embedding = None
        for proc in self.attn_processors.values():
            proc.set_id_embedding(None)

    def __call__(self, image, dit_model, **kwargs):
        """Full pipeline: extract ID from image and inject into DiT.

        Args:
            image: (H, W, 3) uint8 numpy BGR image
            dit_model: Flux DiT model
        Returns:
            id_embedding: (1, num_queries, 2048) mx.array or None
        """
        id_embedding = self.extract_id_embedding(image)
        if id_embedding is not None:
            if not self.attn_processors:
                self.setup_attn_processors(dit_model)
            self.inject_id(id_embedding)
        return id_embedding
