from fusion_mlx.api.images import ImageGenerateRequest
from fusion_mlx.audio.registry import resolve_audio_alias
from fusion_mlx.config import DEFAULT_ALIASES
from fusion_mlx.server import resolve_model_id


class TestImageGenerateRequestVariant:
    def test_default_variant_is_none(self):
        req = ImageGenerateRequest(prompt="test")
        assert req.variant is None

    def test_variant_set(self):
        req = ImageGenerateRequest(prompt="test", variant="depth")
        assert req.variant == "depth"

    def test_guidance_default_is_none(self):
        req = ImageGenerateRequest(prompt="test")
        assert req.guidance is None

    def test_guidance_set(self):
        req = ImageGenerateRequest(prompt="test", guidance=4.0)
        assert req.guidance == 4.0

    def test_control_image_optional(self):
        req = ImageGenerateRequest(prompt="test", control_image="/tmp/canny.png")
        assert req.control_image == "/tmp/canny.png"

    def test_controlnet_strength_optional(self):
        req = ImageGenerateRequest(prompt="test", controlnet_strength=0.8)
        assert req.controlnet_strength == 0.8

    def test_reference_images_optional(self):
        req = ImageGenerateRequest(
            prompt="test", reference_images=["/tmp/a.png", "/tmp/b.png"]
        )
        assert len(req.reference_images) == 2

    def test_edit_image_and_mask(self):
        req = ImageGenerateRequest(
            prompt="test", edit_image="/tmp/edit.png", mask_image="/tmp/mask.png"
        )
        assert req.edit_image == "/tmp/edit.png"
        assert req.mask_image == "/tmp/mask.png"

    def test_depth_image_optional(self):
        req = ImageGenerateRequest(prompt="test", depth_image="/tmp/depth.png")
        assert req.depth_image == "/tmp/depth.png"

    def test_image_strength_optional(self):
        req = ImageGenerateRequest(prompt="test", image_strength=0.5)
        assert req.image_strength == 0.5

    def test_invalid_variant_accepted_by_model(self):
        # Variant validation happens in the route handler, not Pydantic
        req = ImageGenerateRequest(prompt="test", variant="nonexistent")
        assert req.variant == "nonexistent"


class TestImageAliasResolution:
    def test_flux2_alias_registered(self):
        assert DEFAULT_ALIASES.get("flux-2") == "flux2-klein-9b-4bit"

    def test_kokoro_not_in_llm_alias_table(self):
        assert DEFAULT_ALIASES.get("kokoro") is None

    def test_resolve_flux2_alias(self):
        assert resolve_model_id("flux-2") == "flux2-klein-9b-4bit"

    def test_resolve_kokoro_via_audio_registry(self):
        entry = resolve_audio_alias("kokoro")
        assert entry is not None
        assert entry.hf_id == "mlx-community/Kokoro-82M-bf16"
        assert resolve_model_id("kokoro") == "kokoro"

    def test_existing_alias_unchanged(self):
        assert resolve_model_id("claude-4.6-sonnet") == "Qwen3.6-27B-mxfp8"

    def test_unknown_model_returned_as_is(self):
        assert resolve_model_id("not-a-real-model") == "not-a-real-model"
