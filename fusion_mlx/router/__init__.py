"""Router — request routing by modality.

Routes requests to the correct engine based on content type:
text -> LLM, images -> VLM, audio -> AudioEngine, image gen -> ImageGenEngine.
Includes Cloud Router for optional upstream fallback.
"""

from .router import RequestRouter
from .cloud_router import CloudRouter

__all__ = ["RequestRouter", "CloudRouter"]
