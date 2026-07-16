"""
HuggingFace integration for MLX-GUI.
Handles model discovery, metadata extraction, and compatibility checking.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

from huggingface_hub import HfApi, list_models, model_info, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
import requests

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a HuggingFace model."""
    id: str
    name: str
    author: str
    downloads: int
    likes: int
    created_at: str
    updated_at: str
    model_type: str
    library_name: Optional[str]
    pipeline_tag: Optional[str]
    tags: List[str]
    size_gb: Optional[float]
    mlx_compatible: bool
    has_mlx_version: bool
    mlx_repo_id: Optional[str]
    estimated_memory_gb: Optional[float]
    description: Optional[str]


class HuggingFaceClient:
    """Client for interacting with HuggingFace Hub."""
    
    def __init__(self, token: Optional[str] = None):
        self.api = HfApi(token=token)
        self.token = token
        self.mlx_tags = {
            "mlx",
            "mlx-lm", 
            "apple-silicon",
            "metal",
            "quantized"
        }
        
        # Parameter counts (in billions) for memory calculation
        self.param_patterns = {
            "0.5b": 0.5,
            "1b": 1.0,
            "1.8b": 1.8,
            "2.7b": 2.7,
            "3b": 3.0,
            "4b": 4.0,
            "6b": 6.0,
            "7b": 7.0,
            "8b": 8.0,
            "9b": 9.0,
            "11b": 11.0,
            "13b": 13.0,
            "14b": 14.0,
            "15b": 15.0,
            "20b": 20.0,
            "22b": 22.0,
            "24b": 24.0,
            "27b": 27.0,  # Gemma-3-27B
            "30b": 30.0,
            "32b": 32.0,
            "34b": 34.0,
            "40b": 40.0,
            "65b": 65.0,
            "70b": 70.0,
            "72b": 72.0,
            "110b": 110.0,
            "175b": 175.0,
            "235b": 235.0,  # Qwen3-235B
            "405b": 405.0,
            "424b": 424.0,  # ERNIE 4.5
            # Trillion-scale models
            "1t": 1000.0,
            "1.02t": 1020.0,  # Kimi-K2-Instruct
            "1.5t": 1500.0,
            "2t": 2000.0,
            # Add additional patterns for unconventional naming
            "1.3b": 1.3,
            "2.8b": 2.8,
            "6.7b": 6.7,
            "12b": 12.0,
            "16b": 16.0,
            "18b": 18.0,
            "25b": 25.0,
            "28b": 28.0,
            "33b": 33.0,
            "42b": 42.0,
            "48b": 48.0,
            "52b": 52.0,
            "56b": 56.0,
            "80b": 80.0,
            "120b": 120.0,
            "180b": 180.0,
            # Patterns for models with different suffixes
            "-1b": 1.0,
            "-2b": 2.0,
            "-3b": 3.0,
            "-7b": 7.0,
            "-8b": 8.0,
            "-13b": 13.0,
            "-70b": 70.0,
            # Patterns for models with 'billion' written out
            "1billion": 1.0,
            "2billion": 2.0,
            "3billion": 3.0,
            "7billion": 7.0,
            "8billion": 8.0,
            "13billion": 13.0,
            "70billion": 70.0,
            # Alternative formats
            "1.0b": 1.0,
            "2.0b": 2.0,
            "3.0b": 3.0,
            "7.0b": 7.0,
            "8.0b": 8.0,
            "13.0b": 13.0,
            "70.0b": 70.0,
        }
    
    def search_trending_mlx_models(self, limit: int = 20) -> List[ModelInfo]:
        """Search for trending MLX models from HuggingFace."""
        try:
            models = list_models(
                library="mlx",
                sort="downloads",
                limit=limit,
                direction=-1,
                token=self.token
            )
            
            model_infos = []
            for model in models:
                try:
                    info = self._extract_model_info(model)
                    if info and info.mlx_compatible:
                        model_infos.append(info)
                except Exception as e:
                    logger.warning(f"Error processing trending model {model.id}: {e}")
                    continue
            
            return model_infos
            
        except Exception as e:
            logger.error(f"Error searching trending MLX models: {e}")
            return []

    def search_mlx_models(self, 
                         query: str = "",
                         limit: int = 50,
                         sort: str = "downloads") -> List[ModelInfo]:
        """
        Search for MLX-compatible models on HuggingFace.
        
        Args:
            query: Search query
            limit: Maximum number of results
            sort: Sort by 'downloads', 'likes', 'created', or 'updated'
            
        Returns:
            List of ModelInfo objects
        """
        try:
            # Get models using both library="mlx" and tags=["mlx"] approaches
            models_dict = {}  # Use dict to deduplicate by model ID
            
            # Approach 1: Models with library="mlx" 
            try:
                library_models = list_models(
                    search=query,
                    library="mlx",
                    limit=limit,
                    sort=sort,
                    direction=-1,
                    token=self.token
                )
                for model in library_models:
                    models_dict[model.id] = model
            except Exception as e:
                logger.warning(f"Library filter failed: {e}")
            
            # Approach 2: Models with mlx tag (broader set)
            try:
                tag_models = list_models(
                    search=query,
                    tags=["mlx"],
                    limit=limit,
                    sort=sort,
                    direction=-1,
                    token=self.token
                )
                for model in tag_models:
                    models_dict[model.id] = model
            except Exception as e:
                logger.warning(f"Tag filter failed: {e}")
            
            # Convert back to list and sort by downloads
            models = list(models_dict.values())
            models = sorted(models, key=lambda x: getattr(x, 'downloads', 0), reverse=True)[:limit]
            
            model_infos = []
            for model in models:
                try:
                    info = self._extract_model_info(model)
                    if info:
                        model_infos.append(info)
                except Exception as e:
                    logger.warning(f"Error processing model {model.id}: {e}")
                    continue
            
            return model_infos
            
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                logger.warning(f"Rate limited by HuggingFace: {e}")
                return []
            else:
                logger.error(f"Error searching MLX models: {e}")
                return []
    
    def search_models_by_author(self, author: str, limit: int = 20) -> List[ModelInfo]:
        """Search for models by a specific author."""
        try:
            models = list_models(
                author=author,
                limit=limit,
                sort="downloads",
                direction=-1
            )
            
            model_infos = []
            for model in models:
                try:
                    info = self._extract_model_info(model)
                    if info and info.mlx_compatible:
                        model_infos.append(info)
                except Exception as e:
                    logger.warning(f"Error processing model {model.id}: {e}")
                    continue
            
            return model_infos
            
        except Exception as e:
            logger.error(f"Error searching models by author {author}: {e}")
            return []
    
    def get_popular_mlx_models(self, limit: int = 20) -> List[ModelInfo]:
        """Get popular MLX models by downloads."""
        return self.search_mlx_models(limit=limit, sort="downloads")
    
    def get_recent_mlx_models(self, limit: int = 20) -> List[ModelInfo]:
        """Get recently updated MLX models."""
        return self.search_mlx_models(limit=limit, sort="updated")
    
    def get_model_details(self, model_id: str) -> Optional[ModelInfo]:
        """Get detailed information about a specific model."""
        try:
            model = model_info(model_id)
            return self._extract_model_info(model)
        except (RepositoryNotFoundError, RevisionNotFoundError) as e:
            logger.error(f"Model {model_id} not found: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting model details for {model_id}: {e}")
            return None
    
    def _extract_model_info(self, model) -> Optional[ModelInfo]:
        """Extract ModelInfo from HuggingFace model object."""
        try:
            # Check if model.id is None
            if not hasattr(model, 'id') or model.id is None:
                logger.warning("Model object has no id attribute or id is None")
                return None
            
            # Parse model ID
            parts = model.id.split("/")
            author = parts[0] if len(parts) > 1 else "unknown"
            name = parts[-1]
            
            # Check if model is MLX compatible
            tags = getattr(model, 'tags', None) or []
            library_name = getattr(model, 'library_name', None)
            
            # Model is MLX compatible if:
            # 1. Has MLX tags
            # 2. Has library="mlx" 
            # 3. Is in mlx-community namespace
            # 4. Has explicit MLX in tags
            mlx_compatible = (
                any(tag and tag.lower() in self.mlx_tags for tag in tags if tag is not None) or
                library_name == "mlx" or
                (model.id and "mlx-community/" in model.id) or
                "mlx" in tags
            )
            
            has_mlx_version = "mlx" in tags or library_name == "mlx"
            
            # If not explicitly MLX tagged, check if there's an MLX version
            mlx_repo_id = None
            if not mlx_compatible:
                # Check if there's a corresponding MLX version
                potential_mlx_id = f"mlx-community/{name}"
                try:
                    mlx_model = model_info(potential_mlx_id)
                    if mlx_model:
                        mlx_repo_id = potential_mlx_id
                        has_mlx_version = True
                        mlx_compatible = True
                except:
                    pass
            
            # Estimate model size and memory requirements
            size_gb = self._estimate_model_size(model)
            # Size calculation already includes MLX overhead, use as-is
            estimated_memory_gb = size_gb
            
            # Get model type
            model_type = self._determine_model_type(model, tags)
            
            return ModelInfo(
                id=model.id,
                name=name,
                author=author,
                downloads=getattr(model, 'downloads', 0),
                likes=getattr(model, 'likes', 0),
                created_at=model.created_at.isoformat() if hasattr(model, 'created_at') and model.created_at else None,
                updated_at=model.last_modified.isoformat() if hasattr(model, 'last_modified') and model.last_modified else None,
                model_type=model_type,
                library_name=getattr(model, 'library_name', None),
                pipeline_tag=getattr(model, 'pipeline_tag', None),
                tags=tags,
                size_gb=size_gb,
                mlx_compatible=mlx_compatible,
                has_mlx_version=has_mlx_version,
                mlx_repo_id=mlx_repo_id,
                estimated_memory_gb=estimated_memory_gb,
                description=getattr(model, 'description', None)
            )
            
        except Exception as e:
            logger.error(f"Error extracting model info: {e}")
            return None
    
    def _estimate_model_size(self, model) -> Optional[float]:
        """Calculate model memory requirements from parameter count and quantization."""
        try:
            # Check if model.id is None
            if not hasattr(model, 'id') or model.id is None:
                logger.warning("Model object has no id attribute or id is None in _estimate_model_size")
                return None
            
            model_name = model.id.lower()
            param_count_billions = None
            matched_pattern = None
            
            # First, try to extract parameter count from model card content
            try:
                description = getattr(model, 'description', '') or ''
                
                # Look for parameter patterns in description
                import re
                
                # Patterns to match parameter counts in model cards
                param_patterns = [
                    r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s+param',        # "3.68B params"
                    r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s+parameter',    # "3.68B parameters"
                    r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s+model',        # "3.68B model"
                    r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s+weights',      # "3.68B weights"
                    r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s*-?\s*param',   # "3.68B-param"
                    r'Parameters?:\s*(\d+(?:\.\d+)?)\s*[Bb]',             # "Parameters: 3.68B"
                    r'Model size:\s*(\d+(?:\.\d+)?)\s*[Bb]',             # "Model size: 3.68B"
                    r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s*parameter',    # "3.68Billion parameter"
                ]
                
                for pattern in param_patterns:
                    matches = re.findall(pattern, description, re.IGNORECASE)
                    if matches:
                        param_count_billions = float(matches[0])
                        matched_pattern = f"card-{pattern}"
                        logger.debug(f"Found parameter count in model card for {model.id}: {param_count_billions}B")
                        break
                
            except Exception as e:
                logger.debug(f"Could not parse model card for {model.id}: {e}")
            
            # If not found in description, try safetensors metadata (where HuggingFace stores "3.68B params")
            if not param_count_billions:
                try:
                    model_info_data = self.api.model_info(model.id)
                    if hasattr(model_info_data, 'safetensors') and model_info_data.safetensors:
                        # Look through safetensors metadata for parameter information
                        for file_name, metadata in model_info_data.safetensors.items():
                            # Check for parameter count in metadata
                            if isinstance(metadata, dict):
                                # Sometimes stored as 'total_size' in parameter count
                                if 'total' in metadata:
                                    total_params = metadata.get('total', 0)
                                    if total_params > 1000000:  # Reasonable parameter count
                                        param_count_billions = total_params / 1e9
                                        matched_pattern = f"safetensors-total"
                                        logger.debug(f"Found parameter count in safetensors for {model.id}: {param_count_billions}B")
                                        break
                                
                                # Check metadata strings for patterns like "3.68B"
                                metadata_str = str(metadata)
                                for pattern in param_patterns:
                                    matches = re.findall(pattern, metadata_str, re.IGNORECASE)
                                    if matches:
                                        param_count_billions = float(matches[0])
                                        matched_pattern = f"safetensors-{pattern}"
                                        logger.debug(f"Found parameter count in safetensors metadata for {model.id}: {param_count_billions}B")
                                        break
                                if param_count_billions:
                                    break
                except Exception as e:
                    logger.debug(f"Could not parse safetensors metadata for {model.id}: {e}")
            
            # If not found in metadata, try to extract from model name as final fallback
            if not param_count_billions:
                best_match_length = 0
                
                # Use word boundaries to ensure we match complete parameter specifications
                import re
                
                for pattern, param_count in self.param_patterns.items():
                    # Create regex pattern with word boundaries
                    # Match the pattern only when it's surrounded by non-alphanumeric characters
                    escaped_pattern = re.escape(pattern)
                    regex_pattern = rf'(?<![a-zA-Z0-9]){escaped_pattern}(?![a-zA-Z0-9])'
                    
                    if re.search(regex_pattern, model_name, re.IGNORECASE):
                        if len(pattern) > best_match_length:
                            param_count_billions = param_count
                            best_match_length = len(pattern)
                            matched_pattern = f"name-{pattern}"
            
            if param_count_billions:
                # Determine bits per parameter from quantization
                bits_per_param = 16  # Default FP16
                
                if "4bit" in model_name or "4-bit" in model_name or "qat-4bit" in model_name or "dwq" in model_name:
                    bits_per_param = 4
                elif "6bit" in model_name or "6-bit" in model_name:
                    bits_per_param = 6
                elif "8bit" in model_name or "8-bit" in model_name or "int8" in model_name:
                    bits_per_param = 8
                elif "int4" in model_name:
                    bits_per_param = 4
                elif "bf16" in model_name or "fp16" in model_name:
                    bits_per_param = 16
                elif "fp32" in model_name:
                    bits_per_param = 32
                
                # Calculate base memory: params * bits_per_param / 8 bits_per_byte
                base_memory_gb = (param_count_billions * 1e9 * bits_per_param) / (8 * 1024**3)
                
                # Add MLX overhead (25% for inference, activations, etc.)
                total_memory_gb = base_memory_gb * 1.25
                
                logger.debug(f"Model {model.id}: {param_count_billions}B params (source: {matched_pattern}), {bits_per_param}-bit = {total_memory_gb:.1f}GB")
                return total_memory_gb
            
            # If we can't determine parameter count, don't guess - return None
            logger.debug(f"Could not determine parameter count for {model.id} - no size estimate available")
            return None
            
        except Exception as e:
            logger.debug(f"Error estimating model size for {model.id}: {e}")
            return None
    
    def _determine_model_type(self, model, tags: List[str]) -> str:
        """Determine model type from tags and metadata."""
        tags_lower = [tag.lower() for tag in tags if tag is not None]
        pipeline_tag = (getattr(model, 'pipeline_tag', None) or '').lower()
        
        # Check for embedding models first (by pipeline tag)
        if pipeline_tag in ['feature-extraction', 'sentence-similarity']:
            return 'embedding'
        
        # Check for embedding models by tags
        if any(tag in tags_lower for tag in ['embedding', 'sentence-transformers', 'feature-extraction', 'sentence-similarity']):
            return 'embedding'
        
        # Check for embedding models by name patterns (comprehensive list)
        # Safely handle case where model.id might be None
        if hasattr(model, 'id') and model.id is not None:
            model_name_lower = model.id.lower()
            embedding_patterns = [
                'embedding', 'bge-', 'e5-', 'all-minilm', 'sentence',
                'modernbert-embed', 'arctic-embed', 'gte-', 'embed-base',
                'multilingual-e5', 'nomicai-modernbert', 'tasksource-modernbert',
                'snowflake-arctic', 'qwen3-embedding', 'qwen-embedding'
            ]
            if any(pattern in model_name_lower for pattern in embedding_patterns):
                return 'embedding'
        
        # Check for multimodal capabilities (expanded detection)
        # Priority: Check pipeline tags first for multimodal
        multimodal_pipeline_tags = [
            'image-text-to-text', 'audio-text-to-text', 'video-text-to-text',
            'image-to-text', 'visual-question-answering', 'multimodal'
        ]
        if pipeline_tag in multimodal_pipeline_tags:
            return 'multimodal'
        
        # Check multimodal tags (expanded list)
        multimodal_tags = [
            'multimodal', 'vision', 'image-text', 'vlm', 'visual-language-model',
            'image-text-to-text', 'audio-text-to-text', 'video-text-to-text',
            'image-to-text', 'visual-question-answering', 'vqa'
        ]
        if any(tag in tags_lower for tag in multimodal_tags):
            return 'multimodal'
        
        # Check for models that combine multiple modalities (smart detection)
        has_vision = any(tag in tags_lower for tag in ['vision', 'image', 'visual', 'computer-vision'])
        has_audio = any(tag in tags_lower for tag in ['audio', 'speech', 'automatic-speech-recognition'])
        has_text = any(tag in tags_lower for tag in ['text-generation', 'language-modeling', 'causal-lm'])
        
        # If model has multiple modalities, classify as multimodal
        modality_count = sum([has_vision, has_audio, has_text])
        if modality_count >= 2:
            return 'multimodal'
        
        # Check for vision-only models
        if has_vision or any(tag in tags_lower for tag in ['computer-vision', 'image-classification', 'object-detection']):
            return 'vision'
        
        # Check for audio-only models (only if not already classified as multimodal)
        if has_audio or any(tag in tags_lower for tag in ['text-to-speech', 'tts', 'whisper']):
            return 'audio'
        
        # Check for text generation models
        if pipeline_tag in ['text-generation', 'text2text-generation', 'conversational']:
            return 'text'
        
        # Check tags for text generation
        if any(tag in tags_lower for tag in ['text-generation', 'language-modeling', 'causal-lm']):
            return 'text'
        
        return 'text'  # Default
    
    def get_specific_trending_models(self) -> List[str]:
        """Get list of specific high-priority trending models to add."""
        return [
            "mlx-community/SmolLM3-3B-4bit",
            "mlx-community/SmolLM3-3B-Instruct-4bit",
            "mlx-community/Kimi-Dev-72B-4bit-DWQ",
            "mlx-community/Kimi-K2-Instruct-4bit",
            "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "mlx-community/Gemma-2-9B-it-4bit", 
            "mlx-community/Qwen3-30B-A3B-4bit-DWQ",
            "mlx-community/Qwen3-235B-A22B-4bit-DWQ",
            "mlx-community/Mistral-Small-Instruct-2409-4bit",
            "mlx-community/Codestral-22B-v0.1-4bit",
            "mlx-community/Mistral-Nemo-Instruct-2407-4bit"
        ]
    
    def get_specific_embedding_models(self) -> List[str]:
        """Get list of high-priority MLX embedding models."""
        # Static list of known high-quality models
        static_models = [
            "mlx-community/bge-large-en-v1.5-4bit",
            "mlx-community/bge-base-en-v1.5-4bit", 
            "mlx-community/e5-large-v2-4bit",
            "mlx-community/e5-base-v2-4bit",
            "mlx-community/nomic-embed-text-v1.5-4bit",
            "mlx-community/gte-large-4bit",
            "mlx-community/sentence-transformers-all-MiniLM-L6-v2-4bit",
            "mlx-community/multilingual-e5-large-4bit",
            "mlx-community/text-embedding-ada-002-4bit",
            # Add popular sentence-similarity models
            "mlx-community/multilingual-e5-base-mlx",
            "mlx-community/multilingual-e5-small-mlx",
            "mlx-community/nomicai-modernbert-embed-base-4bit",
            "mlx-community/all-MiniLM-L6-v2-4bit",
            "mlx-community/snowflake-arctic-embed-l-v2.0-4bit"
        ]
        
        # Try to get dynamic list from search, fallback to static
        try:
            dynamic_models = self.search_embedding_models(limit=15)
            if dynamic_models:
                # Combine static and dynamic, prioritizing dynamic
                dynamic_ids = [model.id for model in dynamic_models[:10]]
                combined = dynamic_ids + [model for model in static_models if model not in dynamic_ids]
                return combined[:15]  # Limit to 15 total
        except Exception as e:
            logger.warning(f"Failed to get dynamic embedding models: {e}")
        
        return static_models

    def get_model_categories(self) -> Dict[str, List[str]]:
        """Get categorized lists of popular MLX models."""
        categories = {
            'Trending Models': self.get_specific_trending_models(),
            'Embedding Models': self.get_specific_embedding_models(),
            'Popular Chat': [],
            'Popular TTS': [],
            'Popular STT': [],
            'Vision Models': [],
            'Multimodal Models': [],
            'Code Models': [],
            'Small Models (< 10GB)': [],
            'Large Models (> 50GB)': []
        }
        
        try:
            # Get popular models
            popular_models = self.get_popular_mlx_models(limit=100)
            
            for model in popular_models:
                # Add to appropriate categories
                if model.model_type == 'vision':
                    categories['Vision Models'].append(model.id)
                elif model.model_type == 'multimodal':
                    categories['Multimodal Models'].append(model.id)
                elif model.model_type == 'audio':
                    # Split audio models into TTS vs STT
                    # Safely handle case where model.id might be None
                    if model.id and any(keyword in model.id.lower() for keyword in ['tts', 'text-to-speech', 'speech-synthesis', 'kokoro', 'bark', 'f5-tts']):
                        categories['Popular TTS'].append(model.id)
                    elif model.id and any(keyword in model.id.lower() for keyword in ['whisper', 'stt', 'speech-to-text', 'transcrib', 'asr', 'parakeet', 'tdt']):
                        categories['Popular STT'].append(model.id)
                    else:
                        # Default audio models to STT if unclear
                        categories['Popular STT'].append(model.id)
                
                # Check for code models
                if any(tag in (getattr(model, 'tags', None) or []) for tag in ['code', 'coding', 'programming']):
                    categories['Code Models'].append(model.id)
                
                # Check for chat models (text and multimodal that are conversational)
                if (model.model_type in ['text', 'multimodal'] and 
                    any(tag in (getattr(model, 'tags', None) or []) for tag in ['chat', 'conversational', 'instruct', 'assistant'])):
                    categories['Popular Chat'].append(model.id)
                
                # Size-based categories
                if model.size_gb and model.size_gb < 10:
                    categories['Small Models (< 10GB)'].append(model.id)
                elif model.size_gb and model.size_gb > 50:
                    categories['Large Models (> 50GB)'].append(model.id)
                
                # Add uncategorized text models to Popular Chat if they seem conversational
                if (model.model_type == 'text' and 
                    not any(model.id in cat for cat in categories.values()) and
                    model.id not in categories['Popular Chat']):
                    # Default text models go to Popular Chat
                    categories['Popular Chat'].append(model.id)
            
            # Limit each category to top 10
            for category in categories:
                categories[category] = categories[category][:10]
            
            return categories
            
        except Exception as e:
            logger.error(f"Error getting model categories: {e}")
            return categories
    
    def search_compatible_models(self, query: str, max_memory_gb: float) -> List[ModelInfo]:
        """Search for models compatible with available memory."""
        models = self.search_mlx_models(query, limit=100)
        
        compatible_models = []
        for model in models:
            if model.estimated_memory_gb and model.estimated_memory_gb <= max_memory_gb:
                compatible_models.append(model)
        
        # Sort by popularity (downloads)
        compatible_models.sort(key=lambda x: x.downloads, reverse=True)
        
        return compatible_models[:20]  # Return top 20
    
    def search_audio_models(self, query: str = "", limit: int = 20) -> List[ModelInfo]:
        """Search specifically for MLX audio models (Whisper, TTS, etc.)."""
        # Search for audio-related terms
        audio_queries = []
        if query:
            audio_queries.append(f"{query} audio")
            audio_queries.append(f"{query} speech")
        else:
            audio_queries = ["whisper mlx", "tts mlx", "audio mlx", "speech mlx"]
        
        all_audio_models = []
        
        for audio_query in audio_queries:
            try:
                models = self.search_mlx_models(audio_query, limit=limit)
                for model in models:
                    # Filter for audio models
                    # Safely handle case where model.id might be None
                    model_id_check = model.id and any(keyword in model.id.lower() for keyword in ['whisper', 'tts', 'speech', 'audio', 'parakeet'])
                    if (model.model_type == 'audio' or 
                        any(tag in (getattr(model, 'tags', None) or []) for tag in ['audio', 'speech', 'whisper', 'tts', 'automatic-speech-recognition']) or
                        model_id_check):
                        if model.id not in [m.id for m in all_audio_models]:
                            all_audio_models.append(model)
            except Exception as e:
                logger.warning(f"Error searching for audio models with query '{audio_query}': {e}")
                continue
        
        # Sort by downloads and return top results
        all_audio_models.sort(key=lambda x: x.downloads, reverse=True)
        return all_audio_models[:limit]
    
    def search_tts_models(self, query: str = "", limit: int = 10) -> List[ModelInfo]:
        """Search specifically for Text-to-Speech models."""
        tts_queries = []
        if query:
            tts_queries.append(f"{query} tts")
            tts_queries.append(f"{query} text-to-speech")
        else:
            tts_queries = ["tts mlx", "text-to-speech mlx", "kokoro", "bark mlx", "f5-tts mlx"]
        
        tts_models = []
        for tts_query in tts_queries:
            try:
                models = self.search_mlx_models(tts_query, limit=limit)
                for model in models:
                    # Safely handle case where model.id might be None
                    model_id_check = model.id and any(keyword in model.id.lower() for keyword in ['tts', 'text-to-speech', 'speech-synthesis', 'kokoro', 'bark', 'f5-tts'])
                    if (model_id_check or
                        any(tag in (getattr(model, 'tags', None) or []) for tag in ['text-to-speech', 'tts'])):
                        if model.id not in [m.id for m in tts_models]:
                            tts_models.append(model)
            except Exception as e:
                logger.warning(f"Error searching TTS models with query '{tts_query}': {e}")
                continue
        
        tts_models.sort(key=lambda x: x.downloads, reverse=True)
        return tts_models[:limit]
    
    def search_stt_models(self, query: str = "", limit: int = 10) -> List[ModelInfo]:
        """Search specifically for Speech-to-Text models using HuggingFace pipeline filters."""
        try:
            stt_models = []
            models_dict = {}
            
            # Method 1: Use pipeline_tag=automatic-speech-recognition&library=mlx (exact match to URL)
            try:
                logger.info("Searching for STT models using pipeline_tag=automatic-speech-recognition&library=mlx")
                
                # This matches exactly: https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&library=mlx&sort=trending
                pipeline_models = list_models(
                    pipeline_tag="automatic-speech-recognition",
                    library="mlx",
                    sort="downloads", 
                    limit=50,  # More focused search with library filter
                    direction=-1,
                    token=self.token
                )
                
                # Convert generator to list and count
                pipeline_models_list = list(pipeline_models)
                logger.info(f"Found {len(pipeline_models_list)} MLX STT models with library filter")
                
                for model in pipeline_models_list:
                    try:
                        model_info = self._extract_model_info(model)
                        if model_info and model_info.id not in models_dict:
                            models_dict[model_info.id] = model_info
                            stt_models.append(model_info)
                    except Exception as e:
                        logger.warning(f"Error processing STT model {getattr(model, 'id', 'unknown')}: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Error searching STT models with pipeline filter: {e}")
            
            # Method 2: Fallback to text search for additional models
            stt_queries = []
            if query:
                stt_queries.append(f"{query} whisper")
                stt_queries.append(f"{query} stt")
                stt_queries.append(f"{query} parakeet")
            else:
                stt_queries = ["whisper mlx", "stt mlx", "speech-to-text mlx", "transcription mlx", "parakeet mlx", "parakeet tdt"]
            
            for stt_query in stt_queries:
                try:
                    models = self.search_mlx_models(stt_query, limit=20)
                    for model in models:
                        # Safely handle case where model.id might be None
                        model_id_check = model.id and any(keyword in model.id.lower() for keyword in ['whisper', 'stt', 'speech-to-text', 'transcrib', 'asr', 'parakeet', 'tdt'])
                        if (model_id_check or
                            any(tag in (getattr(model, 'tags', None) or []) for tag in ['automatic-speech-recognition', 'speech-to-text', 'transcription'])):
                            if model.id not in models_dict:
                                models_dict[model.id] = model
                                stt_models.append(model)
                except Exception as e:
                    logger.warning(f"Error searching STT models with query '{stt_query}': {e}")
                    continue
            
            stt_models.sort(key=lambda x: x.downloads, reverse=True)
            return stt_models[:limit]
            
        except Exception as e:
            logger.error(f"Error in search_stt_models: {e}")
            return []
    
    def search_embedding_models(self, query: str = "", limit: int = 20) -> List[ModelInfo]:
        """Search specifically for embedding models using HuggingFace pipeline filters."""
        try:
            embedding_models = []
            models_dict = {}
            
            # Method 1a: Use pipeline_tag=feature-extraction&library=mlx
            try:
                logger.info("Searching for embedding models using pipeline_tag=feature-extraction&library=mlx")
                
                # This matches: https://huggingface.co/models?pipeline_tag=feature-extraction&library=mlx&sort=downloads
                feature_models = list_models(
                    pipeline_tag="feature-extraction",
                    library="mlx",
                    sort="downloads", 
                    limit=50,
                    direction=-1,
                    token=self.token
                )
                
                feature_models_list = list(feature_models)
                logger.info(f"Found {len(feature_models_list)} MLX feature-extraction models")
                
                for model in feature_models_list:
                    models_dict[model.id] = model
                    logger.debug(f"Found MLX feature-extraction model: {model.id}")
                
            except Exception as e:
                logger.warning(f"Error searching feature-extraction models: {e}")
            
            # Method 1b: Use pipeline_tag=sentence-similarity&library=mlx (NEW!)
            try:
                logger.info("Searching for embedding models using pipeline_tag=sentence-similarity&library=mlx")
                
                # This matches: https://huggingface.co/models?pipeline_tag=sentence-similarity&library=mlx&sort=trending
                sentence_models = list_models(
                    pipeline_tag="sentence-similarity",
                    library="mlx",
                    sort="downloads", 
                    limit=50,
                    direction=-1,
                    token=self.token
                )
                
                sentence_models_list = list(sentence_models)
                logger.info(f"Found {len(sentence_models_list)} MLX sentence-similarity models")
                
                for model in sentence_models_list:
                    if model.id not in models_dict:  # Avoid duplicates
                        models_dict[model.id] = model
                        logger.debug(f"Found MLX sentence-similarity model: {model.id}")
                
            except Exception as e:
                logger.warning(f"Error searching sentence-similarity models: {e}")
            
            total_pipeline_models = len(models_dict)
            logger.info(f"Total MLX embedding models from pipeline searches: {total_pipeline_models}")
            
            # Method 1.5: Fallback without library filter for broader coverage
            try:
                logger.info("Fallback: Searching feature-extraction models without library filter")
                
                all_embedding_models = list_models(
                    pipeline_tag="feature-extraction",
                    sort="downloads",
                    limit=100,
                    direction=-1,
                    token=self.token
                )
                
                all_embedding_list = list(all_embedding_models)
                logger.info(f"Found {len(all_embedding_list)} total feature-extraction models")
                
                # Filter for MLX-compatible ones manually
                mlx_fallback_count = 0
                for model in all_embedding_list:
                    if model.id not in models_dict:  # Only add if not already found
                        model_tags = getattr(model, 'tags', []) or []
                        # Safely handle case where model.id might be None
                        model_id_check = model.id and (model.id.startswith('mlx-community/') or 'mlx' in model.id.lower())
                        is_mlx = (any(tag.lower() in self.mlx_tags for tag in model_tags) or 
                                 model_id_check)
                        
                        if is_mlx:
                            models_dict[model.id] = model
                            mlx_fallback_count += 1
                            logger.debug(f"Found additional MLX embedding model: {model.id}")
                
                logger.info(f"Added {mlx_fallback_count} additional MLX embedding models from fallback search")
                
            except Exception as e:
                logger.warning(f"Error in fallback embedding search: {e}")
            
            # Method 2: Search with MLX tag and embedding-related terms
            try:
                embedding_queries = []
                if query:
                    embedding_queries.append(f"{query} embedding")
                    embedding_queries.append(f"{query} sentence")
                else:
                    embedding_queries = ["embedding mlx", "sentence mlx", "embed mlx"]
                
                for embedding_query in embedding_queries:
                    try:
                        tag_models = list_models(
                            tags=["mlx"],
                            search=embedding_query,
                            sort="downloads", 
                            limit=30,
                            direction=-1,
                            token=self.token
                        )
                        
                        # Convert generator to list and count
                        tag_models_list = list(tag_models)
                        logger.info(f"Found {len(tag_models_list)} models with MLX tag and '{embedding_query}' search")
                        
                        for model in tag_models_list:
                            models_dict[model.id] = model
                            logger.debug(f"Found embedding model via tag search: {model.id}")
                            
                    except Exception as e:
                        logger.warning(f"Error searching with query '{embedding_query}': {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Error in embedding tag search: {e}")
            
            # Method 3: Search for known embedding model patterns
            try:
                embedding_patterns = [
                    "bge-", "e5-", "all-MiniLM", "sentence-transformers", 
                    "nomic-embed", "gte-", "intfloat", "Qwen3-Embedding",
                    "multilingual-e5", "text-embedding", "embed-"
                ]
                for pattern in embedding_patterns:
                    try:
                        pattern_models = list_models(
                            search=pattern,
                            sort="downloads", 
                            limit=20,
                            direction=-1,
                            token=self.token
                        )
                        
                        pattern_models_list = list(pattern_models)
                        logger.info(f"Found {len(pattern_models_list)} models with pattern '{pattern}'")
                        
                        for model in pattern_models_list:
                            # Check if model has MLX in tags or is from mlx-community
                            # Safely handle case where model.id might be None
                            model_tags = getattr(model, 'tags', []) or []
                            model_id_check = model.id and (model.id.startswith('mlx-community/') or 'mlx' in model.id.lower())
                            is_mlx = (any(tag.lower() in self.mlx_tags for tag in model_tags) or 
                                     model_id_check)
                            
                            if is_mlx:
                                models_dict[model.id] = model
                                logger.debug(f"Found embedding model via pattern search: {model.id}")
                                
                    except Exception as e:
                        logger.warning(f"Error searching with pattern '{pattern}': {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Error in embedding pattern search: {e}")
            
            # Convert dict to list and extract model info
            models = list(models_dict.values())
            logger.info(f"Total unique embedding models found: {len(models)}")
            
            for model in models:
                try:
                    info = self._extract_model_info(model)
                    if info:
                        # Mark as embedding type
                        info.model_type = 'embedding'
                        embedding_models.append(info)
                except Exception as e:
                    logger.warning(f"Error processing embedding model {model.id}: {e}")
                    continue
            
            # Sort by downloads and return top results
            embedding_models.sort(key=lambda x: x.downloads, reverse=True)
            logger.info(f"Returning {min(len(embedding_models), limit)} embedding models")
            return embedding_models[:limit]
            
        except Exception as e:
            logger.error(f"Error searching embedding models: {e}")
            return []
    
    def search_vision_models(self, query: str = "", limit: int = 10) -> List[ModelInfo]:
        """Search specifically for Vision/Multimodal models using HuggingFace pipeline filters."""
        try:
            vision_models = []
            models_dict = {}
            
            # Method 1: Exact match to HuggingFace URL - pipeline_tag=image-text-to-text&library=mlx&sort=trending
            try:
                logger.info("Searching for vision models using HuggingFace pipeline_tag=image-text-to-text, library=mlx")
                
                # This matches exactly: https://huggingface.co/models?pipeline_tag=image-text-to-text&library=mlx&sort=trending
                # Note: HuggingFace API uses "downloads" for trending, not "trending"
                pipeline_models = list_models(
                    pipeline_tag="image-text-to-text",
                    library="mlx",
                    sort="downloads",
                    limit=50,  # Get more to filter from
                    direction=-1,
                    token=self.token
                )
                
                # Convert generator to list and count
                pipeline_models_list = list(pipeline_models)
                logger.info(f"Found {len(pipeline_models_list)} image-text-to-text models with MLX library")
                
                for model in pipeline_models_list:
                    models_dict[model.id] = model
                    logger.debug(f"Found vision model via pipeline filter: {model.id}")
                
            except Exception as e:
                logger.warning(f"Error using HuggingFace pipeline_tag filter: {e}")
                
            # Method 1.5: Try without library filter in case some models aren't tagged properly
            try:
                logger.info("Searching for image-text-to-text models without library filter")
                
                all_vision_models = list_models(
                    pipeline_tag="image-text-to-text",
                    sort="downloads",
                    limit=100,
                    direction=-1,
                    token=self.token
                )
                
                # Convert generator to list and count
                all_vision_models_list = list(all_vision_models)
                logger.info(f"Found {len(all_vision_models_list)} total image-text-to-text models")
                
                # Filter for MLX-compatible ones manually
                mlx_count = 0
                for model in all_vision_models_list:
                    # Check if model has MLX in tags or is from mlx-community
                    # Safely handle case where model.id might be None
                    model_tags = getattr(model, 'tags', []) or []
                    model_id_check = model.id and (model.id.startswith('mlx-community/') or 'mlx' in model.id.lower())
                    is_mlx = (any(tag.lower() in self.mlx_tags for tag in model_tags) or 
                             model_id_check)
                    
                    if is_mlx:
                        models_dict[model.id] = model
                        mlx_count += 1
                        logger.debug(f"Found MLX vision model: {model.id}")
                
                logger.info(f"Found {mlx_count} MLX-compatible vision models from broader search")
                
            except Exception as e:
                logger.warning(f"Error searching all image-text-to-text models: {e}")
            
            # Method 2: Also search by tags for broader coverage
            try:
                # Search with MLX tag and vision-related terms
                tag_models = list_models(
                    tags=["mlx"],
                    search="llava",
                    sort="downloads", 
                    limit=20,
                    direction=-1,
                    token=self.token
                )
                
                # Convert generator to list and count
                tag_models_list = list(tag_models)
                logger.info(f"Found {len(tag_models_list)} models with MLX tag and 'llava' search")
                
                vision_count = 0
                for model in tag_models_list:
                    # Only include if it looks like a vision model
                    # Safely handle case where model.id might be None
                    model_id_check = model.id and any(keyword in model.id.lower() for keyword in ['llava', 'vision', 'vl', 'multimodal'])
                    if (hasattr(model, 'pipeline_tag') and model.pipeline_tag == 'image-text-to-text') or \
                       model_id_check:
                        if model.id not in models_dict:  # Avoid duplicates
                            models_dict[model.id] = model
                            vision_count += 1
                            logger.debug(f"Found vision model via tag search: {model.id}")
                
                logger.info(f"Added {vision_count} additional vision models from tag search")
                
            except Exception as e:
                logger.warning(f"Error using tag search: {e}")
            
            # Convert to ModelInfo objects
            models = list(models_dict.values())
            logger.info(f"Processing {len(models)} total unique vision models")
            
            for model in models:
                try:
                    info = self._extract_model_info(model)
                    if info and info.mlx_compatible:
                        vision_models.append(info)
                        logger.debug(f"Added vision model: {info.id} (type: {info.model_type})")
                except Exception as e:
                    logger.warning(f"Error processing vision model {model.id}: {e}")
                    continue
            
            # Sort by trending/downloads and return top results
            vision_models.sort(key=lambda x: x.downloads, reverse=True)
            final_models = vision_models[:limit]
            
            logger.info(f"Returning {len(final_models)} vision models")
            for model in final_models:
                logger.info(f"  - {model.id} ({model.downloads} downloads)")
            
            return final_models
            
        except Exception as e:
            logger.error(f"Error searching vision models: {e}")
            return []


# Global HuggingFace client instance
_hf_client: Optional[HuggingFaceClient] = None


def get_huggingface_client(token: Optional[str] = None) -> HuggingFaceClient:
    """Get the global HuggingFace client instance."""
    global _hf_client
    if _hf_client is None or (token and _hf_client.token != token):
        # Check for token in environment if not provided
        if not token:
            import os
            token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
        
        _hf_client = HuggingFaceClient(token=token)
    return _hf_client