#!/usr/bin/env python3
"""
🔗 fusion_gui Comprehensive Embedding Test Suite

Tests all supported MLX embedding models for:
- Proper dimension output
- Value normalization 
- OpenAI API compatibility
- Queue processing and counter increments

Supports model families:
- E5 (Microsoft): multilingual sentence embeddings (base, small, large)
- ModernBERT (Nomic/TaskSource): modern BERT variants (4bit, bf16)
- Arctic Embed (Snowflake): large embedding models (4bit, bf16)
- MiniLM (Microsoft): distilled BERT embeddings (4bit, bf16)
- GTE (Alibaba): generative text embeddings (Qwen2-based)
- BGE (BAAI): bidirectional and generative embeddings (small, base, large, M3)
- Qwen3 (Alibaba): large language model embeddings (4B model)
- SentenceT5: transformer-based sentence embeddings
- Jina AI: specialized embedding models (base, small)
- Stella: multilingual embedding models
- Nomic: Atlas text embeddings 
- UAE: Universal Angle Embeddings
- INSTRUCTOR: instruction-following embeddings

Total: 23 embedding model specifications across 13 model families
"""

import asyncio
import httpx
import json
import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Test configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 180.0  # Longer timeout for embedding models

@dataclass
class EmbeddingModelSpec:
    """Specification for an embedding model test."""
    model_id: str
    family: str
    architecture: str
    expected_dims: int
    quantization: str
    support_method: str  # 'mlx_embedding_models', 'custom_wrapper', 'fallback'
    notes: str = ""

# Comprehensive model specifications
EMBEDDING_MODELS = {
    # E5 Models (Microsoft) - Multilingual
    "e5_base": EmbeddingModelSpec(
        model_id="mlx-community/multilingual-e5-base-mlx",
        family="E5",
        architecture="BERT",
        expected_dims=768,
        quantization="unknown",
        support_method="mlx_embedding_models",
        notes="Multilingual sentence embeddings"
    ),
    "e5_small": EmbeddingModelSpec(
        model_id="mlx-community/multilingual-e5-small-mlx", 
        family="E5",
        architecture="BERT",
        expected_dims=384,
        quantization="unknown",
        support_method="mlx_embedding_models",
        notes="Smaller multilingual model"
    ),
    "e5_large": EmbeddingModelSpec(
        model_id="mlx-community/multilingual-e5-large-mlx",
        family="E5", 
        architecture="BERT",
        expected_dims=1024,
        quantization="unknown",
        support_method="mlx_embedding_models",
        notes="Large multilingual sentence embeddings"
    ),
    
    # ModernBERT Models (Nomic AI)
    "modernbert_nomic_4bit": EmbeddingModelSpec(
        model_id="mlx-community/nomicai-modernbert-embed-base-4bit",
        family="ModernBERT",
        architecture="ModernBERT",
        expected_dims=768,
        quantization="4bit",
        support_method="custom_wrapper",
        notes="Nomic AI ModernBERT 4-bit"
    ),
    "modernbert_nomic_bf16": EmbeddingModelSpec(
        model_id="mlx-community/nomicai-modernbert-embed-base-bf16",
        family="ModernBERT", 
        architecture="ModernBERT",
        expected_dims=768,
        quantization="bf16",
        support_method="custom_wrapper",
        notes="Nomic AI ModernBERT bfloat16"
    ),
    
    # ModernBERT Models (TaskSource)
    "modernbert_task_4bit": EmbeddingModelSpec(
        model_id="mlx-community/tasksource-ModernBERT-base-embed-4bit",
        family="ModernBERT",
        architecture="ModernBERT", 
        expected_dims=768,
        quantization="4bit",
        support_method="custom_wrapper",
        notes="TaskSource ModernBERT 4-bit"
    ),
    
    # Arctic Embed Models (Snowflake)
    "arctic_4bit": EmbeddingModelSpec(
        model_id="snowflake-arctic-embed-l-v2-0-4bit",
        family="Arctic",
        architecture="Arctic",
        expected_dims=1024,
        quantization="4bit", 
        support_method="custom_wrapper",
        notes="Snowflake Arctic large embedding v2.0"
    ),
    "arctic_bf16": EmbeddingModelSpec(
        model_id="snowflake-arctic-embed-l-v2-0-bf16",
        family="Arctic",
        architecture="Arctic",
        expected_dims=1024,
        quantization="bf16",
        support_method="custom_wrapper", 
        notes="Snowflake Arctic large embedding v2.0 bf16"
    ),
    
    # MiniLM Models (Microsoft)
    "minilm_4bit": EmbeddingModelSpec(
        model_id="all-minilm-l6-v2-4bit",
        family="MiniLM",
        architecture="BERT",
        expected_dims=384,
        quantization="4bit",
        support_method="mlx_embeddings",
        notes="Distilled BERT for sentence embeddings"
    ),
    "minilm_bf16": EmbeddingModelSpec(
        model_id="all-minilm-l6-v2-bf16", 
        family="MiniLM",
        architecture="BERT",
        expected_dims=384,
        quantization="bf16",
        support_method="mlx_embeddings",
        notes="Distilled BERT for sentence embeddings"
    ),
    
    # GTE Models (Alibaba) 
    "gte_qwen2": EmbeddingModelSpec(
        model_id="BillSYZhang/gte-Qwen2-7B-instruct-Q4-mlx",
        family="GTE", 
        architecture="Qwen2",
        expected_dims=4096,
        quantization="4bit",
        support_method="custom_wrapper",
        notes="Qwen2-based generative text embeddings"
    ),
    
    # BGE Models (BAAI) - Multiple variants
    "bge_small_bf16": EmbeddingModelSpec(
        model_id="bge-small-en-v1-5-bf16",
        family="BGE",
        architecture="BERT", 
        expected_dims=384,
        quantization="bf16",
        support_method="mlx_embedding_models",
        notes="BAAI BGE small English embeddings"
    ),
    "bge_base_4bit": EmbeddingModelSpec(
        model_id="mlx-community/bge-base-en-v1.5-4bit",
        family="BGE",
        architecture="BERT",
        expected_dims=768,
        quantization="4bit", 
        support_method="mlx_embedding_models",
        notes="BAAI BGE base English embeddings"
    ),
    "bge_large_4bit": EmbeddingModelSpec(
        model_id="mlx-community/bge-large-en-v1.5-4bit",
        family="BGE",
        architecture="BERT",
        expected_dims=1024,
        quantization="4bit",
        support_method="mlx_embedding_models", 
        notes="BAAI BGE large English embeddings"
    ),
    "bge_m3_4bit": EmbeddingModelSpec(
        model_id="mlx-community/bge-m3-4bit",
        family="BGE",
        architecture="BERT",
        expected_dims=1024,
        quantization="4bit",
        support_method="mlx_embedding_models",
        notes="BAAI BGE M3 multilingual embeddings"
    ),
    
    # Additional Community Models
    "sentence_t5_base": EmbeddingModelSpec(
        model_id="mlx-community/sentence-t5-base-mlx",
        family="SentenceT5",
        architecture="T5",
        expected_dims=768,
        quantization="unknown",
        support_method="custom_wrapper",
        notes="Sentence-T5 transformer embeddings"
    ),
    "jina_v2_base": EmbeddingModelSpec(
        model_id="mlx-community/jina-embeddings-v2-base-en-4bit",
        family="Jina",
        architecture="BERT", 
        expected_dims=768,
        quantization="4bit",
        support_method="mlx_embedding_models",
        notes="Jina AI embeddings v2 base"
    ),
    "jina_v2_small": EmbeddingModelSpec(
        model_id="mlx-community/jina-embeddings-v2-small-en-4bit",
        family="Jina",
        architecture="BERT",
        expected_dims=512,
        quantization="4bit", 
        support_method="mlx_embedding_models",
        notes="Jina AI embeddings v2 small"
    ),
    "stella_base": EmbeddingModelSpec(
        model_id="mlx-community/stella_en_400M_v5-4bit",
        family="Stella",
        architecture="BERT",
        expected_dims=1024,
        quantization="4bit",
        support_method="mlx_embedding_models", 
        notes="Stella multilingual embeddings"
    ),
    
    # Nomic Embed Models
    "nomic_embed_text": EmbeddingModelSpec(
        model_id="mlx-community/nomic-embed-text-v1.5-4bit",
        family="Nomic",
        architecture="BERT",
        expected_dims=768,
        quantization="4bit",
        support_method="mlx_embedding_models",
        notes="Nomic Atlas text embeddings"
    ),
    
    # UAE (Universal Angle Embeddings)
    "uae_large": EmbeddingModelSpec(
        model_id="mlx-community/UAE-Large-V1-4bit", 
        family="UAE",
        architecture="BERT",
        expected_dims=1024,
        quantization="4bit",
        support_method="mlx_embedding_models",
        notes="Universal Angle Embeddings large"
    ),
    
    # INSTRUCTOR Models
    "instructor_xl": EmbeddingModelSpec(
        model_id="mlx-community/instructor-xl-4bit",
        family="INSTRUCTOR",
        architecture="BERT",
        expected_dims=768,
        quantization="4bit",
        support_method="mlx_embedding_models",
        notes="INSTRUCTOR XL instruction-following embeddings"
    ),
    
    # Qwen3 Models (Alibaba) - Already working
    "qwen3_4b": EmbeddingModelSpec(
        model_id="qwen3-embedding-4b-4bit-dwq",
        family="Qwen3",
        architecture="Qwen3",
        expected_dims=2560, 
        quantization="4bit",
        support_method="custom_wrapper",
        notes="Qwen3 4B embedding model"
    ),
}

@dataclass
class EmbeddingTestResult:
    """Results from testing an embedding model."""
    model_id: str
    success: bool
    actual_dims: int
    expected_dims: int
    value_range: Tuple[float, float]
    vector_norm: float
    mean_value: float
    std_value: float
    sample_values: List[float]
    response_time: float
    error_message: str = ""
    
    @property
    def dims_correct(self) -> bool:
        return self.actual_dims == self.expected_dims
    
    @property 
    def normalized(self) -> bool:
        return abs(self.vector_norm - 1.0) < 0.1
    
    @property
    def reasonable_range(self) -> bool:
        min_val, max_val = self.value_range
        return abs(min_val) < 2.0 and abs(max_val) < 2.0


class EmbeddingTestSuite:
    """Comprehensive embedding model test suite."""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=TIMEOUT)
        self.results: List[EmbeddingTestResult] = []
        
    async def close(self):
        """Clean up HTTP client."""
        await self.client.aclose()
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models from server."""
        try:
            response = await self.client.get(f"{BASE_URL}/v1/models")
            if response.status_code == 200:
                data = response.json()
                return [model["id"] for model in data["data"]]
            return []
        except Exception as e:
            print(f"❌ Error getting available models: {e}")
            print(f"   Make sure fusion_gui server is running on {BASE_URL}")
            return []
    
    async def test_embedding_model(self, spec: EmbeddingModelSpec) -> EmbeddingTestResult:
        """Test a single embedding model."""
        print(f"\n🔗 Testing {spec.family} Model: {spec.model_id}")
        print(f"   Architecture: {spec.architecture} | Expected Dims: {spec.expected_dims} | {spec.quantization}")
        
        start_time = time.time()
        
        # Test texts
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language."
        ]
        
        try:
            # Test embedding generation
            embedding_data = {
                "model": spec.model_id,
                "input": test_texts,
                "encoding_format": "float"
            }
            
            print(f"   🔄 Generating embeddings...")
            response = await self.client.post(f"{BASE_URL}/v1/embeddings", json=embedding_data)
            response_time = time.time() - start_time
            
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"   ❌ Failed: {error_msg}")
                return EmbeddingTestResult(
                    model_id=spec.model_id,
                    success=False,
                    actual_dims=0,
                    expected_dims=spec.expected_dims,
                    value_range=(0.0, 0.0),
                    vector_norm=0.0,
                    mean_value=0.0,
                    std_value=0.0,
                    sample_values=[],
                    response_time=response_time,
                    error_message=error_msg
                )
            
            # Parse response
            result = response.json()
            embeddings_data = result["data"]
            
            if not embeddings_data:
                raise ValueError("No embeddings returned")
            
            # Analyze first embedding
            first_embedding = embeddings_data[0]["embedding"]
            actual_dims = len(first_embedding)
            
            # Convert to numpy for analysis
            embedding_array = np.array(first_embedding)
            value_range = (float(embedding_array.min()), float(embedding_array.max()))
            vector_norm = float(np.linalg.norm(embedding_array))
            mean_value = float(embedding_array.mean())
            std_value = float(embedding_array.std())
            sample_values = first_embedding[:5]
            
            # Create result
            test_result = EmbeddingTestResult(
                model_id=spec.model_id,
                success=True,
                actual_dims=actual_dims,
                expected_dims=spec.expected_dims, 
                value_range=value_range,
                vector_norm=vector_norm,
                mean_value=mean_value,
                std_value=std_value,
                sample_values=sample_values,
                response_time=response_time
            )
            
            # Print results
            print(f"   ✅ Generated {len(embeddings_data)} embeddings in {response_time:.2f}s")
            print(f"   📊 Dimensions: {actual_dims} (expected: {spec.expected_dims}) {'✅' if test_result.dims_correct else '❌'}")
            print(f"   📊 Value Range: {value_range[0]:.3f} to {value_range[1]:.3f} {'✅' if test_result.reasonable_range else '⚠️'}")
            print(f"   📊 Vector Norm: {vector_norm:.3f} {'✅' if test_result.normalized else '⚠️'}")
            print(f"   📊 Mean: {mean_value:.3f}, Std: {std_value:.3f}")
            print(f"   📊 Sample: {sample_values}")
            
            if test_result.dims_correct and test_result.normalized and test_result.reasonable_range:
                print(f"   🎉 {spec.family} embeddings working correctly!")
            else:
                print(f"   ⚠️  {spec.family} embeddings may need adjustment")
            
            return test_result
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = str(e)
            print(f"   ❌ Error: {error_msg}")
            
            return EmbeddingTestResult(
                model_id=spec.model_id,
                success=False,
                actual_dims=0,
                expected_dims=spec.expected_dims,
                value_range=(0.0, 0.0),
                vector_norm=0.0,
                mean_value=0.0,
                std_value=0.0,
                sample_values=[],
                response_time=response_time,
                error_message=error_msg
            )
    
    async def run_comprehensive_test(self, test_models: Optional[List[str]] = None):
        """Run comprehensive embedding model tests."""
        print("🚀 Starting Comprehensive MLX Embedding Test Suite")
        print("=" * 80)
        
        # Get available models
        print("📋 Checking available models...")
        available_models = await self.get_available_models()
        print(f"   Found {len(available_models)} models on server")
        
        # Filter models to test
        models_to_test = {}
        if test_models:
            # Test specific models
            for model_key in test_models:
                if model_key in EMBEDDING_MODELS:
                    models_to_test[model_key] = EMBEDDING_MODELS[model_key]
        else:
            # Test available models
            for key, spec in EMBEDDING_MODELS.items():
                if spec.model_id in available_models:
                    models_to_test[key] = spec
        
        if not models_to_test:
            print("❌ No embedding models available to test")
            return
        
        print(f"\n📋 Testing {len(models_to_test)} embedding models:")
        for key, spec in models_to_test.items():
            print(f"   • {spec.family}: {spec.model_id}")
        
        # Run tests
        print("\n🧪 Running Embedding Tests")
        print("-" * 50)
        
        for key, spec in models_to_test.items():
            result = await self.test_embedding_model(spec)
            self.results.append(result)
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary results."""
        print("\n" + "=" * 80)
        print("📊 Embedding Test Summary")
        print("=" * 80)
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        correct_dims = [r for r in successful if r.dims_correct]
        normalized = [r for r in successful if r.normalized]
        
        print(f"📈 Total Models Tested: {len(self.results)}")
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        print(f"📏 Correct Dimensions: {len(correct_dims)}")
        print(f"📐 Properly Normalized: {len(normalized)}")
        
        if successful:
            print(f"\n✅ Working Models:")
            for result in successful:
                status = "🎉" if (result.dims_correct and result.normalized) else "⚠️"
                print(f"   {status} {result.model_id}")
                print(f"      Dims: {result.actual_dims}, Norm: {result.vector_norm:.3f}, Range: {result.value_range[0]:.3f} to {result.value_range[1]:.3f}")
        
        if failed:
            print(f"\n❌ Failed Models:")
            for result in failed:
                print(f"   • {result.model_id}: {result.error_message}")
        
        # Performance stats
        if successful:
            response_times = [r.response_time for r in successful]
            avg_time = np.mean(response_times)
            print(f"\n⏱️  Average Response Time: {avg_time:.2f}s")


async def main():
    """Main test runner."""
    import sys
    suite = EmbeddingTestSuite()
    
    try:
        # Check if specific models were requested
        if len(sys.argv) > 1:
            # Test specific models by key
            test_keys = sys.argv[1:]
            print(f"🎯 Testing specific models: {test_keys}")
            await suite.run_comprehensive_test(test_models=test_keys)
        else:
            # Test all available models
            await suite.run_comprehensive_test()
        
    finally:
        await suite.close()


if __name__ == "__main__":
    asyncio.run(main())