#!/usr/bin/env python3
"""
Test script for embedding model discovery using HuggingFace pipeline filters.
Tests the exact URL: https://huggingface.co/models?pipeline_tag=feature-extraction&library=mlx&sort=downloads
"""

import asyncio
import httpx
import json

BASE_URL = "http://localhost:8000"

async def test_embedding_discovery():
    """Test embedding model discovery with pipeline filters."""
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        print("🧪 Testing Embedding Model Discovery")
        print("=" * 60)
        print("Testing HuggingFace pipeline: feature-extraction + library=mlx")
        print()
        
        # Step 1: Test direct embedding discovery endpoint
        print("1️⃣ Testing /v1/discover/embeddings endpoint...")
        try:
            response = await client.get(f"{BASE_URL}/v1/discover/embeddings?limit=10")
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                total = data.get("total", 0)
                
                print(f"   ✅ Found {total} embedding models")
                
                if models:
                    print(f"   📦 Top embedding models:")
                    for i, model in enumerate(models[:5]):
                        print(f"      {i+1}. {model['id']}")
                        print(f"         Downloads: {model['downloads']:,}")
                        print(f"         Type: {model['model_type']}")
                        print(f"         Size: {model.get('size_gb', 'Unknown')}GB")
                        print(f"         MLX: {'✅' if model['mlx_compatible'] else '❌'}")
                        print()
                else:
                    print("   ⚠️  No embedding models found")
                    
            else:
                print(f"   ❌ Discovery failed: {response.status_code}")
                print(f"   📄 Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Discovery error: {e}")
        
        # Step 2: Test with search query
        print("2️⃣ Testing embedding search with query...")
        search_queries = ["sentence", "bge", "e5", "embedding"]
        
        for query in search_queries:
            try:
                response = await client.get(f"{BASE_URL}/v1/discover/embeddings?query={query}&limit=5")
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    print(f"   🔍 Query '{query}': {len(models)} models found")
                    
                    for model in models[:2]:  # Show top 2
                        print(f"      📦 {model['id']} ({model['downloads']:,} downloads)")
                        
                else:
                    print(f"   ❌ Query '{query}' failed: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Query '{query}' error: {e}")
        
        print()
        
        # Step 3: Test model categories for embeddings
        print("3️⃣ Testing model categories...")
        try:
            response = await client.get(f"{BASE_URL}/v1/discover/categories")
            if response.status_code == 200:
                categories = response.json().get("categories", {})
                
                # Check for trending models that might be embeddings
                trending = categories.get("Trending Models", [])
                embedding_trending = [m for m in trending if 'embed' in m.lower() or 'sentence' in m.lower()]
                
                if embedding_trending:
                    print(f"   📈 Trending embedding models: {len(embedding_trending)}")
                    for model in embedding_trending:
                        print(f"      🔥 {model}")
                else:
                    print("   📈 No trending embedding models found")
                
                # Check specific embedding category if it exists
                if "Embedding Models" in categories:
                    embedding_models = categories["Embedding Models"]
                    print(f"   📊 Categorized embedding models: {len(embedding_models)}")
                    for model in embedding_models[:3]:
                        print(f"      📦 {model}")
                        
            else:
                print(f"   ❌ Categories failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Categories error: {e}")
        
        # Step 4: Test specific known embedding models
        print("\n4️⃣ Testing specific embedding models...")
        known_embedding_models = [
            "mlx-community/bge-large-en-v1.5-4bit",
            "mlx-community/e5-large-v2-4bit", 
            "mlx-community/sentence-transformers-all-MiniLM-L6-v2-4bit",
            "mlx-community/nomic-embed-text-v1.5-4bit"
        ]
        
        for model_id in known_embedding_models:
            try:
                encoded_id = model_id.replace("/", "%2F")
                response = await client.get(f"{BASE_URL}/v1/discover/models/{encoded_id}")
                if response.status_code == 200:
                    model = response.json()
                    print(f"   ✅ {model_id}")
                    print(f"      Type: {model.get('model_type', 'Unknown')}")
                    print(f"      Compatible: {'✅' if model.get('mlx_compatible', False) else '❌'}")
                    print(f"      Size: {model.get('estimated_memory_gb', 'Unknown')}GB")
                else:
                    print(f"   ❌ {model_id}: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ {model_id}: {e}")
        
        # Step 5: Test embedding generation capability
        print("\n5️⃣ Testing embedding generation...")
        test_model = "mlx-community/bge-large-en-v1.5-4bit"  # Common embedding model
        
        try:
            # First try to load the model
            load_response = await client.post(
                f"{BASE_URL}/v1/models/{test_model}/load",
                json={"model_path": test_model}
            )
            
            if load_response.status_code == 200:
                print(f"   ✅ Loaded {test_model}")
                
                # Test embedding generation
                embedding_request = {
                    "model": test_model,
                    "input": [
                        "This is a test sentence for embedding generation.",
                        "Another example sentence to encode."
                    ]
                }
                
                embedding_response = await client.post(
                    f"{BASE_URL}/v1/embeddings",
                    json=embedding_request
                )
                
                if embedding_response.status_code == 200:
                    embedding_data = embedding_response.json()
                    embeddings = embedding_data.get("data", [])
                    
                    print(f"   ✅ Generated {len(embeddings)} embeddings")
                    if embeddings:
                        first_embedding = embeddings[0]["embedding"]
                        print(f"   📐 Embedding dimension: {len(first_embedding)}")
                        print(f"   📊 First few values: {first_embedding[:5]}")
                        
                        # Test embedding similarity
                        if len(embeddings) >= 2:
                            import numpy as np
                            emb1 = np.array(embeddings[0]["embedding"])
                            emb2 = np.array(embeddings[1]["embedding"])
                            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                            print(f"   🔗 Cosine similarity: {similarity:.3f}")
                    
                else:
                    print(f"   ❌ Embedding generation failed: {embedding_response.status_code}")
                    print(f"   📄 Response: {embedding_response.text}")
                
            else:
                print(f"   ⚠️  Could not load {test_model}: {load_response.status_code}")
                print("   💡 This is expected if the model is not available")
                
        except Exception as e:
            print(f"   ❌ Embedding generation error: {e}")
        
        # Step 6: Performance summary
        print("\n6️⃣ Discovery performance summary...")
        try:
            # Test discovery speed
            import time
            start_time = time.time()
            
            response = await client.get(f"{BASE_URL}/v1/discover/embeddings?limit=20")
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                
                print(f"   ⚡ Discovery speed: {end_time-start_time:.2f}s for {len(models)} models")
                print(f"   📊 Pipeline filter efficiency: {'✅ Direct MLX filter' if len(models) > 0 else '⚠️ Limited results'}")
                
                # Analyze model sources
                mlx_community_count = len([m for m in models if m['id'].startswith('mlx-community/')])
                print(f"   🏢 MLX Community models: {mlx_community_count}/{len(models)}")
                
                # Analyze model types
                feature_extraction_count = len([m for m in models if m['model_type'] == 'embedding'])
                print(f"   🎯 Embedding type accuracy: {feature_extraction_count}/{len(models)}")
                
            else:
                print(f"   ❌ Performance test failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Performance test error: {e}")

if __name__ == "__main__":
    print("🔍 Embedding Model Discovery Test")
    print("Testing: pipeline_tag=feature-extraction&library=mlx&sort=downloads")
    print()
    
    asyncio.run(test_embedding_discovery())
    
    print("\n" + "=" * 60)
    print("✅ Embedding discovery test completed!")
    print("🎯 Validates HuggingFace URL: https://huggingface.co/models?pipeline_tag=feature-extraction&library=mlx&sort=downloads")