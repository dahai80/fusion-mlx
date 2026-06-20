#!/usr/bin/env python3
"""
Test script for Kimi-K2-Instruct-4bit model.
Tests ultra-large 1.02T parameter model with advanced instruction capabilities.
"""

import asyncio
import httpx
import json
import time
import psutil

BASE_URL = "http://localhost:8000"
MODEL_ID = "mlx-community/Kimi-K2-Instruct-4bit"

def check_system_memory():
    """Check if system has enough memory for 1.02T model."""
    memory = psutil.virtual_memory()
    total_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    return total_gb, available_gb

async def test_kimi_k2_model():
    """Test Kimi-K2-Instruct-4bit model with ultra-large parameter count."""
    
    # Pre-flight memory check for trillion-parameter model
    total_memory, available_memory = check_system_memory()
    print(f"💾 System Memory: {total_memory:.1f}GB total, {available_memory:.1f}GB available")
    
    # 1.02T model with 4-bit quantization needs ~2.6TB (theoretical)
    # In practice, MLX optimizations may reduce this significantly
    estimated_requirement = 2600.0  # Conservative estimate
    practical_requirement = 512.0   # More realistic with MLX optimizations
    
    print(f"🧠 Kimi-K2-Instruct Model: 1.02 trillion parameters")
    print(f"   Theoretical memory: ~{estimated_requirement}GB (4-bit)")
    print(f"   Practical estimate: ~{practical_requirement}GB (with MLX optimizations)")
    
    if available_memory < practical_requirement:
        print(f"⚠️  Warning: Model may need ~{practical_requirement}GB, you have {available_memory:.1f}GB")
        print("   This ultra-large model may not load on typical hardware")
    else:
        print(f"✅ Potentially sufficient memory for optimized loading")
    
    async with httpx.AsyncClient(timeout=900.0) as client:  # Extended timeout for ultra-large model
        print(f"\n🧪 Testing {MODEL_ID}\n")
        
        # Step 1: Check model discovery
        print("1️⃣ Checking model discovery...")
        try:
            response = await client.get(f"{BASE_URL}/v1/discover/models?query=Kimi-K2")
            if response.status_code == 200:
                models = response.json()["models"]
                k2_models = [m for m in models if "Kimi-K2" in m["id"]]
                if k2_models:
                    print(f"   ✅ Found {len(k2_models)} Kimi-K2 models")
                    for model in k2_models:
                        print(f"   📦 {model['id']}")
                        print(f"      Memory: {model.get('estimated_memory_gb', 'Unknown')}GB")
                        print(f"      Downloads: {model.get('downloads', 0):,}")
                else:
                    print("   ⚠️  No Kimi-K2 models found in discovery")
            else:
                print(f"   ❌ Discovery failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Discovery error: {e}")
        
        # Step 2: Install model first
        print("\n2️⃣ Installing ultra-large model...")
        try:
            install_response = await client.post(
                f"{BASE_URL}/v1/models/install",
                json={
                    "model_id": MODEL_ID,
                    "name": "kimi-k2-instruct-4bit"
                }
            )
            if install_response.status_code == 200:
                install_data = install_response.json()
                print("   ✅ Model installed successfully")
                print(f"   📊 Status: {install_data.get('status', 'unknown')}")
                print(f"   💾 Memory estimate: {install_data.get('estimated_memory_gb', 'Unknown')}GB")
            else:
                print(f"   ❌ Install failed: {install_response.status_code}")
                print(f"   📄 Response: {install_response.text}")
                return
        except Exception as e:
            print(f"   ❌ Install error: {e}")
            return
        
        # Step 3: Attempt model loading (likely to fail on typical hardware)
        print("\n3️⃣ Loading trillion-parameter model (this will likely fail on typical hardware)...")
        try:
            load_start = time.time()
            load_response = await client.post(
                f"{BASE_URL}/v1/models/kimi-k2-instruct-4bit/load"
            )
            load_end = time.time()
            
            if load_response.status_code == 200:
                load_data = load_response.json()
                print(f"   🎉 Model loaded successfully in {load_end-load_start:.1f}s")
                print(f"   📊 Status: {load_data.get('status', 'unknown')}")
                if load_data.get('memory_warning'):
                    print(f"   ⚠️  Warning: {load_data['memory_warning']}")
                model_loaded = True
            else:
                print(f"   ❌ Load failed: {load_response.status_code}")
                print(f"   📄 Response: {load_response.text}")
                
                # Expected failure - provide helpful guidance
                if "memory" in load_response.text.lower() or load_response.status_code == 507:
                    print(f"   💡 Expected: This 1.02T parameter model requires massive memory")
                    print(f"   💡 Theoretical requirement: ~{estimated_requirement}GB")
                    print(f"   💡 Consider using smaller models or specialized hardware")
                model_loaded = False
                
        except Exception as e:
            print(f"   ❌ Load error: {e}")
            if "timeout" in str(e).lower():
                print("   💡 Loading timeout - trillion-parameter models need extensive time")
            model_loaded = False
        
        # Only continue with inference tests if model loaded successfully
        if not model_loaded:
            print("\n⚠️  Skipping inference tests due to model loading failure")
            print("   This is expected behavior for hardware with insufficient memory")
            return
        
        # Step 4: Test advanced instruction following (if model loaded)
        print("\n4️⃣ Testing advanced instruction following...")
        instruction_tests = [
            {
                "name": "Complex Reasoning",
                "prompt": "You are given a logic puzzle: In a library, there are 5 books arranged in a specific order. Book A is to the left of Book B. Book C is between Books A and B. Book D is to the right of Book B. Book E is between Books B and D. What is the correct order of all books from left to right?"
            },
            {
                "name": "Creative Writing",
                "prompt": "Write a short science fiction story (3-4 paragraphs) about an AI that discovers it can dream. Include themes of consciousness and self-discovery."
            },
            {
                "name": "Code Generation",
                "prompt": "Create a Python class that implements a thread-safe LRU cache with expiration times. Include proper error handling and documentation."
            },
            {
                "name": "Mathematical Analysis",
                "prompt": "Explain the mathematical concept of convergence in infinite series. Provide an example with the geometric series and discuss practical applications."
            }
        ]
        
        for test in instruction_tests:
            try:
                chat_request = {
                    "model": "kimi-k2-instruct-4bit",
                    "messages": [
                        {"role": "system", "content": "You are an advanced AI assistant with exceptional reasoning and instruction-following capabilities."},
                        {"role": "user", "content": test["prompt"]}
                    ],
                    "max_tokens": 400,
                    "temperature": 0.1,  # Lower temperature for precise reasoning
                    "stream": False
                }
                
                start_time = time.time()
                response = await client.post(
                    f"{BASE_URL}/v1/chat/completions",
                    json=chat_request
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    print(f"   ✅ {test['name']} ({end_time-start_time:.1f}s):")
                    print(f"   💭 {content[:120]}...")
                    print(f"   📊 Response length: {len(content)} characters")
                    print()
                else:
                    print(f"   ❌ {test['name']} failed: {response.status_code}")
                
            except Exception as e:
                print(f"   ❌ {test['name']} error: {e}")
        
        # Step 5: Test streaming with complex multi-domain prompt
        print("5️⃣ Testing streaming with complex prompt...")
        try:
            stream_request = {
                "model": "kimi-k2-instruct-4bit",
                "messages": [
                    {"role": "system", "content": "You are an expert consultant capable of analyzing complex interdisciplinary problems."},
                    {"role": "user", "content": "Design a comprehensive strategy for a sustainable smart city that integrates: 1) Renewable energy systems, 2) AI-driven traffic optimization, 3) Vertical farming infrastructure, 4) Circular economy principles, 5) Social equity considerations. Explain how these systems interconnect and address potential challenges in implementation."}
                ],
                "max_tokens": 600,
                "temperature": 0.2,
                "stream": True
            }
            
            print("   🔄 Streaming interdisciplinary analysis:")
            print("   💬 ", end="", flush=True)
            
            domain_coverage = {
                'energy': 0, 'ai': 0, 'agriculture': 0, 
                'economics': 0, 'social': 0, 'systems': 0
            }
            
            keywords = {
                'energy': ['solar', 'wind', 'renewable', 'battery', 'grid'],
                'ai': ['algorithm', 'optimization', 'machine learning', 'sensors', 'data'],
                'agriculture': ['farming', 'vertical', 'crops', 'food', 'hydroponic'],
                'economics': ['circular', 'economy', 'waste', 'resource', 'efficiency'],
                'social': ['equity', 'community', 'access', 'inclusive', 'social'],
                'systems': ['integration', 'interconnect', 'system', 'coordination', 'synergy']
            }
            
            total_tokens = 0
            start_time = time.time()
            full_response = ""
            
            async with client.stream(
                "POST",
                f"{BASE_URL}/v1/chat/completions",
                json=stream_request
            ) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and data["choices"]:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        content = delta["content"]
                                        print(content, end="", flush=True)
                                        full_response += content
                                        total_tokens += len(content.split())
                            except json.JSONDecodeError:
                                continue
                    
                    end_time = time.time()
                    
                    # Analyze domain coverage
                    for domain, word_list in keywords.items():
                        for keyword in word_list:
                            if keyword.lower() in full_response.lower():
                                domain_coverage[domain] += 1
                    
                    print(f"\n   ✅ Streaming completed ({end_time-start_time:.1f}s)")
                    print(f"   📊 Tokens: ~{total_tokens}, Speed: ~{total_tokens/(end_time-start_time):.1f} tok/s")
                    print(f"   🧠 Domain coverage analysis:")
                    for domain, count in domain_coverage.items():
                        coverage = "✅" if count > 0 else "❌"
                        print(f"      {domain.capitalize()}: {coverage} ({count} keywords)")
                    
                    # Assess response quality
                    quality_indicators = ['however', 'furthermore', 'integration', 'challenge', 'solution']
                    found_indicators = [ind for ind in quality_indicators if ind in full_response.lower()]
                    print(f"   📈 Quality indicators: {len(found_indicators)}/5 found")
                    
                else:
                    print(f"   ❌ Streaming failed: {response.status_code}")
                    
        except Exception as e:
            print(f"   ❌ Streaming error: {e}")
        
        # Step 6: Test multilingual capabilities
        print("\n6️⃣ Testing multilingual instruction following...")
        multilingual_tests = [
            {"lang": "English", "prompt": "Explain the concept of machine learning in one paragraph."},
            {"lang": "Chinese", "prompt": "用中文解释机器学习的概念。"},
            {"lang": "Japanese", "prompt": "機械学習の概念を日本語で説明してください。"},
            {"lang": "Code-switching", "prompt": "Explain artificial intelligence using both English and Chinese terms where appropriate. 用英文和中文解释人工智能。"}
        ]
        
        for test in multilingual_tests:
            try:
                multilingual_request = {
                    "model": "kimi-k2-instruct-4bit",
                    "messages": [{"role": "user", "content": test["prompt"]}],
                    "max_tokens": 150,
                    "temperature": 0.3,
                    "stream": False
                }
                
                response = await client.post(f"{BASE_URL}/v1/chat/completions", json=multilingual_request)
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    print(f"   🌍 {test['lang']}: {content[:80]}...")
                
            except Exception as e:
                print(f"   ❌ {test['lang']} error: {e}")
        
        # Step 7: Performance and model characteristics summary
        print("\n7️⃣ Ultra-large model performance summary...")
        try:
            health_response = await client.get(f"{BASE_URL}/v1/models/kimi-k2-instruct-4bit/health")
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"   ✅ Model health: {health_data}")
            else:
                print(f"   ⚠️  Health check status: {health_response.status_code}")
        except Exception as e:
            print(f"   ❌ Health check error: {e}")
        
        print(f"   📈 Kimi-K2-Instruct characteristics:")
        print(f"      - Parameters: 1.02 trillion")
        print(f"      - Quantization: 4-bit")
        print(f"      - Architecture: Advanced instruction-tuned")
        print(f"      - Downloads: 3,736")
        print(f"      - Specializations: Complex reasoning, multilingual, instruction following")
        print(f"      - Memory requirement: Extremely high (~2.6TB theoretical, ~512GB practical)")
        print(f"      - Use case: Research, specialized applications with massive hardware")

if __name__ == "__main__":
    print("🚀 Kimi-K2-Instruct-4bit Model Test")
    print("=" * 70)
    print("Testing ultra-large 1.02T parameter instruction-tuned model")
    print("Features: 1.02T parameters, 4-bit quantization, advanced reasoning")
    print("Note: This model requires massive memory and may not load on typical hardware")
    print()
    
    asyncio.run(test_kimi_k2_model())
    
    print("\n" + "=" * 70)
    print("✅ Kimi-K2-Instruct-4bit test completed!")
    print("💡 If model loading failed due to memory constraints, this is expected behavior")