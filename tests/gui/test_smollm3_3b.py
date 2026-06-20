#!/usr/bin/env python3
"""
Test script for SmolLM3-3B-4bit model.
Tests both streaming and non-streaming inference.
"""

import asyncio
import httpx
import json
import time

BASE_URL = "http://localhost:8000"
MODEL_ID = "mlx-community/SmolLM3-3B-4bit"

async def test_smollm3_model():
    """Test SmolLM3-3B-4bit model with various prompts."""
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        print(f"🧪 Testing {MODEL_ID}\n")
        
        # Step 1: Check model discovery
        print("1️⃣ Checking model discovery...")
        try:
            encoded_model_id = MODEL_ID.replace("/", "%2F")
            response = await client.get(f"{BASE_URL}/v1/discover/models?query=SmolLM3")
            if response.status_code == 200:
                models = response.json()["models"]
                smollm_models = [m for m in models if "SmolLM3" in m["id"]]
                if smollm_models:
                    print(f"   ✅ Found {len(smollm_models)} SmolLM3 models")
                    for model in smollm_models[:3]:
                        print(f"   📦 {model['id']} - {model.get('size_gb', 'Unknown')}GB")
                else:
                    print("   ⚠️  No SmolLM3 models found in discovery")
            else:
                print(f"   ❌ Discovery failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Discovery error: {e}")
        
        # Step 2: Install model first
        print("\n2️⃣ Installing model...")
        try:
            install_response = await client.post(
                f"{BASE_URL}/v1/models/install",
                json={
                    "model_id": MODEL_ID,
                    "name": "smollm3-3b-4bit"
                }
            )
            if install_response.status_code == 200:
                install_data = install_response.json()
                print("   ✅ Model installed successfully")
                print(f"   📊 Status: {install_data.get('status', 'unknown')}")
                print(f"   💾 Memory: {install_data.get('estimated_memory_gb', 'Unknown')}GB")
            else:
                print(f"   ❌ Install failed: {install_response.status_code}")
                print(f"   📄 Response: {install_response.text}")
                return
        except Exception as e:
            print(f"   ❌ Install error: {e}")
            return
        
        # Step 3: Load model
        print("\n3️⃣ Loading model...")
        try:
            load_response = await client.post(
                f"{BASE_URL}/v1/models/smollm3-3b-4bit/load"
            )
            if load_response.status_code == 200:
                load_data = load_response.json()
                print("   ✅ Model loaded successfully")
                print(f"   📊 Status: {load_data.get('status', 'unknown')}")
                if load_data.get('memory_warning'):
                    print(f"   ⚠️  Warning: {load_data['memory_warning']}")
            else:
                print(f"   ❌ Load failed: {load_response.status_code}")
                print(f"   📄 Response: {load_response.text}")
                return
        except Exception as e:
            print(f"   ❌ Load error: {e}")
            return
        
        # Step 4: Test multilingual capabilities (SmolLM3 supports 8 languages)
        test_prompts = [
            {
                "language": "English",
                "prompt": "Explain quantum computing in simple terms."
            },
            {
                "language": "French", 
                "prompt": "Expliquez l'informatique quantique en termes simples."
            },
            {
                "language": "Spanish",
                "prompt": "Explica la computación cuántica en términos simples."
            },
            {
                "language": "German",
                "prompt": "Erkläre Quantencomputing in einfachen Worten."
            }
        ]
        
        # Step 5: Test non-streaming inference  
        print("\n4️⃣ Testing non-streaming inference...")
        for i, test in enumerate(test_prompts[:2]):  # Test first 2 languages
            try:
                chat_request = {
                    "model": "smollm3-3b-4bit",
                    "messages": [
                        {"role": "user", "content": test["prompt"]}
                    ],
                    "max_tokens": 200,
                    "temperature": 0.7,
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
                    print(f"   ✅ {test['language']} response ({end_time-start_time:.1f}s):")
                    print(f"   💬 {content[:100]}...")
                    print(f"   📊 Tokens: {result.get('usage', {})}")
                else:
                    print(f"   ❌ {test['language']} failed: {response.status_code}")
                    print(f"   📄 {response.text}")
                
            except Exception as e:
                print(f"   ❌ {test['language']} error: {e}")
        
        # Step 6: Test streaming inference
        print("\n5️⃣ Testing streaming inference...")
        try:
            stream_request = {
                "model": "smollm3-3b-4bit",
                "messages": [
                    {"role": "user", "content": "Write a short story about a robot learning to paint."}
                ],
                "max_tokens": 300,
                "temperature": 0.8,
                "stream": True
            }
            
            print("   🔄 Streaming response:")
            print("   💬 ", end="", flush=True)
            
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
                                        print(delta["content"], end="", flush=True)
                            except json.JSONDecodeError:
                                continue
                    print("\n   ✅ Streaming completed successfully")
                else:
                    print(f"   ❌ Streaming failed: {response.status_code}")
                    
        except Exception as e:
            print(f"   ❌ Streaming error: {e}")
        
        # Step 7: Test model capabilities
        print("\n6️⃣ Testing model capabilities...")
        capability_tests = [
            "What are your capabilities?",
            "Can you code in Python?",
            "Do you understand multiple languages?",
            "What is your training data cutoff?"
        ]
        
        for question in capability_tests:
            try:
                capability_request = {
                    "model": "smollm3-3b-4bit",
                    "messages": [{"role": "user", "content": question}],
                    "max_tokens": 100,
                    "temperature": 0.3,
                    "stream": False
                }
                
                response = await client.post(
                    f"{BASE_URL}/v1/chat/completions",
                    json=capability_request
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    print(f"   ❓ {question}")
                    print(f"   💭 {content[:80]}...")
                    print()
                
            except Exception as e:
                print(f"   ❌ Capability test error: {e}")
        
        # Step 8: Health check
        print("7️⃣ Model health check...")
        try:
            health_response = await client.get(f"{BASE_URL}/v1/models/smollm3-3b-4bit/health")
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"   ✅ Model health: {health_data}")
            else:
                print(f"   ⚠️  Health check status: {health_response.status_code}")
        except Exception as e:
            print(f"   ❌ Health check error: {e}")

if __name__ == "__main__":
    print("🤖 SmolLM3-3B-4bit Model Test")
    print("=" * 50)
    print("Testing multilingual 3B parameter model with 4-bit quantization")
    print("Features: 8 languages, 481M parameters, Apache-2.0 license")
    print()
    
    asyncio.run(test_smollm3_model())
    
    print("\n" + "=" * 50)
    print("✅ SmolLM3-3B-4bit test completed!")