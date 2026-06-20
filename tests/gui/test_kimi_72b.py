#!/usr/bin/env python3
"""
Test script for Kimi-Dev-72B-4bit-DWQ model.
Tests both streaming and non-streaming inference for large model.
"""

import asyncio
import httpx
import json
import time
import psutil

BASE_URL = "http://localhost:8000"
MODEL_ID = "mlx-community/Kimi-Dev-72B-4bit-DWQ"

def check_system_memory():
    """Check if system has enough memory for 72B model."""
    memory = psutil.virtual_memory()
    total_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    return total_gb, available_gb

async def test_kimi_72b_model():
    """Test Kimi-Dev-72B-4bit-DWQ model with memory monitoring."""
    
    # Pre-flight memory check
    total_memory, available_memory = check_system_memory()
    print(f"💾 System Memory: {total_memory:.1f}GB total, {available_memory:.1f}GB available")
    
    # 72B model with 4-bit quantization needs ~45GB
    estimated_requirement = 45.0
    if available_memory < estimated_requirement:
        print(f"⚠️  Warning: Model needs ~{estimated_requirement}GB, you have {available_memory:.1f}GB available")
        print("   Model loading may fail or cause system instability")
    else:
        print(f"✅ Sufficient memory for 72B model ({estimated_requirement}GB estimated)")
    
    async with httpx.AsyncClient(timeout=600.0) as client:  # Extended timeout for large model
        print(f"\n🧪 Testing {MODEL_ID}\n")
        
        # Step 1: Check model discovery
        print("1️⃣ Checking model discovery...")
        try:
            response = await client.get(f"{BASE_URL}/v1/discover/models?query=Kimi")
            if response.status_code == 200:
                models = response.json()["models"]
                kimi_models = [m for m in models if "Kimi" in m["id"]]
                if kimi_models:
                    print(f"   ✅ Found {len(kimi_models)} Kimi models")
                    for model in kimi_models:
                        print(f"   📦 {model['id']} - {model.get('estimated_memory_gb', 'Unknown')}GB")
                else:
                    print("   ⚠️  No Kimi models found in discovery")
            else:
                print(f"   ❌ Discovery failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Discovery error: {e}")
        
        # Step 2: Memory estimation check
        print("\n2️⃣ Memory estimation check...")
        try:
            encoded_model_id = MODEL_ID.replace("/", "%2F")
            response = await client.get(f"{BASE_URL}/v1/discover/models/{encoded_model_id}")
            if response.status_code == 200:
                model_info = response.json()
                estimated_memory = model_info.get('estimated_memory_gb', 'Unknown')
                print(f"   📊 Estimated memory: {estimated_memory}GB")
                print(f"   🔧 MLX Compatible: {model_info.get('mlx_compatible', False)}")
            else:
                print(f"   ⚠️  Memory estimation unavailable: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Memory check error: {e}")
        
        # Step 3: Install model first
        print("\n3️⃣ Installing model...")
        try:
            install_response = await client.post(
                f"{BASE_URL}/v1/models/install",
                json={
                    "model_id": MODEL_ID,
                    "name": "kimi-dev-72b-4bit-dwq"
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
        
        # Step 4: Attempt model loading (with warning)
        print("\n4️⃣ Loading large model (this may take several minutes)...")
        try:
            load_start = time.time()
            load_response = await client.post(
                f"{BASE_URL}/v1/models/kimi-dev-72b-4bit-dwq/load"
            )
            load_end = time.time()
            
            if load_response.status_code == 200:
                load_data = load_response.json()
                print(f"   ✅ Model loaded successfully in {load_end-load_start:.1f}s")
                print(f"   📊 Status: {load_data.get('status', 'unknown')}")
                if load_data.get('memory_warning'):
                    print(f"   ⚠️  Warning: {load_data['memory_warning']}")
            else:
                print(f"   ❌ Load failed: {load_response.status_code}")
                print(f"   📄 Response: {load_response.text}")
                
                # If loading fails due to memory, show helpful message
                if "memory" in load_response.text.lower() or load_response.status_code == 507:
                    print(f"   💡 Tip: This 72B model requires ~{estimated_requirement}GB RAM")
                    print("   💡 Consider using a smaller model or system with more memory")
                return
        except Exception as e:
            print(f"   ❌ Load error: {e}")
            if "timeout" in str(e).lower():
                print("   💡 Loading timeout - large models need more time")
            return
        
        # Step 5: Test advanced reasoning (72B model strength)
        print("\n5️⃣ Testing advanced reasoning capabilities...")
        reasoning_prompts = [
            {
                "name": "Mathematical Reasoning",
                "prompt": "If a train travels 120 km in 2 hours, and then 180 km in 3 hours, what is the average speed for the entire journey? Show your work step by step."
            },
            {
                "name": "Logic Puzzle",
                "prompt": "There are 5 houses in a row. The red house is to the left of the blue house. The green house is to the right of the blue house. The yellow house is between the red and blue houses. Where is the white house?"
            },
            {
                "name": "Code Analysis",
                "prompt": "Analyze this Python code and identify potential issues:\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"
            }
        ]
        
        for test in reasoning_prompts:
            try:
                chat_request = {
                    "model": "kimi-dev-72b-4bit-dwq",
                    "messages": [
                        {"role": "user", "content": test["prompt"]}
                    ],
                    "max_tokens": 300,
                    "temperature": 0.1,  # Lower temperature for reasoning
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
                    print(f"   💭 {content[:150]}...")
                    if len(content) > 150:
                        print(f"   📝 [Response truncated - full length: {len(content)} chars]")
                    print()
                else:
                    print(f"   ❌ {test['name']} failed: {response.status_code}")
                
            except Exception as e:
                print(f"   ❌ {test['name']} error: {e}")
        
        # Step 6: Test streaming with complex prompt
        print("6️⃣ Testing streaming inference...")
        try:
            stream_request = {
                "model": "kimi-dev-72b-4bit-dwq",
                "messages": [
                    {"role": "user", "content": "Write a detailed technical explanation of how transformer attention mechanisms work, including the mathematical foundations. Make it suitable for a computer science graduate student."}
                ],
                "max_tokens": 500,
                "temperature": 0.3,
                "stream": True
            }
            
            print("   🔄 Streaming technical explanation:")
            print("   💬 ", end="", flush=True)
            
            token_count = 0
            stream_start = time.time()
            
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
                                        token_count += len(content.split())
                            except json.JSONDecodeError:
                                continue
                    
                    stream_end = time.time()
                    print(f"\n   ✅ Streaming completed ({stream_end-stream_start:.1f}s)")
                    print(f"   📊 Approximate tokens: {token_count}")
                    if token_count > 0:
                        tokens_per_sec = token_count / (stream_end - stream_start)
                        print(f"   🚀 Speed: ~{tokens_per_sec:.1f} tokens/second")
                else:
                    print(f"   ❌ Streaming failed: {response.status_code}")
                    
        except Exception as e:
            print(f"   ❌ Streaming error: {e}")
        
        # Step 7: Performance monitoring
        print("\n7️⃣ Performance monitoring...")
        current_memory = psutil.virtual_memory()
        memory_used = (total_memory * 1024**3 - current_memory.available) / (1024**3)
        print(f"   💾 Current memory usage: {memory_used:.1f}GB / {total_memory:.1f}GB")
        print(f"   📈 Memory increase: {memory_used - (total_memory - available_memory):.1f}GB")
        
        # Step 8: Health check
        print("\n8️⃣ Model health check...")
        try:
            health_response = await client.get(f"{BASE_URL}/v1/models/kimi-dev-72b-4bit-dwq/health")
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"   ✅ Model health: {health_data}")
            else:
                print(f"   ⚠️  Health check status: {health_response.status_code}")
        except Exception as e:
            print(f"   ❌ Health check error: {e}")

if __name__ == "__main__":
    print("🤖 Kimi-Dev-72B-4bit-DWQ Model Test")
    print("=" * 60)
    print("Testing large language model with 72B parameters and DWQ quantization")
    print("Features: Advanced reasoning, 4-bit quantization, ~45GB memory requirement")
    print()
    
    asyncio.run(test_kimi_72b_model())
    
    print("\n" + "=" * 60)
    print("✅ Kimi-Dev-72B-4bit-DWQ test completed!")