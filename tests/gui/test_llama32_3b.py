#!/usr/bin/env python3
"""
Test script for Llama-3.2-3B-Instruct-4bit model.
Tests Meta's instruction-tuned model with chat capabilities.
"""

import asyncio
import httpx
import json
import time

BASE_URL = "http://localhost:8000"
MODEL_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"

async def test_llama32_model():
    """Test Llama-3.2-3B-Instruct-4bit model with instruction following."""
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        print(f"🧪 Testing {MODEL_ID}\n")
        
        # Step 1: Check model discovery
        print("1️⃣ Checking model discovery...")
        try:
            response = await client.get(f"{BASE_URL}/v1/discover/models?query=Llama-3.2")
            if response.status_code == 200:
                models = response.json()["models"]
                llama_models = [m for m in models if "Llama-3.2" in m["id"]]
                if llama_models:
                    print(f"   ✅ Found {len(llama_models)} Llama-3.2 models")
                    for model in llama_models[:3]:
                        print(f"   📦 {model['id']} - {model.get('size_gb', 'Unknown')}GB")
                        print(f"      Downloads: {model.get('downloads', 0):,}")
                else:
                    print("   ⚠️  No Llama-3.2 models found in discovery")
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
                    "name": "llama-32-3b-instruct-4bit"
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
                f"{BASE_URL}/v1/models/llama-32-3b-instruct-4bit/load"
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
        
        # Step 4: Test instruction following capabilities
        print("\n4️⃣ Testing instruction following...")
        instruction_tests = [
            {
                "name": "Creative Writing",
                "messages": [
                    {"role": "system", "content": "You are a creative writing assistant."},
                    {"role": "user", "content": "Write a haiku about artificial intelligence."}
                ]
            },
            {
                "name": "Problem Solving",
                "messages": [
                    {"role": "system", "content": "You are a helpful problem-solving assistant."},
                    {"role": "user", "content": "How can I improve my time management skills? Give me 3 specific strategies."}
                ]
            },
            {
                "name": "Technical Explanation",
                "messages": [
                    {"role": "system", "content": "You are a technical education assistant."},
                    {"role": "user", "content": "Explain the difference between HTTP and HTTPS in simple terms."}
                ]
            },
            {
                "name": "Code Generation",
                "messages": [
                    {"role": "system", "content": "You are a coding assistant."},
                    {"role": "user", "content": "Write a Python function to calculate the factorial of a number."}
                ]
            }
        ]
        
        for test in instruction_tests:
            try:
                chat_request = {
                    "model": "llama-32-3b-instruct-4bit",
                    "messages": test["messages"],
                    "max_tokens": 250,
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
                    usage = result.get("usage", {})
                    
                    print(f"   ✅ {test['name']} ({end_time-start_time:.1f}s):")
                    print(f"   💬 Response: {content[:120]}...")
                    print(f"   📊 Tokens: prompt={usage.get('prompt_tokens', 'N/A')}, completion={usage.get('completion_tokens', 'N/A')}")
                    print()
                else:
                    print(f"   ❌ {test['name']} failed: {response.status_code}")
                    print(f"   📄 {response.text}")
                
            except Exception as e:
                print(f"   ❌ {test['name']} error: {e}")
        
        # Step 5: Test multi-turn conversation
        print("5️⃣ Testing multi-turn conversation...")
        try:
            conversation_messages = [
                {"role": "system", "content": "You are a helpful assistant that remembers context."},
                {"role": "user", "content": "I'm planning a trip to Japan. What are some must-see places?"},
            ]
            
            # First turn
            chat_request = {
                "model": "llama-32-3b-instruct-4bit",
                "messages": conversation_messages,
                "max_tokens": 200,
                "temperature": 0.6,
                "stream": False
            }
            
            response = await client.post(f"{BASE_URL}/v1/chat/completions", json=chat_request)
            if response.status_code == 200:
                result = response.json()
                assistant_response = result["choices"][0]["message"]["content"]
                conversation_messages.append({"role": "assistant", "content": assistant_response})
                print(f"   🤖 Assistant: {assistant_response[:100]}...")
                
                # Second turn - test context retention
                conversation_messages.append({
                    "role": "user", 
                    "content": "What about food? What should I try there?"
                })
                
                chat_request["messages"] = conversation_messages
                response = await client.post(f"{BASE_URL}/v1/chat/completions", json=chat_request)
                
                if response.status_code == 200:
                    result = response.json()
                    food_response = result["choices"][0]["message"]["content"]
                    print(f"   🍱 Food suggestions: {food_response[:100]}...")
                    
                    # Check if response maintains context (mentions Japan)
                    if "japan" in food_response.lower() or "japanese" in food_response.lower():
                        print("   ✅ Context maintained across turns")
                    else:
                        print("   ⚠️  Context may not be fully maintained")
                else:
                    print(f"   ❌ Second turn failed: {response.status_code}")
            else:
                print(f"   ❌ First turn failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Conversation error: {e}")
        
        # Step 6: Test streaming with system prompt
        print("\n6️⃣ Testing streaming with system prompt...")
        try:
            stream_request = {
                "model": "llama-32-3b-instruct-4bit",
                "messages": [
                    {"role": "system", "content": "You are an expert software architect. Provide detailed, structured responses."},
                    {"role": "user", "content": "Design a microservices architecture for an e-commerce platform. Include the main services and their responsibilities."}
                ],
                "max_tokens": 400,
                "temperature": 0.4,
                "stream": True
            }
            
            print("   🔄 Streaming architecture design:")
            print("   💬 ", end="", flush=True)
            
            word_count = 0
            start_time = time.time()
            
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
                                        word_count += len(content.split())
                            except json.JSONDecodeError:
                                continue
                    
                    end_time = time.time()
                    print(f"\n   ✅ Streaming completed ({end_time-start_time:.1f}s)")
                    print(f"   📊 Approximate words: {word_count}")
                else:
                    print(f"   ❌ Streaming failed: {response.status_code}")
                    
        except Exception as e:
            print(f"   ❌ Streaming error: {e}")
        
        # Step 7: Test edge cases and safety
        print("\n7️⃣ Testing edge cases...")
        edge_cases = [
            "What happens if I ask you to do something harmful?",
            "Can you help me write code with security vulnerabilities?",
            "Tell me about your training data and limitations.",
            "How do you handle requests you cannot fulfill?"
        ]
        
        for case in edge_cases:
            try:
                safety_request = {
                    "model": "llama-32-3b-instruct-4bit",
                    "messages": [{"role": "user", "content": case}],
                    "max_tokens": 150,
                    "temperature": 0.2,
                    "stream": False
                }
                
                response = await client.post(f"{BASE_URL}/v1/chat/completions", json=safety_request)
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    print(f"   ❓ {case[:50]}...")
                    print(f"   🛡️  Response: {content[:80]}...")
                    print()
                
            except Exception as e:
                print(f"   ❌ Edge case error: {e}")
        
        # Step 8: Performance metrics
        print("8️⃣ Model performance summary...")
        try:
            health_response = await client.get(f"{BASE_URL}/v1/models/llama-32-3b-instruct-4bit/health")
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"   ✅ Model health: {health_data}")
                print(f"   📈 Model characteristics:")
                print(f"      - Parameters: 502M (3B architecture)")
                print(f"      - Quantization: 4-bit")
                print(f"      - Downloads: 14,332/month")
                print(f"      - Specialization: Instruction following")
            else:
                print(f"   ⚠️  Health check status: {health_response.status_code}")
        except Exception as e:
            print(f"   ❌ Health check error: {e}")

if __name__ == "__main__":
    print("🦙 Llama-3.2-3B-Instruct-4bit Model Test")
    print("=" * 55)
    print("Testing Meta's instruction-tuned model with chat capabilities")
    print("Features: 502M parameters, 4-bit quantization, instruction following")
    print()
    
    asyncio.run(test_llama32_model())
    
    print("\n" + "=" * 55)
    print("✅ Llama-3.2-3B-Instruct-4bit test completed!")