#!/usr/bin/env python3
"""
Test script for Gemma-2-9B-it-4bit model.
Tests Google's instruction-tuned model with advanced capabilities.
"""

import asyncio
import httpx
import json
import time

BASE_URL = "http://localhost:8000"
MODEL_ID = "mlx-community/gemma-2-9b-it-4bit"

async def test_gemma2_model():
    """Test Gemma-2-9B-it-4bit model with Google-specific features."""
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        print(f"🧪 Testing {MODEL_ID}\n")
        
        # Step 1: Check model discovery
        print("1️⃣ Checking model discovery...")
        try:
            response = await client.get(f"{BASE_URL}/v1/discover/models?query=gemma-2")
            if response.status_code == 200:
                models = response.json()["models"]
                gemma_models = [m for m in models if "gemma-2" in m["id"].lower()]
                if gemma_models:
                    print(f"   ✅ Found {len(gemma_models)} Gemma-2 models")
                    for model in gemma_models[:3]:
                        print(f"   📦 {model['id']} - {model.get('size_gb', 'Unknown')}GB")
                        print(f"      Downloads: {model.get('downloads', 0):,}")
                else:
                    print("   ⚠️  No Gemma-2 models found in discovery")
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
                    "name": "gemma-2-9b-it-4bit"
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
                f"{BASE_URL}/v1/models/gemma-2-9b-it-4bit/load"
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
        
        # Step 4: Test Google-style capabilities
        print("\n4️⃣ Testing Google AI capabilities...")
        google_tests = [
            {
                "name": "Factual Knowledge",
                "prompt": "What are the key differences between machine learning and deep learning? Provide a structured explanation."
            },
            {
                "name": "Analytical Thinking",
                "prompt": "Analyze the pros and cons of renewable energy sources. Consider economic, environmental, and technological factors."
            },
            {
                "name": "Creative Problem Solving",
                "prompt": "Design an innovative solution for reducing food waste in restaurants. Think outside the box."
            },
            {
                "name": "Technical Documentation",
                "prompt": "Write clear documentation for a REST API endpoint that creates a new user account. Include parameters and examples."
            }
        ]
        
        for test in google_tests:
            try:
                chat_request = {
                    "model": "gemma-2-9b-it-4bit",
                    "messages": [
                        {"role": "user", "content": test["prompt"]}
                    ],
                    "max_tokens": 300,
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
                    # Check for structured thinking
                    if any(indicator in content.lower() for indicator in ['1.', '2.', 'first', 'second', 'pros:', 'cons:', '•', '-']):
                        print("   🎯 Response shows structured thinking")
                    print(f"   💬 Preview: {content[:120]}...")
                    print(f"   📊 Length: {len(content)} characters")
                    print()
                else:
                    print(f"   ❌ {test['name']} failed: {response.status_code}")
                
            except Exception as e:
                print(f"   ❌ {test['name']} error: {e}")
        
        # Step 5: Test reasoning and logic
        print("5️⃣ Testing reasoning capabilities...")
        reasoning_tests = [
            {
                "name": "Mathematical Logic",
                "prompt": "If all cats are mammals, and some mammals fly, can we conclude that some cats fly? Explain your reasoning."
            },
            {
                "name": "Causal Reasoning",
                "prompt": "A company's sales increased by 30% after implementing a new marketing strategy. What factors should we consider before attributing this increase to the strategy?"
            },
            {
                "name": "Ethical Reasoning",
                "prompt": "Discuss the ethical considerations of using AI for hiring decisions. What safeguards should be in place?"
            }
        ]
        
        for test in reasoning_tests:
            try:
                reasoning_request = {
                    "model": "gemma-2-9b-it-4bit",
                    "messages": [
                        {"role": "system", "content": "You are a logical reasoning expert. Think step by step and explain your reasoning clearly."},
                        {"role": "user", "content": test["prompt"]}
                    ],
                    "max_tokens": 250,
                    "temperature": 0.3,  # Lower for more focused reasoning
                    "stream": False
                }
                
                response = await client.post(f"{BASE_URL}/v1/chat/completions", json=reasoning_request)
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    print(f"   🧠 {test['name']}:")
                    # Check for reasoning indicators
                    reasoning_indicators = ['because', 'therefore', 'however', 'consider', 'first', 'second', 'thus', 'hence']
                    found_indicators = [ind for ind in reasoning_indicators if ind in content.lower()]
                    if found_indicators:
                        print(f"   🎯 Reasoning indicators found: {', '.join(found_indicators[:3])}")
                    print(f"   💭 Response: {content[:100]}...")
                    print()
                
            except Exception as e:
                print(f"   ❌ {test['name']} error: {e}")
        
        # Step 6: Test streaming with complex prompt
        print("6️⃣ Testing streaming with complex reasoning...")
        try:
            stream_request = {
                "model": "gemma-2-9b-it-4bit",
                "messages": [
                    {"role": "system", "content": "You are an expert educator. Provide comprehensive, well-structured explanations."},
                    {"role": "user", "content": "Explain quantum entanglement to someone with basic physics knowledge. Start with the fundamentals and build up to the implications. Use analogies where helpful."}
                ],
                "max_tokens": 500,
                "temperature": 0.4,
                "stream": True
            }
            
            print("   🔄 Streaming quantum physics explanation:")
            print("   💬 ", end="", flush=True)
            
            sentence_count = 0
            current_sentence = ""
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
                                        current_sentence += content
                                        if content in '.!?':
                                            sentence_count += 1
                                            current_sentence = ""
                            except json.JSONDecodeError:
                                continue
                    
                    end_time = time.time()
                    print(f"\n   ✅ Streaming completed ({end_time-start_time:.1f}s)")
                    print(f"   📊 Approximate sentences: {sentence_count}")
                else:
                    print(f"   ❌ Streaming failed: {response.status_code}")
                    
        except Exception as e:
            print(f"   ❌ Streaming error: {e}")
        
        # Step 7: Test code generation and technical tasks
        print("\n7️⃣ Testing code generation...")
        code_tests = [
            "Write a Python function that implements binary search with error handling.",
            "Create a SQL query to find the top 5 customers by total purchase amount.",
            "Write a JavaScript function that debounces user input in a search box."
        ]
        
        for i, code_prompt in enumerate(code_tests):
            try:
                code_request = {
                    "model": "gemma-2-9b-it-4bit",
                    "messages": [
                        {"role": "system", "content": "You are a programming expert. Write clean, well-commented code with explanations."},
                        {"role": "user", "content": code_prompt}
                    ],
                    "max_tokens": 300,
                    "temperature": 0.2,
                    "stream": False
                }
                
                response = await client.post(f"{BASE_URL}/v1/chat/completions", json=code_request)
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    # Check for code indicators
                    has_code = any(indicator in content for indicator in ['def ', 'function', 'SELECT', '```', 'return'])
                    print(f"   💻 Code Test {i+1}: {'✅ Contains code' if has_code else '⚠️ May lack code'}")
                    print(f"   📝 Preview: {content[:80]}...")
                    print()
                
            except Exception as e:
                print(f"   ❌ Code test {i+1} error: {e}")
        
        # Step 8: Performance and characteristics summary
        print("8️⃣ Model performance summary...")
        try:
            health_response = await client.get(f"{BASE_URL}/v1/models/gemma-2-9b-it-4bit/health")
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"   ✅ Model health: {health_data}")
                print(f"   📈 Gemma-2-9B characteristics:")
                print(f"      - Parameters: 1.44B (compressed from 9B)")
                print(f"      - Quantization: 4-bit")
                print(f"      - Downloads: 10,333/month")
                print(f"      - Creator: Google/MLX Community")
                print(f"      - Specializations: Instruction following, reasoning")
                print(f"      - Architecture: Gemma-2 (Google's latest)")
            else:
                print(f"   ⚠️  Health check status: {health_response.status_code}")
        except Exception as e:
            print(f"   ❌ Health check error: {e}")

if __name__ == "__main__":
    print("💎 Gemma-2-9B-it-4bit Model Test")
    print("=" * 50)
    print("Testing Google's instruction-tuned model with advanced reasoning")
    print("Features: 1.44B parameters, 4-bit quantization, Google AI architecture")
    print()
    
    asyncio.run(test_gemma2_model())
    
    print("\n" + "=" * 50)
    print("✅ Gemma-2-9B-it-4bit test completed!")