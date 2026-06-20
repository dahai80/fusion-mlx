#!/usr/bin/env python3
"""
Test script specifically for Qwen3-8B-MLX-6bit model.
"""

import asyncio
import httpx
import json

BASE_URL = "http://localhost:8000"
MODEL_ID = "Qwen/Qwen3-8B-MLX-6bit"

async def test_qwen_model():
    """Test specifically with Qwen3-8B-MLX-6bit."""
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        print(f"🧪 Testing with {MODEL_ID}\n")
        
        # Step 1: Get model details from discovery
        print("1️⃣ Getting model details...")
        try:
            # URL encode the model ID
            encoded_model_id = MODEL_ID.replace("/", "%2F")
            response = await client.get(f"{BASE_URL}/v1/discover/models/{encoded_model_id}")
            if response.status_code == 200:
                model_info = response.json()
                print(f"   ✅ Model found: {model_info['name']}")
                print(f"   📊 Memory required: {model_info.get('estimated_memory_gb', 'Unknown')}GB")
                print(f"   🔧 MLX Compatible: {model_info['mlx_compatible']}")
                print(f"   ✅ System Compatible: {model_info['system_compatible']}")
                print(f"   💬 {model_info['compatibility_message']}")
            else:
                print(f"   ❌ Model details failed: {response.status_code}")
                print(f"   📝 Response: {response.text}")
                return
        except Exception as e:
            print(f"   ❌ Error getting model details: {e}")
            return
        
        # Step 2: Install the model
        print(f"\n2️⃣ Installing {MODEL_ID}...")
        try:
            install_data = {
                "model_id": MODEL_ID,
                "name": "qwen3-8b-6bit"  # Give it a simple name
            }
            response = await client.post(f"{BASE_URL}/v1/models/install", json=install_data)
            if response.status_code == 200:
                install_result = response.json()
                model_name = install_result['model_name']
                print(f"   ✅ Model installed: {model_name}")
                print(f"   📊 Status: {install_result['status']}")
            else:
                result = response.json()
                print(f"   ❌ Installation failed: {result.get('detail', 'Unknown error')}")
                print(f"   📝 Full response: {result}")
                return
        except Exception as e:
            print(f"   ❌ Error installing model: {e}")
            return
        
        # Step 3: Check installed models
        print(f"\n3️⃣ Listing installed models...")
        try:
            response = await client.get(f"{BASE_URL}/v1/models")
            if response.status_code == 200:
                models_list = response.json()
                print(f"   ✅ Installed models:")
                for model in models_list['data']:
                    print(f"      - {model['id']}")
            else:
                print(f"   ❌ Failed to list models: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error listing models: {e}")
        
        # Step 4: Test chat completions
        print(f"\n4️⃣ Testing chat completions...")
        try:
            chat_data = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello! What is your name and what can you do?"
                    }
                ],
                "max_tokens": 50,
                "temperature": 0.7
            }
            
            print("   🔄 Sending chat request (this may take a while for first load)...")
            response = await client.post(f"{BASE_URL}/v1/chat/completions", json=chat_data)
            
            if response.status_code == 200:
                result = response.json()
                message = result['choices'][0]['message']['content']
                usage = result['usage']
                
                print(f"   ✅ Success!")
                print(f"   🤖 Response: {message}")
                print(f"   📊 Usage: {usage['total_tokens']} tokens")
            else:
                result = response.json()
                print(f"   ❌ Chat failed: {result.get('detail', 'Unknown error')}")
                print(f"   📝 Status: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error in chat: {e}")
        
        print(f"\n🎉 Test completed!")


if __name__ == "__main__":
    asyncio.run(test_qwen_model())