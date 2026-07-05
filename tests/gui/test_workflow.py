#!/usr/bin/env python3
"""
Test script for fusion_gui complete workflow:
1. Discover models
2. Install a model
3. Load the model
4. Use chat completions
"""

import asyncio
import time

import httpx

BASE_URL = "http://localhost:8000"


async def test_workflow():
    """Test the complete fusion_gui workflow."""

    async with httpx.AsyncClient(timeout=300.0) as client:
        print("🧪 Testing fusion_gui Complete Workflow\n")

        # Step 1: Check system status
        print("1️⃣ Checking system status...")
        try:
            response = await client.get(f"{BASE_URL}/v1/system/status")
            if response.status_code == 200:
                status = response.json()
                print(
                    f"   ✅ System: {status['system']['platform']} {status['system']['architecture']}"
                )
                print(f"   ✅ MLX Compatible: {status['mlx_compatible']}")
                print(
                    f"   ✅ Memory: {status['system']['memory']['available_gb']:.1f}GB available"
                )
            else:
                print(f"   ❌ System status check failed: {response.status_code}")
                return
        except Exception as e:
            print(f"   ❌ Error checking system status: {e}")
            return

        # Step 2: Discover compatible models
        print("\n2️⃣ Discovering compatible models...")
        try:
            response = await client.get(f"{BASE_URL}/v1/discover/compatible?limit=5")
            if response.status_code == 200:
                discovery = response.json()
                models = discovery["models"]
                print(f"   ✅ Found {len(models)} compatible models:")
                for model in models[:3]:  # Show first 3
                    print(
                        f"      - {model['id']} ({model['estimated_memory_gb']:.1f}GB)"
                    )

                # Choose a model for testing
                if models:
                    test_model = models[0]  # Use first compatible model
                    model_id = test_model["id"]
                    print(f"   📋 Selected for testing: {model_id}")
                else:
                    print("   ❌ No compatible models found")
                    return
            else:
                print(f"   ❌ Model discovery failed: {response.status_code}")
                return
        except Exception as e:
            print(f"   ❌ Error discovering models: {e}")
            return

        # Step 3: Install the model
        print(f"\n3️⃣ Installing model: {model_id}")
        try:
            install_data = {"model_id": model_id, "name": None}  # Use default name
            response = await client.post(
                f"{BASE_URL}/v1/models/install", json=install_data
            )
            if response.status_code == 200:
                install_result = response.json()
                model_name = install_result["model_name"]
                print(f"   ✅ Model installed: {model_name}")
                print(
                    f"   📊 Estimated memory: {install_result['estimated_memory_gb']:.1f}GB"
                )
            else:
                result = response.json()
                print(
                    f"   ❌ Installation failed: {result.get('detail', 'Unknown error')}"
                )
                return
        except Exception as e:
            print(f"   ❌ Error installing model: {e}")
            return

        # Step 4: Check installed models (OpenAI format)
        print("\n4️⃣ Checking installed models...")
        try:
            response = await client.get(f"{BASE_URL}/v1/models")
            if response.status_code == 200:
                models_list = response.json()
                print(f"   ✅ Found {len(models_list['data'])} installed models:")
                for model in models_list["data"]:
                    print(f"      - {model['id']}")
            else:
                print(f"   ❌ Failed to list models: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error listing models: {e}")

        # Step 5: Test chat completions
        print(f"\n5️⃣ Testing chat completions with {model_name}...")
        try:
            chat_data = {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant running on Apple Silicon with MLX.",
                    },
                    {
                        "role": "user",
                        "content": "Hello! Can you tell me what you are and how you're running?",
                    },
                ],
                "max_tokens": 100,
                "temperature": 0.7,
            }

            print("   🔄 Sending chat completion request...")
            start_time = time.time()
            response = await client.post(
                f"{BASE_URL}/v1/chat/completions", json=chat_data
            )
            end_time = time.time()

            if response.status_code == 200:
                result = response.json()
                assistant_message = result["choices"][0]["message"]["content"]
                usage = result["usage"]

                print("   ✅ Chat completion successful!")
                print(f"   🤖 Assistant: {assistant_message}")
                print(
                    f"   📊 Tokens: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion = {usage['total_tokens']} total"
                )
                print(f"   ⏱️  Time: {end_time - start_time:.2f}s")

                # Calculate tokens per second
                if end_time > start_time:
                    tps = usage["completion_tokens"] / (end_time - start_time)
                    print(f"   🚀 Speed: {tps:.1f} tokens/second")

            else:
                result = response.json()
                print(
                    f"   ❌ Chat completion failed: {result.get('detail', 'Unknown error')}"
                )
                return

        except Exception as e:
            print(f"   ❌ Error in chat completion: {e}")
            return

        # Step 6: Test another chat message (conversation)
        print("\n6️⃣ Testing follow-up message...")
        try:
            chat_data = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "What is 2 + 2?"},
                ],
                "max_tokens": 50,
                "temperature": 0.1,
            }

            response = await client.post(
                f"{BASE_URL}/v1/chat/completions", json=chat_data
            )
            if response.status_code == 200:
                result = response.json()
                assistant_message = result["choices"][0]["message"]["content"]
                print("   ✅ Follow-up successful!")
                print(f"   🤖 Assistant: {assistant_message}")
            else:
                print(f"   ❌ Follow-up failed: {response.status_code}")

        except Exception as e:
            print(f"   ❌ Error in follow-up: {e}")

        # Step 7: Check model status
        print("\n7️⃣ Checking model status...")
        try:
            response = await client.get(
                f"{BASE_URL}/v1/manager/models/{model_name}/status"
            )
            if response.status_code == 200:
                status = response.json()
                print(f"   ✅ Model status: {status['status']}")
                print(f"   📊 Memory usage: {status['memory_usage_gb']:.1f}GB")
                print(f"   ⏰ Last used: {status['last_used_at']}")
            else:
                print(f"   ❌ Status check failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error checking status: {e}")

        print("\n🎉 Workflow test completed successfully!")
        print("📋 Summary:")
        print(f"   - Model: {model_name}")
        print(f"   - Source: {model_id}")
        print("   - Status: Loaded and responding")
        print("   - API: OpenAI-compatible /v1/chat/completions")


if __name__ == "__main__":
    asyncio.run(test_workflow())
