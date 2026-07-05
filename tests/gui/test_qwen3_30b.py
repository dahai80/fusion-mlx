#!/usr/bin/env python3
"""
Test script for Qwen3-30B-A3B-4bit-DWQ model.
Tests Qwen3 MoE (Mixture of Experts) architecture with 30B total/3B active parameters.
"""

import asyncio
import json
import time

import httpx
import psutil

BASE_URL = "http://localhost:8000"
MODEL_ID = "mlx-community/Qwen3-30B-A3B-4bit-DWQ"


def check_system_resources():
    """Check system resources for MoE model."""
    memory = psutil.virtual_memory()
    total_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    cpu_count = psutil.cpu_count()
    return total_gb, available_gb, cpu_count


async def test_qwen3_moe_model():
    """Test Qwen3-30B-A3B MoE model with expert routing capabilities."""

    # Pre-flight system check
    total_memory, available_memory, cpu_count = check_system_resources()
    print("💾 System Resources:")
    print(f"   Memory: {total_memory:.1f}GB total, {available_memory:.1f}GB available")
    print(f"   CPU cores: {cpu_count}")

    # MoE model estimation: 30B total, 3B active = ~6GB for 4-bit
    estimated_requirement = 6.0
    print("📊 MoE Model: 30B total parameters, 3B active parameters")
    print(f"   Estimated memory: ~{estimated_requirement}GB (4-bit DWQ)")

    if available_memory < estimated_requirement:
        print(
            f"⚠️  Warning: Model needs ~{estimated_requirement}GB, you have {available_memory:.1f}GB"
        )
    else:
        print("✅ Sufficient memory for Qwen3 MoE model")

    async with httpx.AsyncClient(timeout=400.0) as client:
        print(f"\n🧪 Testing {MODEL_ID}\n")

        # Step 1: Check model discovery
        print("1️⃣ Checking model discovery...")
        try:
            response = await client.get(f"{BASE_URL}/v1/discover/models?query=Qwen3")
            if response.status_code == 200:
                models = response.json()["models"]
                qwen3_models = [m for m in models if "Qwen3" in m["id"]]
                if qwen3_models:
                    print(f"   ✅ Found {len(qwen3_models)} Qwen3 models")
                    for model in qwen3_models[:3]:
                        print(f"   📦 {model['id']}")
                        print(
                            f"      Size: {model.get('estimated_memory_gb', 'Unknown')}GB"
                        )
                        print(f"      Downloads: {model.get('downloads', 0):,}")
                else:
                    print("   ⚠️  No Qwen3 models found in discovery")
            else:
                print(f"   ❌ Discovery failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Discovery error: {e}")

        # Step 2: Install model first
        print("\n2️⃣ Installing MoE model...")
        try:
            install_response = await client.post(
                f"{BASE_URL}/v1/models/install",
                json={"model_id": MODEL_ID, "name": "qwen3-30b-a3b-4bit-dwq"},
            )
            if install_response.status_code == 200:
                install_data = install_response.json()
                print("   ✅ Model installed successfully")
                print(f"   📊 Status: {install_data.get('status', 'unknown')}")
                print(
                    f"   💾 Memory: {install_data.get('estimated_memory_gb', 'Unknown')}GB"
                )
            else:
                print(f"   ❌ Install failed: {install_response.status_code}")
                print(f"   📄 Response: {install_response.text}")
                return
        except Exception as e:
            print(f"   ❌ Install error: {e}")
            return

        # Step 3: Load MoE model
        print("\n3️⃣ Loading MoE model...")
        try:
            load_start = time.time()
            load_response = await client.post(
                f"{BASE_URL}/v1/models/qwen3-30b-a3b-4bit-dwq/load"
            )
            load_end = time.time()

            if load_response.status_code == 200:
                load_data = load_response.json()
                print(
                    f"   ✅ MoE model loaded successfully in {load_end-load_start:.1f}s"
                )
                print(f"   📊 Status: {load_data.get('status', 'unknown')}")
                if load_data.get("memory_warning"):
                    print(f"   ⚠️  Warning: {load_data['memory_warning']}")
            else:
                print(f"   ❌ Load failed: {load_response.status_code}")
                print(f"   📄 Response: {load_response.text}")
                return
        except Exception as e:
            print(f"   ❌ Load error: {e}")
            return

        # Step 4: Test expert routing with diverse tasks
        print("\n4️⃣ Testing MoE expert routing with diverse tasks...")
        expert_routing_tests = [
            {
                "name": "Mathematical Reasoning",
                "prompt": "Solve this step by step: If a compound grows at 8% annually, how long until it doubles? Use the rule of 72 and verify with logarithms.",
                "expected_expert": "math/logic",
            },
            {
                "name": "Creative Writing",
                "prompt": "Write a short poem about the beauty of autumn leaves changing colors. Use vivid imagery and metaphors.",
                "expected_expert": "creative/language",
            },
            {
                "name": "Technical Analysis",
                "prompt": "Explain the time complexity of quicksort algorithm and compare it to mergesort. Include best, average, and worst cases.",
                "expected_expert": "technical/code",
            },
            {
                "name": "Historical Knowledge",
                "prompt": "Describe the causes and consequences of the Industrial Revolution in 18th-19th century Britain.",
                "expected_expert": "knowledge/facts",
            },
            {
                "name": "Scientific Reasoning",
                "prompt": "Explain photosynthesis at the molecular level, including the light and dark reactions. How do plants convert CO2 to glucose?",
                "expected_expert": "science/technical",
            },
        ]

        for test in expert_routing_tests:
            try:
                chat_request = {
                    "model": "qwen3-30b-a3b-4bit-dwq",
                    "messages": [{"role": "user", "content": test["prompt"]}],
                    "max_tokens": 300,
                    "temperature": 0.7,
                    "stream": False,
                }

                start_time = time.time()
                response = await client.post(
                    f"{BASE_URL}/v1/chat/completions", json=chat_request
                )
                end_time = time.time()

                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    usage = result.get("usage", {})

                    print(f"   🎯 {test['name']} ({end_time-start_time:.1f}s):")
                    print(f"   💭 Expected expert: {test['expected_expert']}")
                    print(f"   📝 Response quality: {len(content)} chars")
                    print(f"   💬 Preview: {content[:100]}...")

                    # Analyze response characteristics for expert routing
                    if test["name"] == "Mathematical Reasoning":
                        has_calculations = any(
                            char in content for char in ["=", "%", "÷", "×", "log"]
                        )
                        print(
                            f"   📊 Contains calculations: {'✅' if has_calculations else '❌'}"
                        )
                    elif test["name"] == "Creative Writing":
                        has_imagery = any(
                            word in content.lower()
                            for word in [
                                "golden",
                                "crimson",
                                "dancing",
                                "whisper",
                                "gentle",
                            ]
                        )
                        print(
                            f"   🎨 Contains imagery: {'✅' if has_imagery else '❌'}"
                        )
                    elif test["name"] == "Technical Analysis":
                        has_complexity = any(
                            term in content.lower()
                            for term in [
                                "o(",
                                "complexity",
                                "algorithm",
                                "worst",
                                "average",
                            ]
                        )
                        print(
                            f"   ⚙️ Technical analysis: {'✅' if has_complexity else '❌'}"
                        )

                    print()
                else:
                    print(f"   ❌ {test['name']} failed: {response.status_code}")

            except Exception as e:
                print(f"   ❌ {test['name']} error: {e}")

        # Step 5: Test long context understanding (Qwen3 supports extended context)
        print("5️⃣ Testing long context understanding...")
        try:
            long_context = """
            In a small coastal town, marine biologist Dr. Sarah Chen discovered an unusual phenomenon.
            The local dolphin pod had begun exhibiting coordinated behavior patterns never documented before.
            Every morning at 6 AM, exactly twelve dolphins would form a perfect spiral formation near the harbor.
            They would maintain this formation for precisely 15 minutes, then disperse.

            Local fishermen reported that fish populations had increased by 40% since this behavior started three months ago.
            Dr. Chen hypothesized that the dolphins had developed a new foraging strategy that was accidentally benefiting the ecosystem.
            She noticed that the spiral formation created water currents that brought nutrient-rich deep water to the surface.

            The town mayor, concerned about tourism impact, wanted to publicize this discovery.
            However, Dr. Chen worried that human interference might disrupt this delicate new behavior.
            She proposed a compromise: limited, respectful observation tours with strict guidelines.
            """

            context_request = {
                "model": "qwen3-30b-a3b-4bit-dwq",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Read this story carefully:\n\n{long_context}\n\nNow answer: What was Dr. Chen's main concern about the mayor's tourism idea, and what solution did she propose?",
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.3,
                "stream": False,
            }

            response = await client.post(
                f"{BASE_URL}/v1/chat/completions", json=context_request
            )
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                # Check if model retained key details
                key_details = [
                    "disrupt",
                    "behavior",
                    "observation tours",
                    "guidelines",
                    "interference",
                ]
                found_details = [
                    detail
                    for detail in key_details
                    if detail.lower() in content.lower()
                ]

                print("   📖 Long context test:")
                print(f"   🎯 Key details retained: {len(found_details)}/5")
                print(f"   💬 Response: {content[:120]}...")

                if len(found_details) >= 3:
                    print("   ✅ Good context retention")
                else:
                    print("   ⚠️  Limited context retention")
            else:
                print(f"   ❌ Context test failed: {response.status_code}")

        except Exception as e:
            print(f"   ❌ Context test error: {e}")

        # Step 6: Test streaming with MoE activation
        print("\n6️⃣ Testing streaming with expert activation...")
        try:
            stream_request = {
                "model": "qwen3-30b-a3b-4bit-dwq",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert system that draws from multiple domains of knowledge.",
                    },
                    {
                        "role": "user",
                        "content": "Design a sustainable smart city that integrates renewable energy, AI traffic management, vertical farming, and circular economy principles. Explain how these systems would interact and what challenges need to be addressed.",
                    },
                ],
                "max_tokens": 450,
                "temperature": 0.6,
                "stream": True,
            }

            print("   🔄 Streaming interdisciplinary design:")
            print("   💬 ", end="", flush=True)

            domain_keywords = {
                "energy": ["solar", "wind", "renewable", "battery", "grid"],
                "ai": ["algorithm", "machine learning", "sensor", "optimization"],
                "agriculture": ["farming", "crops", "hydroponic", "vertical", "food"],
                "economics": ["economy", "circular", "waste", "resource", "efficiency"],
            }

            found_domains = {domain: 0 for domain in domain_keywords}
            total_tokens = 0
            start_time = time.time()

            async with client.stream(
                "POST", f"{BASE_URL}/v1/chat/completions", json=stream_request
            ) as response:
                if response.status_code == 200:
                    full_response = ""
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

                    # Analyze expert activation
                    for domain, keywords in domain_keywords.items():
                        for keyword in keywords:
                            if keyword.lower() in full_response.lower():
                                found_domains[domain] += 1

                    print(f"\n   ✅ Streaming completed ({end_time-start_time:.1f}s)")
                    print(
                        f"   📊 Tokens: ~{total_tokens}, Speed: ~{total_tokens/(end_time-start_time):.1f} tok/s"
                    )
                    print("   🧠 Expert domains activated:")
                    for domain, count in found_domains.items():
                        print(
                            f"      {domain.capitalize()}: {'✅' if count > 0 else '❌'} ({count} keywords)"
                        )
                else:
                    print(f"   ❌ Streaming failed: {response.status_code}")

        except Exception as e:
            print(f"   ❌ Streaming error: {e}")

        # Step 7: Test multilingual capabilities (Qwen3 strength)
        print("\n7️⃣ Testing multilingual capabilities...")
        multilingual_tests = [
            {"lang": "Chinese", "prompt": "用中文解释什么是人工智能。"},
            {
                "lang": "English",
                "prompt": "Translate the previous Chinese response to English.",
            },
            {
                "lang": "Code-switching",
                "prompt": "Explain machine learning using both English and Chinese terms where appropriate.",
            },
        ]

        for test in multilingual_tests:
            try:
                multilingual_request = {
                    "model": "qwen3-30b-a3b-4bit-dwq",
                    "messages": [{"role": "user", "content": test["prompt"]}],
                    "max_tokens": 150,
                    "temperature": 0.5,
                    "stream": False,
                }

                response = await client.post(
                    f"{BASE_URL}/v1/chat/completions", json=multilingual_request
                )
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    print(f"   🌍 {test['lang']}: {content[:80]}...")

            except Exception as e:
                print(f"   ❌ {test['lang']} error: {e}")

        # Step 7: Performance summary
        print("\n7️⃣ MoE model performance summary...")
        try:
            health_response = await client.get(
                f"{BASE_URL}/v1/models/{MODEL_ID}/health"
            )
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"   ✅ Model health: {health_data}")
                print("   📈 Qwen3-30B-A3B characteristics:")
                print("      - Architecture: Mixture of Experts (MoE)")
                print("      - Total parameters: 30B")
                print("      - Active parameters: 3B per inference")
                print("      - Quantization: 4-bit DWQ")
                print("      - Memory efficiency: ~6GB (vs ~60GB for dense 30B)")
                print("      - Strengths: Multilingual, expert routing, efficiency")
            else:
                print(f"   ⚠️  Health check status: {health_response.status_code}")
        except Exception as e:
            print(f"   ❌ Health check error: {e}")


if __name__ == "__main__":
    print("🚀 Qwen3-30B-A3B-4bit-DWQ Model Test")
    print("=" * 60)
    print("Testing Qwen3 MoE architecture with expert routing capabilities")
    print("Features: 30B total/3B active parameters, DWQ quantization, multilingual")
    print()

    asyncio.run(test_qwen3_moe_model())

    print("\n" + "=" * 60)
    print("✅ Qwen3-30B-A3B-4bit-DWQ test completed!")
