#!/usr/bin/env python3
"""
🖼️ Vision Model Test Script
Tests vision functionality with Gemma 3n model using icon.png
"""

import asyncio
import base64
import httpx
import json
import os
from pathlib import Path

# Test configuration
BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 120.0

async def test_vision_with_icon():
    """Test vision model with the fusion_gui icon."""
    print("🖼️ Testing Vision Model with icon.png")
    print("=" * 50)
    
    # Get the icon.png file path
    icon_path = Path(__file__).parent.parent / "icon.png"
    
    if not icon_path.exists():
        print(f"❌ Icon file not found at: {icon_path}")
        return False
    
    print(f"📂 Using icon file: {icon_path}")
    
    # Read and encode the image
    try:
        with open(icon_path, 'rb') as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            img_url = f"data:image/png;base64,{img_base64}"
        
        print(f"📊 Image size: {len(img_data)} bytes")
        print(f"📊 Base64 size: {len(img_base64)} characters")
        
    except Exception as e:
        print(f"❌ Failed to read image: {e}")
        return False
    
    # Test with different Gemma 3n models
    test_models = [
        "gemma-3n-e4b-it-mlx-8bit",
        "gemma-3n-e4b-it", 
        "gemma-3-27b-it-qat-4bit"  # Fallback text model via MLX-VLM
    ]
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # First, check which models are available
        print("\n🔍 Checking available models...")
        try:
            response = await client.get(f"{BASE_URL}/v1/models")
            if response.status_code == 200:
                models_data = response.json()
                available_models = [model['id'] for model in models_data.get('data', [])]
                print(f"📋 Available models: {len(available_models)}")
                
                # Find the first available test model
                selected_model = None
                for model in test_models:
                    if model in available_models:
                        selected_model = model
                        print(f"✅ Using model: {selected_model}")
                        break
                
                if not selected_model:
                    print(f"❌ None of the test models are available")
                    print(f"   Test models: {test_models}")
                    print(f"   Available: {available_models}")
                    return False
                    
            else:
                print(f"❌ Failed to get models list: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error checking models: {e}")
            return False
        
        # Test vision generation
        print(f"\n🔄 Testing vision with {selected_model}...")
        
        # Create the chat request
        chat_data = {
            "model": selected_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see in this image? Describe it in detail."},
                        {"type": "image_url", "image_url": {"url": img_url}}
                    ]
                }
            ],
            "max_tokens": 100,
            "temperature": 0.1
        }
        
        print("📤 Sending vision request...")
        print(f"   Model: {selected_model}")
        print(f"   Prompt: What do you see in this image? Describe it in detail.")
        print(f"   Image: icon.png ({len(img_data)} bytes)")
        
        try:
            response = await client.post(f"{BASE_URL}/v1/chat/completions", json=chat_data)
            
            if response.status_code == 200:
                result = response.json()
                message = result['choices'][0]['message']['content'].strip()
                usage = result['usage']
                
                print("\n✅ Vision Generation Success!")
                print("=" * 30)
                print(f"🤖 Model Response:")
                print(f"   {message}")
                print(f"\n📊 Usage Stats:")
                print(f"   Total tokens: {usage['total_tokens']}")
                print(f"   Prompt tokens: {usage['prompt_tokens']}")
                print(f"   Completion tokens: {usage['completion_tokens']}")
                
                # Check if the response seems reasonable for the MLX icon
                keywords = ['m', 'letter', 'logo', 'icon', 'symbol', 'text', 'white', 'dark', 'mlx']
                found_keywords = [kw for kw in keywords if kw.lower() in message.lower()]
                
                if found_keywords:
                    print(f"\n🎯 Recognition Quality: Good")
                    print(f"   Found relevant keywords: {found_keywords}")
                else:
                    print(f"\n⚠️ Recognition Quality: Unclear")
                    print(f"   No obvious keywords detected")
                
                return True
                
            else:
                print(f"\n❌ Vision request failed: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('detail', 'Unknown error')}")
                except:
                    print(f"   Raw response: {response.text}")
                return False
                
        except Exception as e:
            print(f"\n❌ Vision request error: {e}")
            return False

async def main():
    """Main test function."""
    print("🚀 fusion_gui Vision Test")
    print("📋 Testing vision model with icon.png")
    print()
    
    success = await test_vision_with_icon()
    
    if success:
        print("\n🎉 Vision test completed successfully!")
    else:
        print("\n💥 Vision test failed!")
    
    return success

if __name__ == "__main__":
    # Check dependencies
    try:
        import httpx
        import PIL
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install httpx pillow")
        exit(1)
    
    # Run the test
    asyncio.run(main())