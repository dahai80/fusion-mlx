#!/usr/bin/env python3
"""
🤖 CyberAI Image Format Test
Tests different ways CyberAI might send images to our OpenAI-compatible endpoint
"""

import asyncio
import base64
import httpx
import json
import os
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 120.0

async def test_different_image_formats():
    """Test various image formats that CyberAI might send."""
    print("🤖 Testing Different Image Formats")
    print("=" * 50)
    
    # Get the icon.png file
    icon_path = Path(__file__).parent.parent / "icon.png"
    if not icon_path.exists():
        print(f"❌ Icon file not found at: {icon_path}")
        return False
    
    # Read the image
    with open(icon_path, 'rb') as img_file:
        img_data = img_file.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
    
    # Test model
    model_name = "gemma-3n-e4b-it-mlx-8bit"
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        
        # Test Format 1: Standard OpenAI format (what our test script uses)
        print("\n📝 Test 1: Standard OpenAI format")
        format1 = {
            "model": model_name,
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "What do you see?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]
                }
            ],
            "max_tokens": 50
        }
        
        success1 = await send_test_request(client, "Standard OpenAI", format1)
        
        # Test Format 2: Different image_url structure (some clients use "image" instead of "url")
        print("\n📝 Test 2: Alternative image_url format")
        format2 = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see?"},
                        {"type": "image_url", "image_url": {"image": f"data:image/png;base64,{img_base64}"}}
                    ]
                }
            ],
            "max_tokens": 50
        }
        
        success2 = await send_test_request(client, "Alternative image_url", format2)
        
        # Test Format 3: Direct base64 without data prefix
        print("\n📝 Test 3: Direct base64 format")
        format3 = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see?"},
                        {"type": "image_url", "image_url": {"url": img_base64}}
                    ]
                }
            ],
            "max_tokens": 50
        }
        
        success3 = await send_test_request(client, "Direct base64", format3)
        
        # Test Format 4: Different content structure (image as separate field)
        print("\n📝 Test 4: Separate image field format")
        format4 = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": "What do you see?",
                    "image": f"data:image/png;base64,{img_base64}"
                }
            ],
            "max_tokens": 50
        }
        
        success4 = await send_test_request(client, "Separate image field", format4)
        
        # Test Format 5: Multiple images array
        print("\n📝 Test 5: Images array format")
        format5 = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": "What do you see?",
                    "images": [f"data:image/png;base64,{img_base64}"]
                }
            ],
            "max_tokens": 50
        }
        
        success5 = await send_test_request(client, "Images array", format5)
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Results Summary")
        print("=" * 50)
        
        results = [
            ("Standard OpenAI", success1),
            ("Alternative image_url", success2), 
            ("Direct base64", success3),
            ("Separate image field", success4),
            ("Images array", success5)
        ]
        
        for name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {name}")
        
        successful_count = sum(1 for _, success in results if success)
        print(f"\n🎯 Success Rate: {successful_count}/{len(results)} ({successful_count/len(results)*100:.1f}%)")
        
        return successful_count > 0

async def send_test_request(client, test_name, request_data):
    """Send a test request and return success status."""
    try:
        print(f"   📤 Sending {test_name} request...")
        
        # Print request structure for debugging
        print(f"   📋 Request structure:")
        if "content" in request_data["messages"][0]:
            content = request_data["messages"][0]["content"]
            if isinstance(content, list):
                print(f"      Content type: array with {len(content)} items")
                for i, item in enumerate(content):
                    if item.get("type") == "text":
                        print(f"      Item {i+1}: text = '{item.get('text', '')}'")
                    elif item.get("type") == "image_url":
                        img_url = item.get("image_url", {})
                        if "url" in img_url:
                            url_preview = img_url["url"][:50] + "..." if len(img_url["url"]) > 50 else img_url["url"]
                            print(f"      Item {i+1}: image_url.url = '{url_preview}'")
                        elif "image" in img_url:
                            img_preview = img_url["image"][:50] + "..." if len(img_url["image"]) > 50 else img_url["image"]
                            print(f"      Item {i+1}: image_url.image = '{img_preview}'")
            else:
                print(f"      Content type: string = '{content}'")
                if "image" in request_data["messages"][0]:
                    img_preview = request_data["messages"][0]["image"][:50] + "..."
                    print(f"      Image field: '{img_preview}'")
                elif "images" in request_data["messages"][0]:
                    images = request_data["messages"][0]["images"]
                    print(f"      Images array: {len(images)} images")
        
        response = await client.post(f"{BASE_URL}/v1/chat/completions", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']['content'].strip()
            print(f"   ✅ Success: '{message[:100]}{'...' if len(message) > 100 else ''}'")
            return True
        else:
            print(f"   ❌ Failed: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"      Error: {error_data.get('detail', 'Unknown')}")
            except:
                print(f"      Raw error: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_different_image_formats())