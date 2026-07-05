#!/usr/bin/env python3
"""
Test script for API key validation in fusion_gui OpenAI endpoints.
"""

import requests

BASE_URL = "http://localhost:8000"


def test_api_key_formats():
    """Test different API key formats"""

    # Test data
    test_cases = [
        {"name": "No API key", "headers": {}, "expected": "should work"},
        {
            "name": "Authorization Bearer",
            "headers": {"Authorization": "Bearer sk-test123456789"},
            "expected": "should work",
        },
        {
            "name": "X-API-Key header",
            "headers": {"x-api-key": "sk-test987654321"},
            "expected": "should work",
        },
        {
            "name": "Both headers",
            "headers": {
                "Authorization": "Bearer sk-bearer123",
                "x-api-key": "sk-xapi456",
            },
            "expected": "should prefer Bearer",
        },
    ]

    print("Testing API key validation on OpenAI endpoints...\n")

    # Test /v1/models endpoint
    print("=== Testing /v1/models ===")
    for case in test_cases:
        try:
            response = requests.get(f"{BASE_URL}/v1/models", headers=case["headers"])
            print(f"✓ {case['name']}: {response.status_code}")
        except Exception as e:
            print(f"✗ {case['name']}: {e}")

    print("\n=== Testing /v1/chat/completions ===")

    # Test chat/completions endpoint
    chat_request = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
    }

    for case in test_cases:
        try:
            response = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                headers={**case["headers"], "Content-Type": "application/json"},
                json=chat_request,
            )
            print(
                f"✓ {case['name']}: {response.status_code} - {response.json().get('detail', 'OK')[:50]}"
            )
        except Exception as e:
            print(f"✗ {case['name']}: {e}")


if __name__ == "__main__":
    test_api_key_formats()
