#!/usr/bin/env python3
"""
Test script for fusion_gui audio transcription endpoint.
Tests the /v1/audio/transcriptions endpoint with parakeet-tdt-0-6b-v2 model.
"""

import requests
import sys
from pathlib import Path
import json
import pytest

pytestmark = pytest.mark.skipif(
    True,
    reason="Integration test requiring running server on localhost:8000",
)

# Configuration
BASE_URL = "http://localhost:8000"
AUDIO_FILE = "test.wav"
MODEL = "parakeet-tdt-0-6b-v2"
RESPONSE_FORMAT = "json"  # Options: json, text, verbose_json, srt, vtt

# Get the directory where this test file is located
TEST_DIR = Path(__file__).parent

def test_audio_transcription():
    """Test the audio transcription endpoint."""

    # Check if audio file exists (look in the same directory as this test file)
    audio_path = TEST_DIR / AUDIO_FILE
    if not audio_path.exists():
        print(f"❌ Error: Audio file '{AUDIO_FILE}' not found")
        print(f"Please create a test audio file named '{AUDIO_FILE}' in the tests directory")
        assert False, f"Audio file '{AUDIO_FILE}' not found"

    print(f"🎵 Testing audio transcription with:")
    print(f"   File: {AUDIO_FILE}")
    print(f"   Model: {MODEL}")
    print(f"   Format: {RESPONSE_FORMAT}")
    print()

    # Prepare the request
    url = f"{BASE_URL}/v1/audio/transcriptions"

    # Prepare multipart form data
    files = {
        'file': (AUDIO_FILE, open(audio_path, 'rb'), 'audio/wav')
    }

    data = {
        'model': MODEL,
        'response_format': RESPONSE_FORMAT
    }

    try:
        print("📡 Sending request to fusion_gui...")
        response = requests.post(url, files=files, data=data)

        # Close the file
        files['file'][1].close()

        # Check response status
        if response.status_code == 200:
            print("✅ Transcription successful!")

            # Parse response based on format
            if RESPONSE_FORMAT == "json":
                result = response.json()
                print(f"📝 Transcription: {result.get('text', 'No text returned')}")

                if 'segments' in result:
                    print("\n📋 Segments:")
                    for i, segment in enumerate(result['segments']):
                        print(f"   {i+1}. [{segment.get('start', 0):.1f}s - {segment.get('end', 0):.1f}s]: {segment.get('text', '')}")

            elif RESPONSE_FORMAT == "text":
                print(f"📝 Transcription: {response.text}")

            elif RESPONSE_FORMAT == "verbose_json":
                result = response.json()
                print(f"📝 Transcription: {result.get('text', 'No text returned')}")
                print(f"📊 Segments: {len(result.get('segments', []))}")

            else:
                print(f"📝 Response ({RESPONSE_FORMAT}): {response.text}")

        else:
            print(f"❌ Error: HTTP {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Detail: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"   Response: {response.text}")

        assert response.status_code == 200

    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to fusion_gui server")
        print("   Make sure the server is running on http://localhost:8000")
        assert False, "Cannot connect to fusion_gui server"

    except Exception as e:
        print(f"❌ Error: {e}")
        assert False, f"Error: {e}"

def check_model_status():
    """Check if the parakeet model is loaded."""
    print("🔍 Checking model status...")

    try:
        # Check if model is loaded
        url = f"{BASE_URL}/v1/models/{MODEL}"
        response = requests.get(url)

        if response.status_code == 200:
            model_info = response.json()
            status = model_info.get('status', 'unknown')
            print(f"   Model '{MODEL}' status: {status}")

            if status != 'loaded':
                print(f"⚠️  Model '{MODEL}' is not loaded. It should load automatically on first use.")
                # Skipping manual loading; rely on fusion_gui to load the model when the first transcription request is made.
            else:
                print("✅ Model is ready!")

        else:
            print(f"❌ Model '{MODEL}' not found")
            print("   You may need to install it first:")
            print(f"   curl -X POST {BASE_URL}/v1/models/install \\")
            print(f'     -H "Content-Type: application/json" \\')
            print(f'     -d \'{{"model_id": "mlx-community/parakeet-tdt-0.6b-v2", "name": "{MODEL}"}}\'')
            assert False, f"Model '{MODEL}' not found"

    except Exception as e:
        print(f"❌ Error checking model status: {e}")
        assert False, f"Error checking model status: {e}"

def main():
    """Main test function."""
    print("🎯 fusion_gui Audio Transcription Test")
    print("=" * 50)

    # Check server connectivity
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ fusion_gui server is not responding")
            sys.exit(1)
        print("✅ fusion_gui server is running")
    except:
        print("❌ Cannot connect to fusion_gui server")
        print("   Make sure it's running on http://localhost:8000")
        sys.exit(1)

    # Check model status
    try:
        check_model_status()
    except AssertionError as e:
        print(f"\n❌ Model check failed: {e}")
        sys.exit(1)

    print()

    # Run the transcription test
    try:
        test_audio_transcription()
        print("\n🎉 Audio transcription test completed successfully!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Audio transcription test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
