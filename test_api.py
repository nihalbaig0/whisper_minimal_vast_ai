#!/usr/bin/env python3
"""
Test script for Whisper ASR API
Usage: python test_api.py <API_URL> [audio_file.wav]
"""

import sys
import requests
import json
from pathlib import Path

def test_health(base_url):
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_root(base_url):
    """Test root endpoint"""
    print("\n🔍 Testing root endpoint...")
    try:
        response = requests.get(base_url, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False

def test_models(base_url):
    """Test models endpoint"""
    print("\n🔍 Testing models endpoint...")
    try:
        response = requests.get(f"{base_url}/models", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Models endpoint failed: {e}")
        return False

def test_transcribe(base_url, audio_file=None):
    """Test transcription endpoint"""
    print("\n🔍 Testing transcription endpoint...")
    
    if audio_file and Path(audio_file).exists():
        print(f"Uploading file: {audio_file}")
        try:
            with open(audio_file, 'rb') as f:
                files = {'file': (Path(audio_file).name, f)}
                response = requests.post(
                    f"{base_url}/transcribe",
                    files=files,
                    timeout=60
                )
            
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print("\n✅ Transcription successful!")
                print(f"Text: {result.get('text', 'N/A')}")
                print(f"Language: {result.get('language', 'N/A')}")
                print(f"Segments: {len(result.get('segments', []))}")
                return True
            else:
                print(f"❌ Transcription failed: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return False
    else:
        print("⚠️  No audio file provided or file not found")
        print("To test transcription, run:")
        print(f"  python {sys.argv[0]} {base_url} your_audio.wav")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <API_URL> [audio_file.wav]")
        print("\nExample:")
        print(f"  python {sys.argv[0]} http://localhost:8000")
        print(f"  python {sys.argv[0]} http://localhost:8000 test_audio.wav")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip('/')
    audio_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("=" * 60)
    print("🎤 Whisper ASR API Test Suite")
    print("=" * 60)
    print(f"Testing API at: {base_url}\n")
    
    results = {
        "health": test_health(base_url),
        "root": test_root(base_url),
        "models": test_models(base_url),
        "transcribe": test_transcribe(base_url, audio_file)
    }
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⚠️  SKIPPED"
        print(f"{test_name:15s}: {status}")
    
    print("\n🌐 Interactive API Documentation:")
    print(f"   {base_url}/docs")
    print()

if __name__ == "__main__":
    main()