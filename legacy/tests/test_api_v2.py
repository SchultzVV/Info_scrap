#!/usr/bin/env python3
"""
Test script for Modern API v2.0
Tests the state-of-the-art multimodal API
"""

import requests
import json
import sys
import os


def test_api_v2(image_path: str, api_url: str = "http://localhost:8000"):
    """Test the modern API endpoints"""
    
    print("=" * 70)
    print("🚀 MODERN API v2.0 TEST - State of the Art 2026")
    print("=" * 70)
    
    # Check image
    if not os.path.exists(image_path):
        print(f"\n❌ Error: Image '{image_path}' not found!")
        sys.exit(1)
    
    print(f"\n📸 Image: {image_path}")
    print(f"🌐 API URL: {api_url}")
    
    # Test 1: Root
    print("\n" + "=" * 70)
    print("TEST 1: Root Endpoint")
    print("=" * 70)
    
    try:
        response = requests.get(f"{api_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Root endpoint OK")
            data = response.json()
            print(f"\nAPI: {data['name']}")
            print(f"Version: {data['version']}")
            print(f"Architecture: {data['architecture']['new']}")
        else:
            print(f"⚠️  Status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API!")
        print("   Start the server:")
        print("   python src/serving/api_v2.py")
        sys.exit(1)
    
    # Test 2: Health
    print("\n" + "=" * 70)
    print("TEST 2: Health Check")
    print("=" * 70)
    
    response = requests.get(f"{api_url}/health", timeout=5)
    if response.status_code == 200:
        print("✅ Health check OK")
        health = response.json()
        print(f"\nStatus: {health['status']}")
        print(f"Model: {health['model']}")
        print(f"Architecture: {health['architecture']}")
    
    # Test 3: Models
    print("\n" + "=" * 70)
    print("TEST 3: Available Models")
    print("=" * 70)
    
    response = requests.get(f"{api_url}/models", timeout=5)
    if response.status_code == 200:
        print("✅ Models endpoint OK")
        models = response.json()
        print(f"\nCurrent model: {models['current_model']}")
        print("\nAvailable models:")
        for model in models['available_models']:
            print(f"  • {model['name']}: {model['description']}")
    
    # Test 4: Extract flyer
    print("\n" + "=" * 70)
    print("TEST 4: Extract Flyer (/flyer/extract)")
    print("=" * 70)
    
    print("\n🚀 Sending image to modern multimodal API...")
    
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/png')}
        params = {
            'remove_duplicates': True,
            'sort_by_confidence': True
        }
        
        response = requests.post(
            f"{api_url}/flyer/extract",
            files=files,
            params=params,
            timeout=60
        )
    
    if response.status_code == 200:
        result = response.json()
        
        print("✅ Extraction complete!")
        print("\n" + "=" * 70)
        print("RESULTS:")
        print("=" * 70)
        
        print(f"\n📊 Metadata:")
        if 'metadata' in result:
            for key, value in result['metadata'].items():
                if value:
                    print(f"   {key}: {value}")
        
        print(f"\n🏷️  Products ({len(result.get('products', []))}):")
        for idx, product in enumerate(result.get('products', []), 1):
            print(f"\n   {idx}. {product['product_name']}")
            print(f"      Price: {product['price_formatted']}")
            if product.get('brand'):
                print(f"      Brand: {product['brand']}")
            if product.get('discount'):
                print(f"      Discount: {product['discount']}")
            print(f"      Confidence: {product['confidence']:.2%}")
        
        # Save result
        with open('api_v2_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print("\n💾 Result saved: api_v2_result.json")
        
    else:
        print(f"❌ Error: Status {response.status_code}")
        print(response.text)
    
    print("\n" + "=" * 70)
    print("✨ API test complete!")
    print("=" * 70)


def main():
    """Main function"""
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'examples/image_example.png'
    
    api_url = os.getenv("API_URL", "http://localhost:8000")
    
    test_api_v2(image_path, api_url)


if __name__ == "__main__":
    main()
