#!/usr/bin/env python3
"""
Test script for API endpoints
Tests the FastAPI server
"""

import requests
import json
import sys
import os


def test_api(image_path: str, api_url: str = "http://localhost:8000"):
    """Test API endpoints"""
    
    print("=" * 70)
    print("🏷️  RETAIL FLYER UNDERSTANDING SYSTEM - API Test")
    print("=" * 70)
    
    # Check image
    if not os.path.exists(image_path):
        print(f"\n❌ Error: Image file '{image_path}' not found!")
        sys.exit(1)
    
    print(f"\n📸 Image: {image_path}")
    print(f"🌐 API URL: {api_url}")
    
    # Test 1: Root endpoint
    print("\n" + "=" * 70)
    print("TEST 1: Root Endpoint")
    print("=" * 70)
    
    try:
        response = requests.get(f"{api_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Root endpoint OK")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"⚠️  Status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API!")
        print("   Make sure the server is running:")
        print("   python api.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Test 2: Health check
    print("\n" + "=" * 70)
    print("TEST 2: Health Check")
    print("=" * 70)
    
    response = requests.get(f"{api_url}/health", timeout=5)
    if response.status_code == 200:
        print("✅ Health check OK")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"⚠️  Status: {response.status_code}")
    
    # Test 3: Analyze flyer
    print("\n" + "=" * 70)
    print("TEST 3: Analyze Flyer")
    print("=" * 70)
    
    print("\n🚀 Sending image to /analyze-flyer endpoint...")
    
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/png')}
        params = {
            'return_visualization': False,
            'return_debug': True
        }
        
        response = requests.post(
            f"{api_url}/analyze-flyer",
            files=files,
            params=params,
            timeout=60
        )
    
    if response.status_code == 200:
        result = response.json()
        
        print("✅ Analysis complete!")
        print("\n" + "=" * 70)
        print("RESULTS:")
        print("=" * 70)
        
        print(f"\n📊 Metadata:")
        if 'metadata' in result:
            for key, value in result['metadata'].items():
                print(f"   {key}: {value}")
        
        print(f"\n🏷️  Products ({len(result.get('products', []))}):")
        for idx, product in enumerate(result.get('products', []), 1):
            print(f"\n   Product #{idx}:")
            print(f"      {product['product_name']}")
            print(f"      Price: {product['price_formatted']}")
            if product['brand']:
                print(f"      Brand: {product['brand']}")
            if product['discount']:
                print(f"      Discount: {product['discount']}")
            print(f"      Confidence: {product['confidence']:.2%}")
        
        # Save result
        with open('api_result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print("\n💾 Full result saved to: api_result.json")
        
    else:
        print(f"❌ Error: Status {response.status_code}")
        print(response.text)
    
    print("\n" + "=" * 70)
    print("✨ API test complete!")
    print("=" * 70)


def main():
    """Main function"""
    
    # Get image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'image_example.png'
    
    # Get API URL
    api_url = os.getenv("API_URL", "http://localhost:8000")
    
    # Run test
    test_api(image_path, api_url)


if __name__ == "__main__":
    main()
