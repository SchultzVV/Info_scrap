#!/usr/bin/env python3
"""
Test script for Modern Multimodal Pipeline (v2.0)
Tests the state-of-the-art inference pipeline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.inference_pipeline import FlyerExtractionService
import json


def test_modern_pipeline(image_path: str):
    """Test the modern multimodal pipeline"""
    
    print("=" * 70)
    print("🚀 MODERN FLYER EXTRACTION - State of the Art 2026")
    print("=" * 70)
    print("\nArchitecture: Multimodal Transformer (end-to-end)")
    print("Old approach: YOLO + OCR + LayoutLM")
    print("New approach: Donut/Pix2Struct → Structured JSON")
    print("=" * 70)
    
    # Check image
    if not Path(image_path).exists():
        print(f"\n❌ Error: Image '{image_path}' not found!")
        sys.exit(1)
    
    print(f"\n📸 Image: {image_path}")
    
    # Initialize service
    print("\n🔧 Initializing modern pipeline...")
    print("-" * 70)
    
    service = FlyerExtractionService(
        model_name='donut',  # or 'pix2struct', 'kosmos2'
        use_gpu=False
    )
    
    print("-" * 70)
    print("✅ Service ready!\n")
    
    # Extract products
    print("🚀 Extracting products...")
    print("=" * 70)
    
    result = service.extract(image_path)
    
    print("=" * 70)
    print("✅ Extraction complete!\n")
    
    # Display results
    print("=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    
    if result['success']:
        print(f"\n✅ Success!")
        print(f"\n📈 Metadata:")
        for key, value in result['metadata'].items():
            if value:
                print(f"   {key}: {value}")
        
        print(f"\n🏷️  Products ({len(result['products'])}):")
        print("-" * 70)
        
        for idx, product in enumerate(result['products'], 1):
            print(f"\n   Product #{idx}:")
            print(f"      Name:       {product['product_name']}")
            print(f"      Price:      {product['price_formatted']}")
            if product['brand']:
                print(f"      Brand:      {product['brand']}")
            if product['discount']:
                print(f"      Discount:   {product['discount']}")
            print(f"      Confidence: {product['confidence']:.2%}")
        
        # Save results
        output_file = 'modern_pipeline_result.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved: {output_file}")
        
    else:
        print("❌ Extraction failed!")
        if 'error' in result:
            print(f"Error: {result['error']}")
    
    print("\n" + "=" * 70)
    print("✨ Test complete!")
    print("=" * 70)


def main():
    """Main function"""
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'examples/image_example.png'
    
    test_modern_pipeline(image_path)


if __name__ == "__main__":
    main()
