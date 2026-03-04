#!/usr/bin/env python3
"""
Test script for Flyer Understanding Pipeline
Tests the complete pipeline locally without API
"""

import cv2
import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import FlyerPipeline


def test_pipeline(image_path: str):
    """Test the complete pipeline"""
    
    print("=" * 70)
    print("🏷️  RETAIL FLYER UNDERSTANDING SYSTEM - Pipeline Test")
    print("=" * 70)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"\n❌ Error: Image file '{image_path}' not found!")
        sys.exit(1)
    
    # Load image
    print(f"\n📸 Loading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print("❌ Error: Could not load image!")
        sys.exit(1)
    
    print(f"✅ Image loaded: {image.shape[1]}x{image.shape[0]}")
    
    # Initialize pipeline
    print("\n🔧 Initializing pipeline...")
    print("-" * 70)
    
    pipeline = FlyerPipeline(
        yolo_model_path='yolov8n.pt',
        yolo_confidence=0.25,
        ocr_lang='pt',
        use_gpu=False
    )
    
    print("-" * 70)
    print("✅ Pipeline initialized!\n")
    
    # Process image
    print("🚀 Processing flyer through pipeline...")
    print("=" * 70)
    
    result = pipeline.process(
        image,
        return_visualization=True,
        return_debug_info=True
    )
    
    print("=" * 70)
    print("✅ Processing complete!\n")
    
    # Display results
    print("=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    
    if result['success']:
        print(f"\n✅ Success! Extracted {result['metadata']['num_products']} products\n")
        
        # Metadata
        print("📈 Processing Metadata:")
        for key, value in result['metadata'].items():
            print(f"   {key}: {value}")
        
        # Products
        print(f"\n🏷️  Extracted Products ({len(result['products'])}):")
        print("-" * 70)
        
        for idx, product in enumerate(result['products'], 1):
            print(f"\n   Product #{idx}:")
            print(f"      Name:       {product['product_name']}")
            print(f"      Brand:      {product['brand']}")
            print(f"      Price:      {product['price_formatted']}")
            print(f"      Discount:   {product['discount']}")
            print(f"      Confidence: {product['confidence']:.2%}")
            print(f"      BBox:       {product['bounding_box']}")
        
        # Debug info
        if 'debug' in result:
            print("\n" + "=" * 70)
            print("🐛 DEBUG INFORMATION")
            print("=" * 70)
            print(json.dumps(result['debug'], indent=2))
        
        # Save visualization
        if 'visualization' in result:
            vis_path = 'pipeline_result.jpg'
            cv2.imwrite(vis_path, result['visualization'])
            print(f"\n💾 Visualization saved: {vis_path}")
        
        # Save JSON
        result_copy = result.copy()
        if 'visualization' in result_copy:
            del result_copy['visualization']
        
        json_path = 'pipeline_result.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_copy, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved: {json_path}")
        
    else:
        print("❌ Processing failed!")
        if 'error' in result:
            print(f"Error: {result['error']}")
    
    print("\n" + "=" * 70)
    print("✨ Test complete!")
    print("=" * 70)


def main():
    """Main function"""
    
    # Get image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use example image
        image_path = 'image_example.png'
    
    # Run test
    test_pipeline(image_path)


if __name__ == "__main__":
    main()
