#!/usr/bin/env python3
"""
Simple script to process an image using the flyer pipeline
Usage: python process_image.py <image_path>
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

def endpoint_that_process_img(img_path: str, use_modern: bool = False) -> Dict[str, Any]:
    """
    Process an image through the flyer analysis pipeline
    
    Args:
        img_path: Path to the image file
        use_modern: If True, use Modern v2 pipeline, else use Legacy v1
        
    Returns:
        Dictionary with extracted products and metadata
    """
    img_path = Path(img_path)
    
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    print(f"🖼️  Processing image: {img_path.name}")
    print(f"📦 Using {'Modern (v2)' if use_modern else 'Legacy (v1)'} pipeline")
    print("=" * 60)
    
    if use_modern:
        # Use Modern Multimodal Pipeline (v2)
        try:
            from pipelines.inference_pipeline import FlyerExtractionService
            
            pipeline = FlyerExtractionService(
                model_name="donut",
                use_gpu=False
            )
            
            result = pipeline.extract(str(img_path))
            
        except Exception as e:
            print(f"⚠️  Modern pipeline failed: {e}")
            print("🔄 Falling back to Legacy pipeline...")
            use_modern = False
    
    if not use_modern:
        # Use Legacy Pipeline (v1)
        from src.pipeline.flyer_pipeline import FlyerPipeline
        
        pipeline = FlyerPipeline(
            yolo_model_path="yolov8n.pt",
            use_gpu=False
        )
        
        result = pipeline.process(str(img_path))
    
    print("\n✅ Processing complete!")
    print(f"📊 Found {len(result.get('products', []))} products")
    
    return result


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python process_image.py <image_path> [--modern]")
        print("\nExamples:")
        print("  python process_image.py examples/image_example.png")
        print("  python process_image.py flyer.jpg --modern")
        sys.exit(1)
    
    img_path = sys.argv[1]
    use_modern = "--modern" in sys.argv or "-m" in sys.argv
    
    try:
        result = endpoint_that_process_img(img_path, use_modern=use_modern)
        
        # Pretty print results
        print("\n" + "=" * 60)
        print("📋 RESULTS:")
        print("=" * 60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Save to file
        output_file = Path("output.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
