"""
Modern Inference Pipeline - State of the Art 2026

End-to-end pipeline using multimodal transformers:
    image → multimodal model → parsing → validation → JSON

Replaces: YOLO + OCR + LayoutLM
With: Single multimodal transformer (Donut/Pix2Struct)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import MultimodalExtractor
from src.parsing import FlyerParser
from src.validation import PriceValidator
from typing import Dict, List, Optional, Union
from PIL import Image
import numpy as np
import time


class FlyerExtractionService:
    """
    Modern flyer extraction service using multimodal AI
    
    Architecture:
        image → transformer → parsing → validation → structured JSON
    
    Advantages:
        ✓ Single end-to-end model
        ✓ No explicit OCR needed
        ✓ Fewer components = fewer errors
        ✓ Better context understanding
    """
    
    def __init__(
        self,
        model_name: str = 'donut',
        model_path: Optional[str] = None,
        use_gpu: bool = False,
        min_price: float = 0.01,
        max_price: float = 10000.0
    ):
        """
        Initialize Flyer Extraction Service
        
        Args:
            model_name: 'donut', 'pix2struct', or 'kosmos2'
            model_path: Path to fine-tuned model
            use_gpu: Use GPU acceleration
            min_price: Minimum valid price
            max_price: Maximum valid price
        """
        
        print("[FlyerExtractionService] Initializing modern pipeline...")
        print(f"[FlyerExtractionService] Model: {model_name}")
        
        # Initialize components
        self.extractor = MultimodalExtractor(
            model_name=model_name,
            model_path=model_path,
            use_gpu=use_gpu
        )
        
        self.parser = FlyerParser()
        
        self.validator = PriceValidator(
            min_price=min_price,
            max_price=max_price
        )
        
        print("[FlyerExtractionService] Pipeline ready!")
    
    def extract(
        self,
        image: Union[np.ndarray, Image.Image, str],
        prompt: Optional[str] = None,
        remove_duplicates: bool = True,
        sort_by_confidence: bool = True
    ) -> Dict:
        """
        Extract structured product data from flyer image
        
        Args:
            image: Input flyer image
            prompt: Optional custom prompt
            remove_duplicates: Remove duplicate products
            sort_by_confidence: Sort by confidence score
        
        Returns:
            Dict with products and metadata
        """
        
        start_time = time.time()
        
        # Step 1: Multimodal extraction
        print("[Pipeline] Step 1/4: Multimodal extraction")
        extraction_result = self.extractor.extract(image, prompt)
        
        # Step 2: Parse output
        print("[Pipeline] Step 2/4: Parsing output")
        products = self.parser.parse(extraction_result.raw_output)
        
        # Step 3: Validate prices
        print("[Pipeline] Step 3/4: Validating prices")
        validated_products, errors = self.validator.validate(products)
        
        # Step 4: Post-processing
        print("[Pipeline] Step 4/4: Post-processing")
        
        if remove_duplicates:
            validated_products = self.validator.remove_duplicates(validated_products)
        
        if sort_by_confidence:
            validated_products = self.validator.sort_by_confidence(validated_products)
        
        # Calculate metrics
        processing_time = time.time() - start_time
        
        # Build response
        return {
            'success': True,
            'products': validated_products,
            'metadata': {
                'num_products': len(validated_products),
                'model_used': extraction_result.model_name,
                'overall_confidence': extraction_result.confidence,
                'processing_time_seconds': round(processing_time, 2),
                'errors': errors if errors else None
            },
            'raw_output': extraction_result.raw_output if errors else None
        }
    
    def batch_extract(
        self,
        images: List[Union[np.ndarray, Image.Image, str]]
    ) -> List[Dict]:
        """
        Extract from multiple flyer images
        
        Args:
            images: List of flyer images
        
        Returns:
            List of extraction results
        """
        
        results = []
        
        for idx, image in enumerate(images):
            print(f"\n[Pipeline] Processing image {idx + 1}/{len(images)}")
            
            try:
                result = self.extract(image)
                results.append(result)
            except Exception as e:
                print(f"[Pipeline] Error processing image {idx + 1}: {e}")
                results.append({
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def health_check(self) -> Dict:
        """Check pipeline health status"""
        
        return {
            'status': 'healthy',
            'components': {
                'multimodal_extractor': self.extractor.is_loaded(),
                'parser': True,
                'validator': True
            },
            'model': self.extractor.model_name
        }


# Convenience function
def extract_from_flyer(
    image_path: str,
    model_name: str = 'donut',
    use_gpu: bool = False
) -> Dict:
    """
    Quick extraction from flyer image
    
    Args:
        image_path: Path to flyer image
        model_name: Model to use
        use_gpu: Use GPU
    
    Returns:
        Extraction results
    """
    
    service = FlyerExtractionService(
        model_name=model_name,
        use_gpu=use_gpu
    )
    
    return service.extract(image_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "examples/image_example.png"
    
    print("=" * 70)
    print("Modern Flyer Extraction Pipeline Test")
    print("=" * 70)
    
    result = extract_from_flyer(image_path)
    
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    
    print(f"\nModel: {result['metadata']['model_used']}")
    print(f"Products found: {result['metadata']['num_products']}")
    print(f"Processing time: {result['metadata']['processing_time_seconds']}s")
    
    print("\nProducts:")
    for idx, product in enumerate(result['products'], 1):
        print(f"\n  {idx}. {product['product_name']}")
        print(f"     Price: {product['price_formatted']}")
        if product['discount']:
            print(f"     Discount: {product['discount']}")
        print(f"     Confidence: {product['confidence']:.2%}")
