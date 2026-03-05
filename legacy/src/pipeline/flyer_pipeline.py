"""
Retail Flyer Understanding Pipeline

Production-grade modular pipeline for extracting structured promotional data
from retail flyer images.

Pipeline stages:
1. Layout Detection (YOLO)
2. Region Cropping
3. OCR Extraction (PaddleOCR)
4. Document Understanding (LayoutLM)
5. Post-processing (Product-Price Linking)
"""

import cv2
import numpy as np
from typing import Dict, List, Optional
import time
from dataclasses import asdict

from ..detectors import YOLOLayoutDetector
from ..ocr import PaddleOCREngine
from ..layout import LayoutLMProcessor
from ..linking import ProductPriceLinker, PriceNormalizer


class FlyerPipeline:
    """
    End-to-end pipeline for retail flyer understanding
    
    Architecture principles:
    - SOLID design
    - Modular components
    - Each stage replaceable
    - Clear separation of concerns
    """
    
    def __init__(
        self,
        yolo_model_path: str = 'yolov8n.pt',
        yolo_confidence: float = 0.25,
        ocr_lang: str = 'pt',
        use_gpu: bool = False
    ):
        """
        Initialize Flyer Pipeline
        
        Args:
            yolo_model_path: Path to YOLO model
            yolo_confidence: YOLO confidence threshold
            ocr_lang: OCR language
            use_gpu: Use GPU acceleration
        """
        print("[FlyerPipeline] Initializing production pipeline...")
        
        # Stage 1: Layout Detection
        self.detector = YOLOLayoutDetector(
            model_path=yolo_model_path,
            confidence_threshold=yolo_confidence,
            device='cuda' if use_gpu else 'cpu'
        )
        
        # Stage 2: OCR Extraction
        self.ocr_engine = PaddleOCREngine(
            lang=ocr_lang,
            use_gpu=use_gpu
        )
        
        # Stage 3: Document Understanding
        self.layout_processor = LayoutLMProcessor(
            use_gpu=use_gpu
        )
        
        # Stage 4: Post-processing
        self.product_linker = ProductPriceLinker()
        self.price_normalizer = PriceNormalizer(currency='BRL')
        
        print("[FlyerPipeline] Pipeline initialized successfully!")
    
    def process(
        self,
        image: np.ndarray,
        return_visualization: bool = False,
        return_debug_info: bool = False
    ) -> Dict:
        """
        Process flyer image through complete pipeline
        
        Args:
            image: Input flyer image (BGR numpy array)
            return_visualization: Include annotated visualization
            return_debug_info: Include detailed debug information
        
        Returns:
            Dict with:
                - products: List of structured product data
                - metadata: Processing metadata
                - visualization: Annotated image (if requested)
                - debug: Debug information (if requested)
        """
        start_time = time.time()
        
        debug_info = {} if return_debug_info else None
        
        # Stage 1: Layout Detection
        print("[Pipeline] Stage 1/5: Layout Detection (YOLO)")
        layout_result = self.detector.detect(image, return_visualization=False)
        regions = layout_result['regions']
        image_size = layout_result['image_size']
        
        if return_debug_info:
            debug_info['stage1_layout_detection'] = {
                'num_regions': len(regions),
                'regions_by_class': {
                    class_name: len(region_list)
                    for class_name, region_list in self.detector.get_regions_by_class(regions).items()
                }
            }
        
        print(f"[Pipeline] Detected {len(regions)} regions")
        
        # Stage 2: Region Cropping
        print("[Pipeline] Stage 2/5: Region Cropping")
        crops = self.detector.crop_regions(image, regions, padding=5)
        
        # Stage 3: OCR Extraction
        print("[Pipeline] Stage 3/5: OCR Extraction (PaddleOCR)")
        ocr_result = self.ocr_engine.extract(image, preprocess=True)
        tokens = ocr_result['tokens']
        
        if return_debug_info:
            debug_info['stage3_ocr_extraction'] = {
                'num_tokens': len(tokens),
                'raw_text': ocr_result['raw_text']
            }
        
        print(f"[Pipeline] Extracted {len(tokens)} OCR tokens")
        
        # Stage 4: Document Understanding
        print("[Pipeline] Stage 4/5: Document Understanding (LayoutLM)")
        layout_understanding = self.layout_processor.process(
            tokens,
            regions,
            image_size
        )
        relationships = layout_understanding['relationships']
        
        if return_debug_info:
            debug_info['stage4_document_understanding'] = {
                'num_relationships': len(relationships)
            }
        
        print(f"[Pipeline] Found {len(relationships)} relationships")
        
        # Extract structured data
        structured_data = self.layout_processor.extract_structured_data(
            relationships,
            ocr_result['tokens']
        )
        
        # Stage 5: Post-processing
        print("[Pipeline] Stage 5/5: Post-processing (Product Linking)")
        products = self.product_linker.link(structured_data)
        
        print(f"[Pipeline] Extracted {len(products)} products")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Build output
        output = {
            'success': True,
            'products': [self._product_to_dict(p) for p in products],
            'metadata': {
                'num_products': len(products),
                'num_regions_detected': len(regions),
                'num_ocr_tokens': len(tokens),
                'num_relationships': len(relationships),
                'processing_time_seconds': round(processing_time, 2),
                'image_size': image_size
            }
        }
        
        # Add visualization
        if return_visualization:
            output['visualization'] = self.detector._visualize(image, regions)
        
        # Add debug info
        if return_debug_info:
            debug_info['total_processing_time'] = processing_time
            output['debug'] = debug_info
        
        return output
    
    def _product_to_dict(self, product) -> Dict:
        """Convert Product dataclass to dict"""
        return {
            'product_name': product.product_name,
            'brand': product.brand,
            'price': product.price,
            'price_formatted': self.price_normalizer.format_price(product.price),
            'discount': product.discount,
            'bounding_box': list(product.bounding_box),
            'confidence': round(product.confidence, 4)
        }
    
    def process_multiple(
        self,
        images: List[np.ndarray]
    ) -> List[Dict]:
        """
        Process multiple flyer images
        
        Args:
            images: List of flyer images
        
        Returns:
            List of results for each image
        """
        results = []
        
        for idx, image in enumerate(images):
            print(f"\n[Pipeline] Processing image {idx + 1}/{len(images)}")
            try:
                result = self.process(image)
                results.append(result)
            except Exception as e:
                print(f"[Pipeline] Error processing image {idx + 1}: {e}")
                results.append({
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def health_check(self) -> Dict:
        """
        Check pipeline health status
        
        Returns:
            Dict with component status
        """
        return {
            'pipeline': 'healthy',
            'components': {
                'yolo_detector': self.detector.is_loaded(),
                'ocr_engine': self.ocr_engine.is_loaded(),
                'layout_processor': self.layout_processor.is_loaded()
            }
        }
