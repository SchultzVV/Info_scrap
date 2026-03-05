"""
LayoutLMv3 Processor for Document Understanding

Infers semantic relationships between layout elements in retail flyers
Learns spatial patterns to link products with prices and discounts
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torch


@dataclass
class DocumentToken:
    """Token with text, bbox, and semantic label"""
    text: str
    bbox: Tuple[int, int, int, int]  # normalized (0-1000)
    label: str
    confidence: float


class LayoutLMProcessor:
    """
    LayoutLMv3 processor for retail flyer document understanding
    
    This module understands spatial relationships between:
    - Product names
    - Prices
    - Discount badges
    - Brand logos
    
    It learns patterns like:
    [product_title]
         ↓
    [price_tag]
    
    Or:
    [brand] → [product_title] → [price_tag] → [discount_badge]
    """
    
    def __init__(
        self,
        model_name: str = 'microsoft/layoutlmv3-base',
        use_gpu: bool = False
    ):
        """
        Initialize LayoutLM Processor
        
        Args:
            model_name: HuggingFace model name
            use_gpu: Use GPU acceleration
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        self.model = None
        self.processor = None
        
        # For production, this would load the actual model
        # For now, we implement rule-based spatial understanding
        self.use_rule_based = True
        
        print(f"[LayoutLM] Initialized (device: {self.device})")
        
        if self.use_rule_based:
            print("[LayoutLM] Using rule-based spatial reasoning (model-free)")
    
    def process(
        self,
        ocr_tokens: List,
        layout_regions: List,
        image_size: Tuple[int, int]
    ) -> Dict:
        """
        Process document to understand spatial relationships
        
        Args:
            ocr_tokens: Tokens from OCR engine
            layout_regions: Regions from YOLO detector
            image_size: (width, height) of original image
        
        Returns:
            Dict with:
                - relationships: Semantic relationships between elements
                - structured_data: Extracted structured information
        """
        # Normalize coordinates
        normalized_tokens = self._normalize_tokens(ocr_tokens, image_size)
        normalized_regions = self._normalize_regions(layout_regions, image_size)
        
        # Build spatial relationships
        if self.use_rule_based:
            relationships = self._rule_based_understanding(
                normalized_tokens,
                normalized_regions
            )
        else:
            relationships = self._model_based_understanding(
                normalized_tokens,
                normalized_regions,
                image_size
            )
        
        return {
            'relationships': relationships,
            'num_relationships': len(relationships)
        }
    
    def _normalize_tokens(self, tokens: List, image_size: Tuple[int, int]) -> List[Dict]:
        """Normalize token bboxes to 0-1000 scale (LayoutLM format)"""
        width, height = image_size
        
        normalized = []
        
        for token in tokens:
            x1, y1, x2, y2 = token.bbox
            
            # Normalize to 0-1000
            norm_x1 = int((x1 / width) * 1000)
            norm_y1 = int((y1 / height) * 1000)
            norm_x2 = int((x2 / width) * 1000)
            norm_y2 = int((y2 / height) * 1000)
            
            normalized.append({
                'text': token.text,
                'bbox': (norm_x1, norm_y1, norm_x2, norm_y2),
                'original_bbox': token.bbox,
                'confidence': token.confidence,
                'center': token.center
            })
        
        return normalized
    
    def _normalize_regions(self, regions: List, image_size: Tuple[int, int]) -> List[Dict]:
        """Normalize region bboxes to 0-1000 scale"""
        width, height = image_size
        
        normalized = []
        
        for region in regions:
            x1, y1, x2, y2 = region.bbox
            
            # Normalize to 0-1000
            norm_x1 = int((x1 / width) * 1000)
            norm_y1 = int((y1 / height) * 1000)
            norm_x2 = int((x2 / width) * 1000)
            norm_y2 = int((y2 / height) * 1000)
            
            normalized.append({
                'class_name': region.class_name,
                'bbox': (norm_x1, norm_y1, norm_x2, norm_y2),
                'original_bbox': region.bbox,
                'confidence': region.confidence,
                'center': region.center
            })
        
        return normalized
    
    def _rule_based_understanding(
        self,
        tokens: List[Dict],
        regions: List[Dict]
    ) -> List[Dict]:
        """
        Rule-based spatial understanding
        
        Learns patterns:
        1. Price tags below product titles
        2. Discount badges near price tags
        3. Brand logos above product titles
        """
        relationships = []
        
        # Group regions by class
        regions_by_class = {}
        for region in regions:
            class_name = region['class_name']
            if class_name not in regions_by_class:
                regions_by_class[class_name] = []
            regions_by_class[class_name].append(region)
        
        # Find product titles
        product_titles = regions_by_class.get('product_title', [])
        price_tags = regions_by_class.get('price_tag', [])
        discount_badges = regions_by_class.get('discount_badge', [])
        brand_logos = regions_by_class.get('brand_logo', [])
        
        # Link product titles with prices (spatial proximity)
        for title in product_titles:
            # Find nearest price below or nearby
            nearest_price = self._find_nearest_region(
                title,
                price_tags,
                direction='below'
            )
            
            if nearest_price:
                relationship = {
                    'type': 'product_price',
                    'product_region': title,
                    'price_region': nearest_price,
                    'confidence': min(title['confidence'], nearest_price['confidence'])
                }
                
                # Find associated discount
                nearest_discount = self._find_nearest_region(
                    nearest_price,
                    discount_badges,
                    direction='any',
                    max_distance=200
                )
                
                if nearest_discount:
                    relationship['discount_region'] = nearest_discount
                
                # Find associated brand
                nearest_brand = self._find_nearest_region(
                    title,
                    brand_logos,
                    direction='above'
                )
                
                if nearest_brand:
                    relationship['brand_region'] = nearest_brand
                
                relationships.append(relationship)
        
        return relationships
    
    def _find_nearest_region(
        self,
        source_region: Dict,
        target_regions: List[Dict],
        direction: str = 'any',
        max_distance: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Find nearest region in specified direction
        
        Args:
            source_region: Source region
            target_regions: List of candidate regions
            direction: 'above', 'below', 'left', 'right', 'any'
            max_distance: Maximum distance in normalized coordinates
        
        Returns:
            Nearest region or None
        """
        if not target_regions:
            return None
        
        source_cx, source_cy = source_region['center']
        
        candidates = []
        
        for target in target_regions:
            target_cx, target_cy = target['center']
            
            # Check direction constraint
            if direction == 'below' and target_cy <= source_cy:
                continue
            elif direction == 'above' and target_cy >= source_cy:
                continue
            elif direction == 'right' and target_cx <= source_cx:
                continue
            elif direction == 'left' and target_cx >= source_cx:
                continue
            
            # Calculate distance
            distance = np.sqrt(
                (target_cx - source_cx) ** 2 +
                (target_cy - source_cy) ** 2
            )
            
            # Check max distance
            if max_distance and distance > max_distance:
                continue
            
            candidates.append((distance, target))
        
        if not candidates:
            return None
        
        # Return nearest
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    
    def _model_based_understanding(
        self,
        tokens: List[Dict],
        regions: List[Dict],
        image_size: Tuple[int, int]
    ) -> List[Dict]:
        """
        Model-based understanding using LayoutLMv3
        
        This would use the actual transformer model to understand
        spatial relationships. For production systems.
        """
        # TODO: Implement actual LayoutLMv3 inference
        # For now, fallback to rule-based
        return self._rule_based_understanding(tokens, regions)
    
    def extract_structured_data(
        self,
        relationships: List[Dict],
        tokens: List[Dict]
    ) -> List[Dict]:
        """
        Extract structured product data from relationships
        
        Args:
            relationships: Spatial relationships
            tokens: OCR tokens
        
        Returns:
            List of structured product dicts
        """
        products = []
        
        for rel in relationships:
            if rel['type'] != 'product_price':
                continue
            
            product = {}
            
            # Extract product title text
            title_region = rel['product_region']
            product['product_name'] = self._extract_text_from_region(
                title_region,
                tokens
            )
            product['product_bbox'] = title_region['original_bbox']
            
            # Extract price text
            price_region = rel['price_region']
            product['price_text'] = self._extract_text_from_region(
                price_region,
                tokens
            )
            product['price_bbox'] = price_region['original_bbox']
            
            # Extract discount if present
            if 'discount_region' in rel:
                discount_region = rel['discount_region']
                product['discount_text'] = self._extract_text_from_region(
                    discount_region,
                    tokens
                )
                product['discount_bbox'] = discount_region['original_bbox']
            
            # Extract brand if present
            if 'brand_region' in rel:
                brand_region = rel['brand_region']
                product['brand_text'] = self._extract_text_from_region(
                    brand_region,
                    tokens
                )
                product['brand_bbox'] = brand_region['original_bbox']
            
            product['confidence'] = rel['confidence']
            
            products.append(product)
        
        return products
    
    def _extract_text_from_region(
        self,
        region: Dict,
        tokens: List[Dict]
    ) -> str:
        """Extract all text tokens that overlap with region"""
        region_bbox = region['original_bbox']
        rx1, ry1, rx2, ry2 = region_bbox
        
        texts = []
        
        for token in tokens:
            tx1, ty1, tx2, ty2 = token['original_bbox']
            
            # Check if token overlaps with region
            if self._boxes_overlap((rx1, ry1, rx2, ry2), (tx1, ty1, tx2, ty2)):
                texts.append(token['text'])
        
        return ' '.join(texts)
    
    def _boxes_overlap(
        self,
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> bool:
        """Check if two boxes overlap"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)
    
    def is_loaded(self) -> bool:
        """Check if processor is ready"""
        return True
