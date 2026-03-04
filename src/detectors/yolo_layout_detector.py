"""
YOLO Layout Detector for Retail Flyers

Detects semantic regions in retail flyer images:
- product_image
- price_tag
- discount_badge
- product_title
- brand_logo
- description
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import os


@dataclass
class DetectedRegion:
    """Represents a detected region in the flyer"""
    class_name: str
    class_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    center: Tuple[float, float]
    area: float


class YOLOLayoutDetector:
    """
    Production-grade YOLO detector for retail flyer layout analysis
    
    Follows SOLID principles:
    - Single Responsibility: Only detects layout regions
    - Open/Closed: Extensible for new region types
    - Liskov Substitution: Can be replaced with other detectors
    - Interface Segregation: Clear, minimal interface
    - Dependency Inversion: Depends on abstractions
    """
    
    # Default classes for retail flyer understanding
    DEFAULT_CLASSES = [
        'product_image',
        'price_tag',
        'discount_badge',
        'product_title',
        'brand_logo',
        'description'
    ]
    
    def __init__(
        self,
        model_path: str = 'yolov8n.pt',
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        """
        Initialize YOLO Layout Detector
        
        Args:
            model_path: Path to YOLO model (.pt file)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load YOLO model"""
        try:
            if os.path.exists(self.model_path):
                print(f"[YOLOLayoutDetector] Loading model: {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                print(f"[YOLOLayoutDetector] Model not found, using YOLOv8n pretrained")
                self.model = YOLO('yolov8n.pt')
            
            print(f"[YOLOLayoutDetector] Device: {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    def detect(
        self,
        image: np.ndarray,
        return_visualization: bool = False
    ) -> Dict:
        """
        Detect layout regions in retail flyer image
        
        Args:
            image: Input image (BGR numpy array)
            return_visualization: If True, includes annotated image
        
        Returns:
            Dict with:
                - regions: List of DetectedRegion objects
                - image_size: (width, height)
                - visualization: Annotated image (if requested)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Run inference
        results = self.model.predict(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        # Parse results
        regions = self._parse_results(results)
        
        # Prepare output
        output = {
            'regions': regions,
            'image_size': (image.shape[1], image.shape[0]),
            'num_regions': len(regions)
        }
        
        # Add visualization if requested
        if return_visualization:
            output['visualization'] = self._visualize(image, regions)
        
        return output
    
    def _parse_results(self, results) -> List[DetectedRegion]:
        """Parse YOLO results into DetectedRegion objects"""
        regions = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                # Extract coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Extract confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                # Calculate center and area
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)
                
                # Create region object
                region = DetectedRegion(
                    class_name=class_name,
                    class_id=class_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    center=(center_x, center_y),
                    area=area
                )
                
                regions.append(region)
        
        # Sort by confidence (highest first)
        regions.sort(key=lambda r: r.confidence, reverse=True)
        
        return regions
    
    def _visualize(
        self,
        image: np.ndarray,
        regions: List[DetectedRegion]
    ) -> np.ndarray:
        """
        Create visualization with bounding boxes and labels
        
        Args:
            image: Original image
            regions: List of detected regions
        
        Returns:
            Annotated image
        """
        vis_image = image.copy()
        
        # Color map for different classes
        colors = {
            'product_image': (0, 255, 0),      # Green
            'price_tag': (0, 0, 255),          # Red
            'discount_badge': (255, 0, 0),     # Blue
            'product_title': (255, 255, 0),    # Cyan
            'brand_logo': (255, 0, 255),       # Magenta
            'description': (0, 255, 255),      # Yellow
        }
        
        for region in regions:
            x1, y1, x2, y2 = region.bbox
            color = colors.get(region.class_name, (128, 128, 128))
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{region.class_name}: {region.confidence:.2f}"
            
            # Get label size
            (label_width, label_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                2
            )
            
            # Draw label background
            cv2.rectangle(
                vis_image,
                (x1, y1 - label_height - baseline - 5),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return vis_image
    
    def crop_regions(
        self,
        image: np.ndarray,
        regions: List[DetectedRegion],
        padding: int = 0
    ) -> Dict[str, np.ndarray]:
        """
        Crop detected regions from image
        
        Args:
            image: Original image
            regions: List of detected regions
            padding: Padding around bounding boxes
        
        Returns:
            Dict mapping region index to cropped image
        """
        crops = {}
        
        h, w = image.shape[:2]
        
        for idx, region in enumerate(regions):
            x1, y1, x2, y2 = region.bbox
            
            # Apply padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Crop
            crop = image[y1:y2, x1:x2]
            crops[idx] = crop
        
        return crops
    
    def filter_regions(
        self,
        regions: List[DetectedRegion],
        class_names: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        min_area: Optional[float] = None
    ) -> List[DetectedRegion]:
        """
        Filter regions by criteria
        
        Args:
            regions: List of regions to filter
            class_names: Keep only these classes
            min_confidence: Minimum confidence threshold
            min_area: Minimum area in pixels
        
        Returns:
            Filtered list of regions
        """
        filtered = regions
        
        if class_names:
            filtered = [r for r in filtered if r.class_name in class_names]
        
        if min_confidence:
            filtered = [r for r in filtered if r.confidence >= min_confidence]
        
        if min_area:
            filtered = [r for r in filtered if r.area >= min_area]
        
        return filtered
    
    def get_regions_by_class(
        self,
        regions: List[DetectedRegion]
    ) -> Dict[str, List[DetectedRegion]]:
        """
        Group regions by class name
        
        Args:
            regions: List of detected regions
        
        Returns:
            Dict mapping class name to list of regions
        """
        grouped = {}
        
        for region in regions:
            if region.class_name not in grouped:
                grouped[region.class_name] = []
            grouped[region.class_name].append(region)
        
        return grouped
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
