"""
YOLO-based Image Analyzer - Detects specific fields in product images
Uses trained YOLO model to locate: title, price, old_price, installment, stock, seller
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import io
import pytesseract
import re


class YoloAnalyzer:
    """
    Analyzes product images using YOLO detection
    
    Expected labels in YOLO model:
    - title: Product title/name
    - old_price: Original/strikethrough price
    - current_price: Sale/current price
    - installment: Payment installments info
    - stock: Stock quantity/availability
    - seller: Seller name/rating
    """
    
    # Label mapping
    EXPECTED_LABELS = {
        'title': 0,
        'old_price': 1,
        'current_price': 2,
        'installment': 3,
        'stock': 4,
        'seller': 5
    }
    
    def __init__(self, model_path: str = 'models/best.pt'):
        """
        Initialize YOLO analyzer
        
        Args:
            model_path: Path to trained YOLO best.pt model
        """
        self.model_path = model_path
        self.model = None
        self.model_loaded = False
        self.model_type = None  # 'custom' or 'tiny'
        self.device = None
        
        # Try to load model if exists
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model - tries custom model first, then tiny model for testing"""
        try:
            from ultralytics import YOLO
            import torch
            
            # PyTorch 2.6+ security fix - allowlist YOLO models
            try:
                torch.serialization.add_safe_globals([__import__('ultralytics.nn.tasks', fromlist=['DetectionModel']).DetectionModel])
            except:
                pass  # Fallback for older PyTorch versions
            
            # Detect device (GPU if available, else CPU)
            self.device = self._detect_device()
            print(f"🖥️  Using device: {self.device}")
            
            if os.path.exists(self.model_path):
                # Load custom trained model
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
                self.model_loaded = True
                self.model_type = 'custom'
                print(f"✅ Custom YOLO model loaded from {self.model_path}")
            else:
                # Fallback to tiny model for testing inference pipeline
                print(f"⚠️  Custom model not found at {self.model_path}")
                print(f"   Loading yolov8n (tiny) for testing inference pipeline...")
                try:
                    self.model = YOLO('yolov8n.pt')  # Nano model - ultra tiny
                    self.model.to(self.device)
                    self.model_loaded = True
                    self.model_type = 'tiny'
                    print(f"✅ YOLOv8n (tiny) model loaded for inference testing")
                except Exception as e:
                    print(f"⚠️  Could not load tiny model: {e}")
                    self.model_loaded = False
                    
        except ImportError:
            print("⚠️  ultralytics not installed - install via: pip install ultralytics")
            self.model_loaded = False
        except Exception as e:
            print(f"⚠️  Error loading YOLO model: {e}")
            self.model_loaded = False
    
    def _detect_device(self) -> str:
        """Detect available device (GPU or CPU)"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"   GPU detected: {gpu_name}")
                return '0'  # CUDA device 0
        except ImportError:
            pass
        
        print(f"   No GPU available, using CPU")
        return 'cpu'
    
    def analyze_image(self, image_data: bytes) -> Dict:
        """
        Analyze image using YOLO detections
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dict with detected fields and OCR'd text
        """
        # Convert to OpenCV format
        pil_image = Image.open(io.BytesIO(image_data))
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        img_height, img_width = opencv_image.shape[:2]
        
        detections = {}
        
        # If YOLO model loaded, use it
        if self.model_loaded:
            detections = self._detect_with_yolo(opencv_image)
        else:
            # Fallback: Return empty detections structure
            detections = self._get_empty_detections()
        
        # Extract text from detected regions
        extracted_info = self._extract_from_detections(opencv_image, detections)
        
        return {
            "success": True,
            "image_dimensions": {
                "width": img_width,
                "height": img_height
            },
            "detections": detections,
            "product": extracted_info,
            "model_status": {
                "loaded": self.model_loaded,
                "path": self.model_path,
                "type": self.model_type,
                "device": self.device
            }
        }
    
    def _detect_with_yolo(self, image: np.ndarray) -> Dict:
        """
        Run YOLO detection on image
        
        Args:
            image: OpenCV image (BGR)
            
        Returns:
            Dict with detections for each label
        """
        try:
            # Run inference
            results = self.model.predict(image, conf=0.5, verbose=False)
            result = results[0]
            
            detections = {label: [] for label in self.EXPECTED_LABELS.keys()}
            
            # Parse detections
            if result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Get class and confidence
                    class_id = int(boxes.cls[i])
                    confidence = float(boxes.conf[i])
                    
                    # Get box coordinates
                    box = boxes.xyxy[i]  # x1, y1, x2, y2
                    x1, y1, x2, y2 = [int(v) for v in box]
                    
                    # Find label name
                    label_name = None
                    for name, idx in self.EXPECTED_LABELS.items():
                        if idx == class_id:
                            label_name = name
                            break
                    
                    if label_name:
                        detections[label_name].append({
                            "bbox": {
                                "x1": x1, "y1": y1,
                                "x2": x2, "y2": y2,
                                "width": x2 - x1,
                                "height": y2 - y1
                            },
                            "confidence": round(confidence, 2)
                        })
            
            return detections
            
        except Exception as e:
            print(f"⚠️  YOLO detection error: {e}")
            return self._get_empty_detections()
    
    def _get_empty_detections(self) -> Dict:
        """Get empty detection structure"""
        return {label: [] for label in self.EXPECTED_LABELS.keys()}
    
    def _extract_from_detections(self, image: np.ndarray, detections: Dict) -> Dict:
        """
        Extract text from detected regions
        
        Args:
            image: OpenCV image
            detections: YOLO detections
            
        Returns:
            Dict with extracted information
        """
        extracted = {
            "title": None,
            "old_price": None,
            "current_price": None,
            "installment": None,
            "stock": None,
            "seller": None
        }
        
        # Process each detected region
        for label, boxes in detections.items():
            if not boxes:
                continue
            
            # Usually take first/best detection for each label
            box = boxes[0]
            bbox = box["bbox"]
            
            # Extract region
            x1, y1 = bbox["x1"], bbox["y1"]
            x2, y2 = bbox["x2"], bbox["y2"]
            
            # Pad region slightly for better OCR
            pad = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(image.shape[1], x2 + pad)
            y2 = min(image.shape[0], y2 + pad)
            
            region = image[y1:y2, x1:x2]
            
            if region.size == 0:
                continue
            
            # OCR on region
            pil_region = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(pil_region, lang='por+eng').strip()
            
            if not text:
                continue
            
            # Parse based on label type
            if label == "title":
                extracted["title"] = text
                
            elif label == "old_price":
                price_data = self._parse_price(text)
                if price_data:
                    extracted["old_price"] = price_data
                    
            elif label == "current_price":
                price_data = self._parse_price(text)
                if price_data:
                    extracted["current_price"] = price_data
                    
            elif label == "installment":
                installment_data = self._parse_installment(text)
                if installment_data:
                    extracted["installment"] = installment_data
                    
            elif label == "stock":
                stock_data = self._parse_stock(text)
                if stock_data:
                    extracted["stock"] = stock_data
                    
            elif label == "seller":
                extracted["seller"] = text
        
        return extracted
    
    def _parse_price(self, text: str) -> Optional[Dict]:
        """Parse price from text"""
        # Match patterns: R$ 123,45 | 123,45 | 123
        patterns = [
            r'R\$\s*(\d+[.,]\d+)',  # R$ 123,45
            r'(\d+[.,]\d+)',          # 123,45
            r'(\d+)',                 # 123
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                price_str = match.group(1)
                try:
                    # Convert Brazilian format to float
                    value = float(price_str.replace('.', '').replace(',', '.'))
                    
                    # Validate range
                    if 0.01 <= value <= 1000000:
                        return {
                            "raw_text": text,
                            "value": value,
                            "formatted": f"R$ {value:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
                        }
                except ValueError:
                    continue
        
        return None
    
    def _parse_installment(self, text: str) -> Optional[Dict]:
        """Parse installment info from text"""
        # Match: 12x R$ 10,50 | 12x de 10,50
        pattern = r'(\d+)\s*x\s*(?:de\s*)?(?:R\$\s*)?(\d+[.,]\d+)?'
        match = re.search(pattern, text)
        
        if match:
            num_installments = int(match.group(1))
            value_str = match.group(2)
            
            if value_str and 1 <= num_installments <= 48:
                try:
                    value = float(value_str.replace('.', '').replace(',', '.'))
                    total = num_installments * value
                    
                    return {
                        "raw_text": text,
                        "installments": num_installments,
                        "value_per_installment": value,
                        "total_value": total
                    }
                except ValueError:
                    pass
        
        return None
    
    def _parse_stock(self, text: str) -> Optional[Dict]:
        """Parse stock quantity from text"""
        # Match: "10 em estoque" | "apenas 5" | "1 disponível"
        patterns = [
            r'(\d+)\s*(?:em estoque|disponív|dispon)',
            r'(?:apenas|aprox\.?)\s*(\d+)',
            r'^(\d+)$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                quantity = int(match.group(1))
                if 0 <= quantity <= 999999:
                    return {
                        "raw_text": text,
                        "quantity": quantity
                    }
        
        return None
