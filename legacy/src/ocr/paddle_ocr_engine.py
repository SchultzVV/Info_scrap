"""
PaddleOCR Engine for Retail Flyer Text Extraction

Production-grade OCR with bounding boxes, text, and confidence scores
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class OCRToken:
    """Represents a single OCR token with spatial information"""
    text: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    center: Tuple[float, float]


class PaddleOCREngine:
    """
    PaddleOCR wrapper for retail flyer text extraction
    
    Provides:
    - Text extraction with bounding boxes
    - Confidence scores
    - Spatial token representation
    - Text preprocessing and normalization
    """
    
    def __init__(
        self,
        lang: str = 'pt',
        use_angle_cls: bool = True,
        use_gpu: bool = False
    ):
        """
        Initialize PaddleOCR Engine
        
        Args:
            lang: Language code ('pt', 'en', etc.)
            use_angle_cls: Enable text angle classification
            use_gpu: Use GPU acceleration
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        
        self.ocr = None
        self._load_engine()
    
    def _load_engine(self) -> None:
        """Load PaddleOCR engine"""
        try:
            from paddleocr import PaddleOCR
            
            print(f"[PaddleOCR] Initializing with lang={self.lang}, gpu={self.use_gpu}")
            
            self.ocr = PaddleOCR(
                lang=self.lang,
                use_angle_cls=self.use_angle_cls,
                use_gpu=self.use_gpu,
                show_log=False
            )
            
            print("[PaddleOCR] Engine loaded successfully")
            
        except ImportError:
            print("[PaddleOCR] PaddleOCR not available, falling back to EasyOCR")
            self._load_easyocr_fallback()
        except Exception as e:
            print(f"[PaddleOCR] Failed to load PaddleOCR: {e}")
            print("[PaddleOCR] Falling back to EasyOCR")
            self._load_easyocr_fallback()
    
    def _load_easyocr_fallback(self) -> None:
        """Fallback to EasyOCR if PaddleOCR unavailable"""
        try:
            import easyocr
            langs = ['pt', 'en'] if self.lang == 'pt' else ['en']
            self.ocr = easyocr.Reader(langs, gpu=self.use_gpu)
            self.using_fallback = True
            print("[PaddleOCR] Using EasyOCR as fallback")
        except Exception as e:
            raise RuntimeError(f"Failed to load both PaddleOCR and EasyOCR: {e}")
    
    def extract(
        self,
        image: np.ndarray,
        preprocess: bool = True
    ) -> Dict:
        """
        Extract text from image with spatial information
        
        Args:
            image: Input image (BGR numpy array)
            preprocess: Apply preprocessing to improve OCR
        
        Returns:
            Dict with:
                - tokens: List of OCRToken objects
                - raw_text: Concatenated text
                - num_tokens: Number of tokens
        """
        if self.ocr is None:
            raise RuntimeError("OCR engine not loaded")
        
        # Preprocess image
        if preprocess:
            image = self._preprocess_image(image)
        
        # Run OCR
        if hasattr(self, 'using_fallback') and self.using_fallback:
            tokens = self._extract_easyocr(image)
        else:
            tokens = self._extract_paddleocr(image)
        
        # Build output
        raw_text = ' '.join([t.text for t in tokens])
        
        return {
            'tokens': tokens,
            'raw_text': raw_text,
            'num_tokens': len(tokens)
        }
    
    def _extract_paddleocr(self, image: np.ndarray) -> List[OCRToken]:
        """Extract text using PaddleOCR"""
        try:
            result = self.ocr.ocr(image, cls=self.use_angle_cls)
            
            if not result or not result[0]:
                return []
            
            tokens = []
            
            for line in result[0]:
                bbox_points = line[0]
                text = line[1][0]
                confidence = float(line[1][1])
                
                # Convert bbox points to (x1, y1, x2, y2)
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))
                
                # Calculate center
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                token = OCRToken(
                    text=text,
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    center=(center_x, center_y)
                )
                
                tokens.append(token)
            
            return tokens
            
        except Exception as e:
            print(f"[PaddleOCR] Extraction error: {e}")
            return []
    
    def _extract_easyocr(self, image: np.ndarray) -> List[OCRToken]:
        """Extract text using EasyOCR (fallback)"""
        try:
            results = self.ocr.readtext(image)
            
            if not results:
                return []
            
            tokens = []
            
            for detection in results:
                bbox_points, text, confidence = detection
                
                # Convert bbox points to (x1, y1, x2, y2)
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))
                
                # Calculate center
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                token = OCRToken(
                    text=text,
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    center=(center_x, center_y)
                )
                
                tokens.append(token)
            
            return tokens
            
        except Exception as e:
            print(f"[EasyOCR] Extraction error: {e}")
            return []
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image: Input image
        
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize if too small
        height, width = gray.shape
        if height < 32 or width < 32:
            scale = max(32 / height, 32 / width)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Convert back to BGR for PaddleOCR
        result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def filter_tokens(
        self,
        tokens: List[OCRToken],
        min_confidence: Optional[float] = None,
        pattern: Optional[str] = None
    ) -> List[OCRToken]:
        """
        Filter tokens by criteria
        
        Args:
            tokens: List of tokens to filter
            min_confidence: Minimum confidence threshold
            pattern: Regex pattern to match text
        
        Returns:
            Filtered list of tokens
        """
        filtered = tokens
        
        if min_confidence:
            filtered = [t for t in filtered if t.confidence >= min_confidence]
        
        if pattern:
            regex = re.compile(pattern, re.IGNORECASE)
            filtered = [t for t in filtered if regex.search(t.text)]
        
        return filtered
    
    def extract_prices(self, tokens: List[OCRToken]) -> List[Dict]:
        """
        Extract price information from tokens
        
        Args:
            tokens: List of OCR tokens
        
        Returns:
            List of dicts with price info
        """
        prices = []
        
        # Price patterns
        patterns = [
            r'R?\$?\s*(\d+)[,.](\d{2})',  # R$ 36,99
            r'(\d{2,})(\d{2})',            # 3699
        ]
        
        for token in tokens:
            for pattern in patterns:
                match = re.search(pattern, token.text)
                if match:
                    if len(match.groups()) == 2:
                        reais = match.group(1)
                        centavos = match.group(2)
                        value = float(f"{reais}.{centavos}")
                    else:
                        value = float(match.group(1))
                    
                    prices.append({
                        'value': value,
                        'raw_text': token.text,
                        'bbox': token.bbox,
                        'confidence': token.confidence,
                        'center': token.center
                    })
                    break
        
        return prices
    
    def is_loaded(self) -> bool:
        """Check if OCR engine is loaded"""
        return self.ocr is not None
