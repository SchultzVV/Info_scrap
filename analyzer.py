"""
Advanced Image Analyzer with ROI detection and proximity analysis
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates"""
    x: int
    y: int
    w: int
    h: int
    text: str
    confidence: float
    
    @property
    def center(self) -> Tuple[int, int]:
        """Calculate center point of bounding box"""
        return (self.x + self.w // 2, self.y + self.h // 2)
    
    @property
    def bottom_center(self) -> Tuple[int, int]:
        """Calculate bottom center point of bounding box"""
        return (self.x + self.w // 2, self.y + self.h)
    
    @property
    def area(self) -> int:
        """Calculate area of bounding box"""
        return self.w * self.h


class AdvancedImageAnalyzer:
    """
    Advanced analyzer for product/price extraction with:
    - ROI detection
    - Proximity analysis
    - Price-title linking
    - Validation and normalization
    """
    
    def __init__(self):
        self.price_pattern = re.compile(r'R\$?\s*[\d.]+(?:,\d{2})?')
        self.number_pattern = re.compile(r'[\d.]+(?:,\d{2})?')
        
    def analyze_image(self, image_data: bytes) -> Dict:
        """
        Main analysis method with ROI-based progressive detection
        
        Strategy:
        1. Find product title ROI (largest text box in upper region)
        2. Infer price ROI below title
        3. Re-run OCR in price region for better accuracy
        4. Apply sequential validation rules
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Structured analysis with products, prices, and relationships
        """
        # Convert to OpenCV format
        pil_image = Image.open(io.BytesIO(image_data))
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        img_height, img_width = opencv_image.shape[:2]
        
        # STEP 1: Detect product card ROIs (major regions)
        rois = self._detect_rois(opencv_image)
        
        # STEP 2: Extract all text to find titles
        line_boxes = self._extract_text_lines(opencv_image)
        
        # STEP 3: Identify TITLE (longest text in upper portion)
        title_box = self._find_title_by_position(line_boxes, img_height)
        
        if not title_box:
            return {
                "success": False,
                "error": "No product title detected",
                "image_dimensions": {"width": img_width, "height": img_height}
            }
        
        # STEP 4: Infer PRICE ROI (region below title)
        price_roi = self._infer_price_roi(title_box, img_width, img_height)
        
        # STEP 5: Extract prices from the inferred ROI with focused OCR
        price_region = opencv_image[
            price_roi['y']:price_roi['y'] + price_roi['h'],
            price_roi['x']:price_roi['x'] + price_roi['w']
        ]
        
        # Re-run OCR on price region for better accuracy
        price_boxes = self._extract_text_with_boxes(price_region)
        
        # DEBUG: Check if "a$" or "5709" was filtered out
        all_price_boxes_debug = []
        pil_region = Image.fromarray(cv2.cvtColor(price_region, cv2.COLOR_BGR2RGB))
        data = pytesseract.image_to_data(pil_region, lang='por+eng', output_type=pytesseract.Output.DICT)
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            if text and ('a$' in text.lower() or '5709' in text):
                all_price_boxes_debug.append({
                    "text": text,
                    "conf": conf,
                    "filtered": conf <= 30
                })
        
        # Adjust coordinates back to full image
        for box in price_boxes:
            box.x += price_roi['x']
            box.y += price_roi['y']
        
        # STEP 6: Classify prices by pattern and position
        # Pass the price region image for strikethrough detection
        classified_prices = self._classify_prices_with_strikethrough(
            price_boxes, 
            price_region,
            price_roi
        )
        
        # STEP 7: Build validated product structure
        product = self._build_validated_product(
            title_box, 
            classified_prices,
            img_height
        )
        
        # Debug info - test strikethrough detection on all price boxes
        debug_price_analysis = []
        for box in price_boxes:
            relative_box = BoundingBox(
                x=box.x - price_roi['x'],
                y=box.y - price_roi['y'],
                w=box.w,
                h=box.h,
                text=box.text,
                confidence=box.confidence
            )
            has_strikethrough = self._detect_strikethrough_in_region(price_region, relative_box)
            debug_price_analysis.append({
                "text": box.text,
                "y": box.y,
                "confidence": round(box.confidence, 2),
                "has_strikethrough": bool(has_strikethrough)  # Convert numpy.bool_ to Python bool
            })
        
        return {
            "success": True,
            "image_dimensions": {"width": img_width, "height": img_height},
            "rois_detected": len(rois),
            "rois": rois[:3],  # Top 3 ROIs
            "title_roi": {
                "x": title_box.x,
                "y": title_box.y,
                "w": title_box.w,
                "h": title_box.h
            },
            "price_roi": price_roi,
            "product": product,
            "debug": {
                "price_boxes_found": len(price_boxes),
                "price_analysis": debug_price_analysis,
                "old_price_search": all_price_boxes_debug if all_price_boxes_debug else "Not found in price region"
            }
        }
    
    def _detect_rois(self, image: np.ndarray) -> List[Dict]:
        """
        Detect regions of interest using contours and edge detection
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rois = []
        min_area = (image.shape[0] * image.shape[1]) * 0.01  # At least 1% of image
        
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter reasonable aspect ratios
            aspect_ratio = w / h if h > 0 else 0
            if 0.2 < aspect_ratio < 10:  # Reasonable product card ratios
                rois.append({
                    "id": idx + 1,
                    "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "area": int(area),
                    "aspect_ratio": round(aspect_ratio, 2)
                })
        
        # Sort by area (largest first)
        rois.sort(key=lambda r: r["area"], reverse=True)
        
        return rois[:5]  # Return top 5 ROIs
    
    def _extract_text_with_boxes(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Extract text with bounding boxes using Tesseract
        """
        # Use PIL Image for Tesseract
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Get detailed data from Tesseract
        data = pytesseract.image_to_data(
            pil_image, 
            lang='por+eng',
            output_type=pytesseract.Output.DICT
        )
        
        boxes = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            # Filter out empty or low confidence text (reduced threshold for "a$" patterns)
            if text and conf > 20:  # Lowered from 30 to capture strikethrough prices like "a$5709"
                box = BoundingBox(
                    x=data['left'][i],
                    y=data['top'][i],
                    w=data['width'][i],
                    h=data['height'][i],
                    text=text,
                    confidence=conf / 100.0
                )
                boxes.append(box)
        
        return boxes
    
    def _extract_text_lines(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Extract text as complete lines (not individual words)
        """
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Get text separated by lines
        text_lines = pytesseract.image_to_string(pil_image, lang='por+eng').split('\n')
        
        # Get bounding boxes for each line
        data = pytesseract.image_to_data(
            pil_image,
            lang='por+eng',
            output_type=pytesseract.Output.DICT
        )
        
        line_boxes = []
        current_line_num = -1
        current_line_words = []
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            line_num = data['line_num'][i]
            
            if not text or conf < 20:  # Lowered from 30 to capture "a$" patterns
                continue
            
            # New line detected
            if line_num != current_line_num:
                # Save previous line if it exists
                if current_line_words:
                    line_boxes.append(self._merge_boxes(current_line_words))
                
                current_line_num = line_num
                current_line_words = []
            
            # Add word to current line
            box = BoundingBox(
                x=data['left'][i],
                y=data['top'][i],
                w=data['width'][i],
                h=data['height'][i],
                text=text,
                confidence=conf / 100.0
            )
            current_line_words.append(box)
        
        # Don't forget the last line
        if current_line_words:
            line_boxes.append(self._merge_boxes(current_line_words))
        
        return line_boxes
    
    def _find_title_by_position(self, line_boxes: List[BoundingBox], img_height: int) -> Optional[BoundingBox]:
        """
        Find product title based on position and characteristics:
        - In upper 40% of image
        - Longest text (> 20 chars)
        - Larger font size (height > 12px)
        - Mixed alphanumeric content
        """
        candidates = []
        upper_region_limit = img_height * 0.4
        
        for box in line_boxes:
            # Must be in upper region
            if box.y > upper_region_limit:
                continue
            
            # Must be reasonably long
            if len(box.text) < 20:
                continue
            
            # Must have letters (not just numbers)
            if not re.search(r'[a-zA-Z]', box.text):
                continue
            
            # Larger font is better
            # Score based on: length + font size
            score = len(box.text) + (box.h * 2)
            
            candidates.append({
                "box": box,
                "score": score
            })
        
        if not candidates:
            return None
        
        # Return highest scoring candidate
        best = max(candidates, key=lambda c: c["score"])
        return best["box"]
    
    def _infer_price_roi(self, title_box: BoundingBox, img_width: int, img_height: int) -> Dict:
        """
        Infer the ROI where prices should be located:
        - Starts slightly ABOVE title bottom (to catch old price near title)
        - Below the title  
        - Extends down by ~200-300 pixels
        - Full width or centered region
        
        Logic: Prices are typically 20-300px from the title
        Old prices might be very close to title
        """
        # Start 60px BEFORE title starts (to catch old price that may be above title)
        roi_start_y = max(0, title_box.y - 60)
        roi_height = min(500, img_height - roi_start_y)  # Up to 500px down from start
        
        # Use centered region or full width
        roi_x = max(0, title_box.x - 50)  # Extend a bit to left
        roi_w = min(img_width - roi_x, title_box.w + 100)  # Extend a bit to right
        
        return {
            "x": int(roi_x),
            "y": int(roi_start_y),
            "w": int(roi_w),
            "h": int(roi_height)
        }
    
    def _classify_prices_by_sequence(self, price_boxes: List[BoundingBox]) -> Dict:
        """
        Classify prices by pattern and sequential position:
        
        Sequential rules:
        1. Price with STRIKETHROUGH (detected via image analysis) = OLD PRICE
        2. Price with 'a$' prefix = OLD PRICE (OCR error for strikethrough)
        3. Price with 'R$' appearing FIRST = CURRENT PRICE
        4. Price with 'x' multiplier = INSTALLMENT PRICE
        5. Vertical order: old price → current price → installment
        """
        classified = {
            "old_price": None,
            "current_price": None,
            "installment": None,
            "discount": None
        }
        
        # Sort boxes by vertical position (top to bottom)
        sorted_boxes = sorted(price_boxes, key=lambda b: b.y)
        
        for box in sorted_boxes:
            text = box.text
            text_lower = text.lower()
            
            # PRIORITY 1: OLD PRICE by STRIKETHROUGH detection
            # Check if this text region has a horizontal line through it
            if classified["old_price"] is None:
                # Check for strikethrough using 'a$' pattern (OCR artifact)
                if bool(re.search(r'[aA]\$\s*\d', text)):
                    classified["old_price"] = box
                    continue
            
            # DISCOUNT: percentage with % or "OFF"
            if classified["discount"] is None:
                if ('%' in text and any(c.isdigit() for c in text)) or 'off' in text_lower:
                    classified["discount"] = box
                    continue
            
            # INSTALLMENT: 'x' multiplier pattern (21x, 12x, etc.)
            # Must have number before 'x' and price after
            if classified["installment"] is None:
                if bool(re.search(r'\d+\s*[xX]', text)) and len(text) > 5:
                    classified["installment"] = box
                    continue
            
            # CURRENT PRICE: R$ with number (but not installment)
            if classified["current_price"] is None:
                has_currency = bool(re.search(r'[rR]\$', text))
                has_number = bool(re.search(r'\d+[.,]\d+', text))
                is_installment = bool(re.search(r'\d+\s*[xX]', text))
                
                if has_currency and has_number and not is_installment:
                    classified["current_price"] = box
                    continue
                
                # Also accept pure numbers (3.699, 185,44) if not already classified
                if has_number and len(text) < 15 and 'x' not in text_lower and 'a$' not in text_lower:
                    # Make sure it's a reasonable price (> 100)
                    value = self._extract_price_value(text)
                    if value and value > 50:
                        classified["current_price"] = box
                        continue
        
        return classified
    
    def _classify_prices_with_strikethrough(self, price_boxes: List[BoundingBox], price_region_image: np.ndarray, price_roi: Dict) -> Dict:
        """
        Classify prices using patterns and position:
        
        Logic:
        1. 'a$' + number = OLD PRICE (OCR artifact from strikethrough R$)
        2. 'R$' + number (strikethrough) = CURRENT PRICE  
        3. '[number]x' + price = INSTALLMENT
        4. Sequential vertical order
        """
        classified = {
            "old_price": None,
            "current_price": None,
            "installment": None,
            "discount": None
        }
        
        # Sort boxes by vertical position (top to bottom)
        sorted_boxes = sorted(price_boxes, key=lambda b: b.y)
        
        # First pass: find installment components ("21x" + nearby price)
        installment_multiplier_box = None
        for i, box in enumerate(sorted_boxes):
            text = box.text
            # Look for multiplier pattern (21x, 12x, etc.)
            if bool(re.match(r'^\d+[xX]$', text.strip())):
                installment_multiplier_box = box
                # Look for nearby price (within 30px horizontally and 5px vertically)
                for j in range(i+1, min(i+5, len(sorted_boxes))):
                    next_box = sorted_boxes[j]
                    vertical_dist = abs(next_box.y - box.y)
                    horizontal_dist = abs(next_box.x - (box.x + box.w))
                    
                    if vertical_dist < 5 and horizontal_dist < 100:
                        # Check if it's a price
                        if bool(re.search(r'\d+[.,]\d+', next_box.text)):
                            # Combine them as installment
                            combined_text = f"{box.text} {next_box.text}"
                            classified["installment"] = BoundingBox(
                                x=box.x,
                                y=box.y,
                                w=(next_box.x + next_box.w) - box.x,
                                h=max(box.h, next_box.h),
                                text=combined_text,
                                confidence=(box.confidence + next_box.confidence) / 2
                            )
                            break
                if classified["installment"]:
                    break
        
        # Second pass: classify prices
        for box in sorted_boxes:
            text = box.text
            text_lower = text.lower()
            
            # Skip if no digits
            if not any(c.isdigit() for c in text):
                continue
            
            # Skip if this box is part of installment
            if classified["installment"] and installment_multiplier_box and (
                box.x == installment_multiplier_box.x or 
                (abs(box.y - installment_multiplier_box.y) < 5 and abs(box.x - (installment_multiplier_box.x + installment_multiplier_box.w)) < 100)
            ):
                continue
            
            # PRIORITY 1: OLD PRICE - 'a$' pattern (OCR reads strikethrough R$ as a$)
            if classified["old_price"] is None:
                if bool(re.search(r'[aA]\$\s*\d', text)):
                    classified["old_price"] = box
                    continue
            
            # PRIORITY 2: DISCOUNT
            if classified["discount"] is None:
                if ('%' in text and any(c.isdigit() for c in text)) or 'off' in text_lower:
                    classified["discount"] = box
                    continue
            
            # PRIORITY 3: CURRENT PRICE - R$ or strikethrough or large standalone number
            if classified["current_price"] is None:
                relative_box = BoundingBox(
                    x=box.x - price_roi['x'],
                    y=box.y - price_roi['y'],
                    w=box.w,
                    h=box.h,
                    text=box.text,
                    confidence=box.confidence
                )
                
                has_strikethrough = self._detect_strikethrough_in_region(price_region_image, relative_box)
                has_currency = bool(re.search(r'[rR]\$', text))
                has_number = bool(re.search(r'\d+[.,]\d+', text))
                
                # Current price: has strikethrough (modern design) OR R$ symbol
                if (has_strikethrough and has_number) or (has_currency and has_number):
                    classified["current_price"] = box
                    continue
                
                # Pure numbers as fallback (3.699, etc.) - first large number
                if has_number and len(text) < 15 and 'x' not in text_lower and 'a$' not in text_lower:
                    value = self._extract_price_value(text)
                    if value and value > 500:  # Current price should be substantial
                        classified["current_price"] = box
                        continue
        
        # POST-CLASSIFICATION VALIDATION: Verify price relationships
        # Rule: old_price > current_price (always)
        if classified["old_price"] and classified["current_price"]:
            old_value = self._extract_price_value(classified["old_price"].text)
            current_value = self._extract_price_value(classified["current_price"].text)
            
            if old_value and current_value and old_value < current_value:
                # Swap them - we got it backwards
                classified["old_price"], classified["current_price"] = classified["current_price"], classified["old_price"]
        
        # Rule: installment total >= current_price
        if classified["installment"] and classified["current_price"]:
            installment_match = re.search(r'(\d+)[xX]\s*[\w$]*\s*(\d+[.,]\d+)', classified["installment"].text)
            if installment_match:
                num_installments = int(installment_match.group(1))
                value_per = self._extract_price_value(installment_match.group(2))
                
                if value_per:
                    installment_total = num_installments * value_per
                    current_value = self._extract_price_value(classified["current_price"].text)
                    
                    # Installment total should be >= current price (may include fees)
                    if current_value and installment_total < (current_value * 0.9):  # 10% tolerance
                        # Something is wrong, installment might be misclassified
                        pass  # Keep classification but flag in debug
        
        return classified
    
    def _detect_strikethrough_in_region(self, image: np.ndarray, bbox: BoundingBox) -> bool:
        """
        Detect if there's a horizontal line (strikethrough) in the text region
        
        Method:
        1. Extract the text region
        2. Convert to grayscale and threshold
        3. Look for strong horizontal lines in the middle vertical region
        4. A strikethrough typically crosses the middle 30-70% of text height
        """
        try:
            # Extract region
            x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
            
            # Add small padding
            pad = 2
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(image.shape[1] - x, w + 2*pad)
            h = min(image.shape[0] - y, h + 2*pad)
            
            region = image[y:y+h, x:x+w]
            
            if region.size == 0:
                return False
            
            # Convert to grayscale
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region
            
            # Apply threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Focus on middle section (where strikethrough would be)
            middle_start = int(h * 0.3)
            middle_end = int(h * 0.7)
            middle_section = binary[middle_start:middle_end, :]
            
            if middle_section.size == 0:
                return False
            
            # Calculate horizontal projection (sum of white pixels per row)
            h_projection = np.sum(middle_section, axis=1)
            
            # Normalize by width
            h_projection_normalized = h_projection / (w * 255)
            
            # A strikethrough line will have high density (> 60% of width)
            max_density = np.max(h_projection_normalized) if len(h_projection_normalized) > 0 else 0
            
            # Threshold: if any row has > 50% horizontal coverage, likely strikethrough
            return max_density > 0.5
            
        except Exception as e:
            # If detection fails, return False
            return False
    
    def _build_validated_product(self, title_box: BoundingBox, classified_prices: Dict, img_height: int) -> Dict:
        """
        Build validated product structure with all price information
        """
        product = {
            "title": title_box.text,
            "title_confidence": round(title_box.confidence, 2),
            "title_bbox": {
                "x": title_box.x,
                "y": title_box.y,
                "w": title_box.w,
                "h": title_box.h
            },
            "old_price": None,
            "current_price": None,
            "installment": None,
            "discount": None,
            "validation": {
                "has_title": True,
                "has_old_price": False,
                "has_current_price": False,
                "has_installment": False,
                "has_discount": False
            }
        }
        
        # Process OLD PRICE
        if classified_prices["old_price"]:
            box = classified_prices["old_price"]
            value = self._extract_price_value(box.text)
            if value:
                product["old_price"] = {
                    "raw_text": box.text,
                    "value": value,
                    "formatted": self._format_brl_price(value),
                    "confidence": round(box.confidence, 2)
                }
                product["validation"]["has_old_price"] = True
        
        # Process CURRENT PRICE
        if classified_prices["current_price"]:
            box = classified_prices["current_price"]
            value = self._extract_price_value(box.text)
            if value:
                product["current_price"] = {
                    "raw_text": box.text,
                    "value": value,
                    "formatted": self._format_brl_price(value),
                    "confidence": round(box.confidence, 2)
                }
                product["validation"]["has_current_price"] = True
        
        # Process INSTALLMENT
        if classified_prices["installment"]:
            box = classified_prices["installment"]
            installment_match = re.search(r'(\d+)\s*x.*?([\d.,]+)', box.text)
            if installment_match:
                num_installments = int(installment_match.group(1))
                installment_value = self._extract_price_value(installment_match.group(2))
                
                if installment_value:
                    product["installment"] = {
                        "raw_text": box.text,
                        "installments": num_installments,
                        "value_per_installment": installment_value,
                        "total_value": installment_value * num_installments,
                        "formatted_total": self._format_brl_price(installment_value * num_installments),
                        "formatted_per_installment": self._format_brl_price(installment_value),
                        "confidence": round(box.confidence, 2)
                    }
                    product["validation"]["has_installment"] = True
        
        # Process DISCOUNT
        if classified_prices["discount"]:
            box = classified_prices["discount"]
            discount_match = re.search(r'(\d+)\s*%', box.text)
            if discount_match:
                product["discount"] = {
                    "raw_text": box.text,
                    "percentage": int(discount_match.group(1)),
                    "confidence": round(box.confidence, 2)
                }
                product["validation"]["has_discount"] = True
        
        # Calculate discount if we have both prices
        if product["old_price"] and product["current_price"]:
            old_val = product["old_price"]["value"]
            current_val = product["current_price"]["value"]
            calculated_discount = round(((old_val - current_val) / old_val) * 100, 1)
            
            # Validate against detected discount
            if product["discount"]:
                detected_discount = product["discount"]["percentage"]
                # Allow 1% tolerance
                if abs(calculated_discount - detected_discount) <= 1:
                    product["validation"]["discount_verified"] = True
                else:
                    product["validation"]["discount_verified"] = False
                    product["validation"]["discount_mismatch"] = {
                        "detected": detected_discount,
                        "calculated": calculated_discount
                    }
            else:
                product["discount"] = {
                    "percentage": calculated_discount,
                    "source": "calculated"
                }
        
        return product
    
    def _format_brl_price(self, value: float) -> str:
        """Format price in Brazilian Real format: R$ 3.699,00"""
        formatted = f"R$ {value:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        return formatted
    
    def _classify_text_boxes(self, boxes: List[BoundingBox], line_boxes: List[BoundingBox] = None) -> Dict[str, List[BoundingBox]]:
        """
        Classify text boxes based on POSITION and LAYOUT (not keywords)
        - Titles: Larger text at the top of product regions
        - Old prices: Crossed-out or with 'a$' prefix (OCR error for strikethrough)
        - Current prices: Main price with R$ or $
        - Installment: Prices with 'x' multiplier pattern
        """
        classified = {
            "titles": [],
            "prices": [],
            "old_prices": [],
            "installment_prices": [],
            "discounts": [],
            "descriptions": [],
            "metadata": []
        }
        
        # Use line_boxes for better title detection (complete lines)
        all_boxes = (line_boxes or []) + boxes
        
        for box in all_boxes:
            text = box.text
            text_lower = text.lower()
            
            # Priority 1: Detect TITLES by characteristics:
            # - Longer text (> 20 chars)
            # - Larger bounding box height (bigger font)
            # - Contains mixed alphanumeric (not just numbers)
            # - Not a price pattern
            is_long_text = len(text) > 20
            is_large_font = box.h > 12  # Larger than typical body text
            has_letters = bool(re.search(r'[a-zA-Z]', text))
            has_numbers = bool(re.search(r'\d', text))
            mixed_content = has_letters and len(text) > 8
            
            # Title pattern: long text with mixed content, larger font
            if is_long_text and mixed_content and is_large_font:
                # Make sure it's not a description (too long) or URL
                if len(text) < 100 and '/' not in text[:10]:
                    classified["titles"].append(box)
                    continue
            
            # Priority 2: Detect OLD PRICES (crossed out)
            # OCR often shows 'a$' instead of 'R$' for strikethrough prices
            if 'a$' in text_lower or (text.startswith('a') and any(c.isdigit() for c in text)):
                classified["old_prices"].append(box)
                continue
            
            # Priority 3: Detect INSTALLMENT PRICES
            # Pattern: "21x R$ 185,44" or similar with 'x' multiplier
            if 'x' in text_lower and any(c.isdigit() for c in text):
                # Check if there's a number before 'x'
                if bool(re.search(r'\d+\s*x', text_lower)):
                    classified["installment_prices"].append(box)
                    continue
            
            # Priority 4: Detect DISCOUNTS
            if 'off' in text_lower or ('%' in text and any(c.isdigit() for c in text)):
                classified["discounts"].append(box)
                continue
            
            # Priority 5: Detect CURRENT PRICES
            # Pattern: Has R$ or $ and numbers with decimal/comma
            has_currency = 'r$' in text_lower or '$' in text
            has_price_format = bool(re.search(r'\d+[.,]\d+', text))
            is_numeric_only = text.replace('.', '').replace(',', '').replace(' ', '').replace('R', '').replace('$', '').isdigit()
            
            if any(c.isdigit() for c in text):
                # Currency with price format
                if has_currency and (has_price_format or is_numeric_only):
                    if 'x' not in text_lower:  # Not installment
                        classified["prices"].append(box)
                        continue
                
                # Pure price format (3.699, 185,44)
                if has_price_format and len(text) < 20 and '/' not in text:
                    classified["prices"].append(box)
                    continue
            
            # Priority 6: Descriptions (longer text without special patterns)
            if len(text) > 40:
                classified["descriptions"].append(box)
                continue
            
            # Everything else is metadata
            classified["metadata"].append(box)
        
        return classified
    
    def _merge_nearby_titles(self, titles: List[BoundingBox]) -> List[BoundingBox]:
        """
        Merge title boxes that are close together horizontally
        (e.g., "iPhone 16e" and "(128 Gb)" and "- Branco")
        """
        if not titles:
            return titles
        
        # Sort by vertical position, then horizontal
        sorted_titles = sorted(titles, key=lambda t: (t.y, t.x))
        
        merged = []
        current_group = [sorted_titles[0]]
        
        for i in range(1, len(sorted_titles)):
            prev = current_group[-1]
            curr = sorted_titles[i]
            
            # Check if on similar vertical level (within 20px) and horizontally close (within 200px)
            vertical_close = abs(curr.y - prev.y) < 20
            horizontal_close = abs(curr.x - (prev.x + prev.w)) < 200
            
            if vertical_close and horizontal_close:
                current_group.append(curr)
            else:
                # Merge current group into one box
                if len(current_group) > 1:
                    merged_box = self._merge_boxes(current_group)
                    merged.append(merged_box)
                else:
                    merged.append(current_group[0])
                current_group = [curr]
        
        # Don't forget the last group
        if len(current_group) > 1:
            merged_box = self._merge_boxes(current_group)
            merged.append(merged_box)
        else:
            merged.append(current_group[0])
        
        return merged
    
    def _merge_boxes(self, boxes: List[BoundingBox]) -> BoundingBox:
        """Merge multiple boxes into one"""
        # Sort boxes by horizontal position
        sorted_boxes = sorted(boxes, key=lambda b: b.x)
        
        # Calculate merged bounding box
        x_min = min(b.x for b in boxes)
        y_min = min(b.y for b in boxes)
        x_max = max(b.x + b.w for b in boxes)
        y_max = max(b.y + b.h for b in boxes)
        
        # Merge text with spaces
        merged_text = " ".join(b.text for b in sorted_boxes)
        
        # Average confidence
        avg_conf = sum(b.confidence for b in boxes) / len(boxes)
        
        return BoundingBox(
            x=x_min,
            y=y_min,
            w=x_max - x_min,
            h=y_max - y_min,
            text=merged_text,
            confidence=avg_conf
        )
    
    def _link_prices_to_titles(self, classified: Dict, image_shape: Tuple) -> List[Dict]:
        """
        Link prices to titles based on spatial proximity
        Separates: old prices, current prices, and installment prices
        """
        # Merge nearby titles first
        titles = self._merge_nearby_titles(classified.get("titles", []))
        current_prices = classified.get("prices", [])
        old_prices = classified.get("old_prices", [])
        installment_prices = classified.get("installment_prices", [])
        discounts = classified.get("discounts", [])
        
        products = []
        
        for title in titles:
            product = {
                "title": title.text,
                "title_bbox": {
                    "x": title.x, "y": title.y, "w": title.w, "h": title.h
                },
                "title_confidence": round(title.confidence, 2),
                "old_price": None,
                "current_price": None,
                "installment": None,
                "discounts": []
            }
            
            # Find nearest prices (within reasonable vertical distance)
            # Use bottom_center of title since prices are typically below
            title_bottom = title.bottom_center
            max_distance = image_shape[0] * 0.5  # Within 50% of image height
            
            # Link OLD PRICE (closest)
            for price in old_prices:
                distance = self._calculate_distance(title_bottom, price.center)
                if distance < max_distance:
                    numeric_value = self._extract_price_value(price.text)
                    if numeric_value:
                        product["old_price"] = {
                            "raw_text": price.text,
                            "value": numeric_value,
                            "bbox": {"x": price.x, "y": price.y, "w": price.w, "h": price.h},
                            "confidence": round(price.confidence, 2),
                            "distance_from_title": round(distance, 2)
                        }
                        break  # Only one old price
            
            # Link CURRENT PRICE (closest)
            for price in current_prices:
                distance = self._calculate_distance(title_bottom, price.center)
                if distance < max_distance:
                    numeric_value = self._extract_price_value(price.text)
                    if numeric_value:
                        product["current_price"] = {
                            "raw_text": price.text,
                            "value": numeric_value,
                            "bbox": {"x": price.x, "y": price.y, "w": price.w, "h": price.h},
                            "confidence": round(price.confidence, 2),
                            "distance_from_title": round(distance, 2)
                        }
                        break  # Only one current price
            
            # Link INSTALLMENT PRICE (closest)
            for price in installment_prices:
                distance = self._calculate_distance(title_bottom, price.center)
                if distance < max_distance:
                    # Extract installment info: "21x R$ 185,44"
                    installment_match = re.search(r'(\d+)\s*x.*?([\d.,]+)', price.text)
                    if installment_match:
                        num_installments = int(installment_match.group(1))
                        installment_value = self._extract_price_value(installment_match.group(2))
                        
                        product["installment"] = {
                            "raw_text": price.text,
                            "installments": num_installments,
                            "value_per_installment": installment_value,
                            "total_value": installment_value * num_installments if installment_value else None,
                            "bbox": {"x": price.x, "y": price.y, "w": price.w, "h": price.h},
                            "confidence": round(price.confidence, 2),
                            "distance_from_title": round(distance, 2)
                        }
                        break  # Only one installment option
                    
                    product["prices"].append({
                        "raw_text": price.text,
                        "value": numeric_value,
                        "bbox": {
                            "x": price.x, "y": price.y, "w": price.w, "h": price.h
                        },
                        "confidence": round(price.confidence, 2),
                        "distance_from_title": round(distance, 2)
                    })
            
            # Find related discounts
            for discount in discounts:
                distance = self._calculate_distance(title_bottom, discount.center)
                
                if distance < max_distance:
                    product["discounts"].append({
                        "text": discount.text,
                        "bbox": {
                            "x": discount.x, "y": discount.y, "w": discount.w, "h": discount.h
                        },
                        "confidence": round(discount.confidence, 2)
                    })
            
            # Sort prices by distance (closest first)
            product["prices"].sort(key=lambda p: p["distance_from_title"])
            
            products.append(product)
        
        return products
    
    def _validate_and_normalize(self, products: List[Dict]) -> List[Dict]:
        """
        Validate and normalize product data
        """
        validated = []
        
        for product in products:
            # Ensure we have at least one price
            if not product["prices"]:
                continue
            
            # Normalize prices
            normalized_prices = []
            for price_info in product["prices"]:
                if price_info["value"] is not None:
                    normalized_prices.append({
                        "raw_text": price_info["raw_text"],
                        "value": price_info["value"],
                        "formatted": f"R$ {price_info['value']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'),
                        "confidence": price_info["confidence"],
                        "position": "primary" if normalized_prices == [] else "secondary"
                    })
            
            # Extract discount percentage
            discount_pct = None
            for discount in product.get("discounts", []):
                match = re.search(r'(\d+)\s*%', discount["text"])
                if match:
                    discount_pct = int(match.group(1))
                    break
            
            validated_product = {
                "product_name": product["title"],
                "confidence": product["title_confidence"],
                "prices": normalized_prices,
                "discount_percentage": discount_pct,
                "validation": {
                    "has_title": True,
                    "has_prices": len(normalized_prices) > 0,
                    "prices_count": len(normalized_prices),
                    "has_discount": discount_pct is not None
                }
            }
            
            validated.append(validated_product)
        
        return validated
    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _extract_price_value(self, text: str) -> Optional[float]:
        """
        Extract numeric price value from text
        
        Examples:
            "R$ 3.699" -> 3699.0
            "R$ 5.709" -> 5709.0
            "R$ 185,44" -> 185.44
            "21x R$ 185,44" -> 185.44
        """
        # Remove currency symbols, 'x' multiplier, and letters (keep numbers)
        # Handle both R$ and a$ (OCR mistake)
        cleaned = re.sub(r'[Rr]\$|[Aa]\$|x\s*', '', text)
        # Remove remaining letters except decimal separators
        cleaned = re.sub(r'[a-zA-Z]', '', cleaned).strip()
        
        # Handle Brazilian format (. for thousands, , for decimals)
        # 1. Both . and , : 3.699,00 -> 3699.00
        if '.' in cleaned and ',' in cleaned:
            cleaned = cleaned.replace('.', '').replace(',', '.')
        # 2. Only comma: 185,44 -> 185.44
        elif ',' in cleaned:
            cleaned = cleaned.replace(',', '.')
        # 3. Multiple dots: 1.234.567 -> 1234567
        elif cleaned.count('.') > 1:
            cleaned = cleaned.replace('.', '')
        # 4. Single dot with 3 digits after: 3.699 -> 3699 (Brazilian thousands separator)
        elif '.' in cleaned:
            parts = cleaned.split('.')
            if len(parts) == 2 and len(parts[1]) == 3:
                # Likely thousands separator: 3.699 = 3699
                cleaned = cleaned.replace('.', '')
            # Otherwise keep as is (decimal point)
        
        # Try to convert to float
        try:
            return float(cleaned)
        except ValueError:
            return None
