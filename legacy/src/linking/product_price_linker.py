"""
Product-Price Linker

Links product information with prices and discounts using spatial reasoning
Post-processes and normalizes commercial data
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Product:
    """Structured product data"""
    product_name: str
    brand: Optional[str]
    price: float
    price_raw: str
    discount: Optional[str]
    bounding_box: Tuple[int, int, int, int]
    confidence: float


class ProductPriceLinker:
    """
    Links products with prices and discounts
    
    Post-processing rules:
    1. Extract numeric price from text
    2. Normalize currency format
    3. Associate product with nearest price
    4. Extract discount percentage
    """
    
    def __init__(self):
        """Initialize Product-Price Linker"""
        pass
    
    def link(
        self,
        structured_data: List[Dict]
    ) -> List[Product]:
        """
        Link and normalize product data
        
        Args:
            structured_data: Raw structured data from LayoutLM
        
        Returns:
            List of Product objects with normalized data
        """
        products = []
        
        for data in structured_data:
            try:
                product = self._process_product(data)
                if product:
                    products.append(product)
            except Exception as e:
                print(f"[ProductPriceLinker] Error processing product: {e}")
                continue
        
        return products
    
    def _process_product(self, data: Dict) -> Optional[Product]:
        """Process single product data"""
        
        # Extract product name
        product_name = data.get('product_name', '').strip()
        if not product_name:
            product_name = "Unknown Product"
        
        # Extract brand
        brand = data.get('brand_text', '').strip() if 'brand_text' in data else None
        
        # Extract and normalize price
        price_text = data.get('price_text', '')
        price = self._extract_price(price_text)
        
        # Extract discount
        discount_text = data.get('discount_text', '') if 'discount_text' in data else None
        discount = self._extract_discount(discount_text) if discount_text else None
        
        # Get bounding box (use product bbox as main)
        bbox = data.get('product_bbox', (0, 0, 0, 0))
        
        # Calculate confidence
        confidence = data.get('confidence', 0.5)
        
        # Create product object
        if price is not None:
            product = Product(
                product_name=product_name,
                brand=brand,
                price=price,
                price_raw=price_text,
                discount=discount,
                bounding_box=bbox,
                confidence=confidence
            )
            return product
        
        return None
    
    def _extract_price(self, text: str) -> Optional[float]:
        """
        Extract numeric price from text
        
        Supports formats:
        - R$ 36,99
        - 36.99
        - 3699 (assumes last 2 digits are cents)
        """
        if not text:
            return None
        
        # Remove spaces
        text = text.replace(' ', '')
        
        # Patterns
        patterns = [
            r'R?\$?\s*(\d+)[,.](\d{2})',  # R$ 36,99 or 36.99
            r'(\d{2,})(\d{2})',            # 3699 (min 4 digits)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) == 2:
                    reais = match.group(1)
                    centavos = match.group(2)
                    return float(f"{reais}.{centavos}")
                else:
                    return float(match.group(1))
        
        # Try simple number
        match = re.search(r'(\d+)', text)
        if match:
            num = match.group(1)
            if len(num) >= 3:
                # Assume last 2 digits are cents
                reais = num[:-2]
                centavos = num[-2:]
                return float(f"{reais}.{centavos}")
            else:
                return float(num)
        
        return None
    
    def _extract_discount(self, text: str) -> Optional[str]:
        """
        Extract discount information
        
        Supports:
        - 20%
        - -20%
        - DESCONTO 20%
        """
        if not text:
            return None
        
        # Look for percentage
        match = re.search(r'(-?\d+)\s*%', text)
        if match:
            return f"{match.group(1)}%"
        
        # Look for "OFF" patterns
        match = re.search(r'(\d+)\s*OFF', text, re.IGNORECASE)
        if match:
            return f"{match.group(1)}%"
        
        return text.strip() if text else None
