"""
Flyer Parser - Converts model output to product schema

Handles various output formats from multimodal transformers
and normalizes to consistent product structure
"""

import json
import re
from typing import List, Dict, Any, Optional


class FlyerParser:
    """
    Parses multimodal model output to structured product data
    
    Handles:
    - JSON output
    - Structured text
    - Semi-structured text
    - Mixed formats
    """
    
    def __init__(self):
        """Initialize Flyer Parser"""
        self.product_schema = {
            'product_name': str,
            'price': float,
            'discount': str,
            'brand': str,
            'confidence': float
        }
    
    def parse(self, raw_output: str) -> List[Dict[str, Any]]:
        """
        Parse raw model output to product list
        
        Args:
            raw_output: Raw text from multimodal model
        
        Returns:
            List of product dictionaries
        """
        
        # Try different parsing strategies
        products = None
        
        # Strategy 1: JSON parsing
        products = self._try_json_parse(raw_output)
        if products:
            return self._normalize_products(products)
        
        # Strategy 2: Structured text parsing
        products = self._try_structured_text_parse(raw_output)
        if products:
            return self._normalize_products(products)
        
        # Strategy 3: Pattern matching
        products = self._try_pattern_matching(raw_output)
        if products:
            return self._normalize_products(products)
        
        # Fallback: empty list
        print("[FlyerParser] Could not parse output, returning empty list")
        return []
    
    def _try_json_parse(self, text: str) -> Optional[List[Dict]]:
        """Try to parse as JSON"""
        
        try:
            # Try direct JSON parse
            data = json.loads(text)
            
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                if 'products' in data:
                    return data['products']
                elif 'items' in data:
                    return data['items']
                else:
                    return [data]
            
            return None
            
        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, text)
            
            for match in matches:
                try:
                    obj = json.loads(match)
                    if isinstance(obj, dict):
                        return [obj]
                except:
                    continue
            
            return None
    
    def _try_structured_text_parse(self, text: str) -> Optional[List[Dict]]:
        """Parse structured text format"""
        
        products = []
        
        # Look for product blocks
        # Example:
        # Product: Coffee
        # Price: R$ 13.99
        # Discount: 20%
        
        product_blocks = re.split(r'\n\s*\n', text)
        
        for block in product_blocks:
            product = {}
            
            # Extract fields
            name_match = re.search(r'(?:Product|Nome|Item):\s*(.+?)(?:\n|$)', block, re.IGNORECASE)
            price_match = re.search(r'(?:Price|Preço|Valor):\s*(?:R?\$?\s*)?(\d+[,.]?\d*)', block, re.IGNORECASE)
            discount_match = re.search(r'(?:Discount|Desconto):\s*(.+?)(?:\n|$)', block, re.IGNORECASE)
            brand_match = re.search(r'(?:Brand|Marca):\s*(.+?)(?:\n|$)', block, re.IGNORECASE)
            
            if name_match or price_match:
                product['name'] = name_match.group(1).strip() if name_match else 'Unknown'
                product['price'] = price_match.group(1).strip() if price_match else '0'
                product['discount'] = discount_match.group(1).strip() if discount_match else None
                product['brand'] = brand_match.group(1).strip() if brand_match else None
                
                products.append(product)
        
        return products if products else None
    
    def _try_pattern_matching(self, text: str) -> Optional[List[Dict]]:
        """Extract products using regex patterns"""
        
        products = []
        
        # Pattern 1: "Product - R$ 13.99 - 20% OFF"
        pattern1 = r'([A-Z][^-\n]+?)\s*-\s*(?:R?\$?\s*)?(\d+[,.]?\d*)\s*(?:-\s*(\d+%\s*OFF))?'
        matches = re.finditer(pattern1, text, re.IGNORECASE)
        
        for match in matches:
            products.append({
                'name': match.group(1).strip(),
                'price': match.group(2).strip(),
                'discount': match.group(3).strip() if match.group(3) else None
            })
        
        # Pattern 2: Line-by-line with price
        if not products:
            lines = text.split('\n')
            for line in lines:
                price_match = re.search(r'(?:R?\$?\s*)?(\d+[,.]?\d{2})', line)
                if price_match:
                    # Extract product name (text before price)
                    name_part = line[:price_match.start()].strip()
                    if name_part and len(name_part) > 3:
                        products.append({
                            'name': name_part,
                            'price': price_match.group(1)
                        })
        
        return products if products else None
    
    def _normalize_products(self, products: List[Dict]) -> List[Dict[str, Any]]:
        """Normalize product dictionaries to schema"""
        
        normalized = []
        
        for product in products:
            normalized_product = {
                'product_name': self._get_field(product, ['name', 'product_name', 'item', 'product'], 'Unknown Product'),
                'price': self._extract_price(product),
                'discount': self._get_field(product, ['discount', 'desconto', 'off'], None),
                'brand': self._get_field(product, ['brand', 'marca'], None),
                'confidence': self._get_field(product, ['confidence', 'score'], 0.8)
            }
            
            # Only include if has valid price
            if normalized_product['price'] > 0:
                normalized.append(normalized_product)
        
        return normalized
    
    def _get_field(self, product: Dict, field_names: List[str], default: Any) -> Any:
        """Get field from product dict, trying multiple names"""
        
        for field_name in field_names:
            if field_name in product:
                value = product[field_name]
                if value is not None and value != '':
                    return value
        
        return default
    
    def _extract_price(self, product: Dict) -> float:
        """Extract and normalize price value"""
        
        # Get price field
        price = self._get_field(product, ['price', 'preço', 'valor', 'cost'], 0)
        
        # Convert to float
        if isinstance(price, (int, float)):
            return float(price)
        
        if isinstance(price, str):
            # Remove currency symbols
            price_clean = re.sub(r'[R$\s]', '', price)
            
            # Handle Brazilian format (comma as decimal)
            if ',' in price_clean:
                price_clean = price_clean.replace('.', '').replace(',', '.')
            
            try:
                return float(price_clean)
            except ValueError:
                # Try to extract first number
                numbers = re.findall(r'\d+[.,]?\d*', price)
                if numbers:
                    num_str = numbers[0].replace(',', '.')
                    return float(num_str)
        
        return 0.0
    
    def validate_schema(self, products: List[Dict]) -> bool:
        """Validate that products match expected schema"""
        
        for product in products:
            for field, field_type in self.product_schema.items():
                if field not in product:
                    return False
                
                value = product[field]
                if value is not None and not isinstance(value, field_type):
                    return False
        
        return True
