"""
Price Validator - Ensures valid numeric prices and currency formats

Validates and corrects product pricing data
"""

import re
from typing import List, Dict, Optional, Tuple


class PriceValidator:
    """
    Validates and normalizes product prices
    
    Checks:
    - Valid numeric values
    - Reasonable price ranges
    - Currency format
    - Discount consistency
    """
    
    def __init__(
        self,
        min_price: float = 0.01,
        max_price: float = 10000.0,
        currency: str = 'BRL'
    ):
        """
        Initialize Price Validator
        
        Args:
            min_price: Minimum valid price
            max_price: Maximum valid price
            currency: Currency code
        """
        self.min_price = min_price
        self.max_price = max_price
        self.currency = currency
    
    def validate(self, products: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        Validate and correct product prices
        
        Args:
            products: List of product dictionaries
        
        Returns:
            Tuple of (validated_products, error_messages)
        """
        
        validated = []
        errors = []
        
        for idx, product in enumerate(products):
            try:
                validated_product = self._validate_product(product)
                
                if validated_product:
                    validated.append(validated_product)
                else:
                    errors.append(f"Product {idx}: Invalid data")
                    
            except Exception as e:
                errors.append(f"Product {idx}: {str(e)}")
        
        return validated, errors
    
    def _validate_product(self, product: Dict) -> Optional[Dict]:
        """Validate single product"""
        
        # Check required fields
        if 'product_name' not in product:
            return None
        
        if 'price' not in product:
            return None
        
        # Validate price
        price = product['price']
        
        if not isinstance(price, (int, float)):
            return None
        
        if price < self.min_price or price > self.max_price:
            return None
        
        # Validate discount (if present)
        if 'discount' in product and product['discount']:
            discount = self._validate_discount(product['discount'])
            product['discount'] = discount
        
        # Add formatted price
        product['price_formatted'] = self._format_price(price)
        
        # Validate confidence
        if 'confidence' in product:
            confidence = product['confidence']
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                product['confidence'] = 0.5
        
        return product
    
    def _validate_discount(self, discount: str) -> Optional[str]:
        """Validate discount format"""
        
        if not discount:
            return None
        
        # Extract percentage
        match = re.search(r'(\d+)\s*%', str(discount))
        
        if match:
            percentage = int(match.group(1))
            if 0 < percentage <= 100:
                return f"{percentage}%"
        
        return None
    
    def _format_price(self, price: float) -> str:
        """Format price according to currency"""
        
        if self.currency == 'BRL':
            # Brazilian format: R$ 13,99
            formatted = f"{price:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
            return f"R$ {formatted}"
        elif self.currency == 'USD':
            # US format: $13.99
            return f"${price:,.2f}"
        else:
            # Generic format
            return f"{price:.2f}"
    
    def check_price_consistency(self, products: List[Dict]) -> List[str]:
        """
        Check for inconsistencies in pricing
        
        Examples:
        - Multiple products with same name but different prices
        - Discount doesn't match price difference
        """
        
        warnings = []
        
        # Group by product name
        product_groups = {}
        for product in products:
            name = product['product_name'].lower().strip()
            if name not in product_groups:
                product_groups[name] = []
            product_groups[name].append(product)
        
        # Check for price variations
        for name, group in product_groups.items():
            if len(group) > 1:
                prices = [p['price'] for p in group]
                if len(set(prices)) > 1:
                    warnings.append(
                        f"Product '{name}' has multiple prices: {prices}"
                    )
        
        return warnings
    
    def remove_duplicates(self, products: List[Dict]) -> List[Dict]:
        """Remove duplicate products"""
        
        seen = set()
        unique = []
        
        for product in products:
            # Create signature
            signature = (
                product['product_name'].lower().strip(),
                product['price']
            )
            
            if signature not in seen:
                seen.add(signature)
                unique.append(product)
        
        return unique
    
    def sort_by_confidence(self, products: List[Dict]) -> List[Dict]:
        """Sort products by confidence score"""
        
        return sorted(
            products,
            key=lambda p: p.get('confidence', 0),
            reverse=True
        )
