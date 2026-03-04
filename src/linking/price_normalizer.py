"""
Price Normalizer

Normalizes price formats and currency
"""

import re
from typing import Optional


class PriceNormalizer:
    """
    Normalizes price data to standard format
    
    Handles:
    - Currency symbols (R$, $, etc.)
    - Decimal separators (, or .)
    - Thousand separators
    """
    
    def __init__(self, currency: str = 'BRL'):
        """
        Initialize Price Normalizer
        
        Args:
            currency: Currency code (BRL, USD, etc.)
        """
        self.currency = currency
    
    def normalize(self, price_text: str) -> Optional[float]:
        """
        Normalize price text to float
        
        Args:
            price_text: Raw price text
        
        Returns:
            Normalized price as float
        """
        if not price_text:
            return None
        
        # Remove currency symbols
        text = re.sub(r'[R$\s]', '', price_text)
        
        # Handle Brazilian format (comma as decimal)
        if ',' in text:
            # Check if it's thousand separator or decimal
            parts = text.split(',')
            if len(parts) == 2 and len(parts[1]) == 2:
                # Decimal separator
                text = text.replace('.', '').replace(',', '.')
        
        # Extract number
        try:
            return float(text)
        except ValueError:
            return None
    
    def format_price(self, value: float, currency_symbol: bool = True) -> str:
        """
        Format price for display
        
        Args:
            value: Price value
            currency_symbol: Include currency symbol
        
        Returns:
            Formatted price string
        """
        if self.currency == 'BRL':
            formatted = f"{value:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
            if currency_symbol:
                return f"R$ {formatted}"
            return formatted
        else:
            formatted = f"{value:,.2f}"
            if currency_symbol:
                return f"$ {formatted}"
            return formatted
