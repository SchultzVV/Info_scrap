"""
E-commerce Parser - Extrai informações de produtos de páginas web
"""
import re
from typing import Dict, Optional, List


class EcommerceParser:
    """Parser especializado para páginas de e-commerce"""
    
    # Palavras de interface a serem filtradas
    INTERFACE_KEYWORDS = [
        'buscar', 'seguir', 'seguidor', 'cookies', 'aceitar', 'configurar',
        'para você', 'mais vendidos', 'ofertas', 'menu', 'carrinho',
        'entrar', 'cadastrar', 'categoria', 'filtrar', 'ordenar'
    ]
    
    def parse(self, text: str, lines: List[str]) -> Dict:
        """
        Parse texto extraído de página de e-commerce
        
        Args:
            text: Texto completo
            lines: Linhas separadas
            
        Returns:
            Dict com informações estruturadas
        """
        # Filtrar linhas de interface
        product_lines = self._filter_interface_lines(lines)
        
        # Extrair componentes
        title = self._extract_title(product_lines)
        prices = self._extract_prices(text, lines)
        discount = self._extract_discount(text, lines)
        installment = self._extract_installment(text, lines)
        shipping = self._extract_shipping(text, lines)
        
        # Determinar preços antigo e atual
        old_price = None
        current_price = None
        
        if len(prices) >= 2:
            # Se tem 2+ preços, o maior é o antigo
            sorted_prices = sorted(prices, key=lambda p: p['value'], reverse=True)
            old_price = sorted_prices[0]
            current_price = sorted_prices[1]
        elif len(prices) == 1:
            current_price = prices[0]
            
            # INFERIR preço antigo usando desconto (se disponível)
            # Fórmula: old_price = current_price / (1 - discount/100)
            if discount and current_price:
                discount_decimal = discount['percentage'] / 100.0
                inferred_old_value = current_price['value'] / (1 - discount_decimal)
                
                old_price = {
                    "raw_text": f"(inferido de {discount['percentage']}% OFF)",
                    "value": round(inferred_old_value, 2),
                    "formatted": self._format_brl_price(inferred_old_value),
                    "inferred": True,
                    "confidence": 0.85
                }
        
        # Montar resposta estruturada
        product = {
            "title": title,
            "old_price": old_price,
            "current_price": current_price,
            "installment": installment,
            "discount": discount,
            "shipping": shipping,
            "validation": {
                "has_title": bool(title),
                "has_old_price": bool(old_price),
                "has_current_price": bool(current_price),
                "has_installment": bool(installment),
                "has_discount": bool(discount),
                "has_free_shipping": bool(shipping and 'grátis' in shipping.lower())
            }
        }
        
        return {
            "success": True,
            "product": product,
            "debug": {
                "filtered_lines": product_lines,
                "all_prices_found": prices,
                "raw_text_length": len(text)
            }
        }
    
    def _filter_interface_lines(self, lines: List[str]) -> List[str]:
        """Remove linhas com palavras de interface"""
        filtered = []
        for line in lines:
            line_lower = line.lower()
            # Pular linhas vazias
            if not line.strip():
                continue
            # Pular se contém palavras de interface
            if any(keyword in line_lower for keyword in self.INTERFACE_KEYWORDS):
                continue
            # Pular se é muito curta (< 5 caracteres)
            if len(line.strip()) < 5:
                continue
            filtered.append(line)
        return filtered
    
    def _extract_title(self, lines: List[str]) -> Optional[str]:
        """
        Extrai título do produto
        Heurística: linha mais longa que contenha palavras de produto
        """
        if not lines:
            return None
        
        # Procurar linha com características de título de produto
        candidates = []
        for line in lines:
            # Títulos geralmente têm:
            # - Mais de 15 caracteres
            # - Letras maiúsculas e minúsculas
            # - Pode ter números (tamanho, modelo)
            if len(line) > 15 and any(c.isupper() for c in line):
                candidates.append(line)
        
        if candidates:
            # Retorna a linha mais longa
            return max(candidates, key=len)
        
        # Fallback: primeira linha com mais de 10 chars
        for line in lines:
            if len(line) > 10:
                return line
        
        return None
    
    def _extract_prices(self, text: str, lines: List[str]) -> List[Dict]:
        """
        Extrai todos os preços em formato brasileiro
        Padrões: R$ 123,45 | R$123.456,78 | 123,45
        Também detecta preços riscados mal lidos (como "816728" = R$ 167,28)
        """
        prices = []
        
        # Regex para preços brasileiros
        # Captura: R$ 1.234,56 ou 1.234,56 ou 1234,56 ou 1234
        patterns = [
            r'R\$\s*(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)',  # R$ 1.234,56
            r'R\$\s*(\d+(?:,\d{2})?)',                    # R$ 124,50 ou R$ 124
            r'(?<![\d.,])(\d{1,3}(?:\.\d{3})+,\d{2})(?![\d.,])',  # 1.234,56 (isolado)
            r'(?<![\d.,])(\d+,\d{2})(?![\d.,])',          # 124,50 (isolado)
        ]
        
        seen_values = set()
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                price_str = match.group(1) if match.lastindex else match.group(0)
                
                # Limpar e converter
                value = self._parse_brl_price(price_str)
                
                # Validar: preço deve estar entre R$ 0,01 e R$ 1.000.000
                if value and 0.01 <= value <= 1000000 and value not in seen_values:
                    seen_values.add(value)
                    prices.append({
                        "raw_text": match.group(0),
                        "value": value,
                        "formatted": self._format_brl_price(value)
                    })
        
        # Detectar números "órfãos" que podem ser preços riscados mal lidos
        # Exemplo: "816728," entre título e preço atual = R$ 167,28
        orphan_prices = self._extract_orphan_prices(lines, seen_values)
        prices.extend(orphan_prices)
        
        # Ordenar por valor (maior primeiro)
        return sorted(prices, key=lambda p: p['value'], reverse=True)
    
    def _extract_discount(self, text: str, lines: List[str]) -> Optional[Dict]:
        """
        Extrai desconto
        Padrões: 25% OFF | 25% | Desconto de 25% | 25%OFF (sem espaço)
        """
        # Procurar percentual + OFF (com ou sem espaço)
        patterns = [
            r'(\d+)\s*%\s*(?:OFF|off|Off)',  # 25% OFF
            r'(\d+)%(?:OFF|off|Off)',         # 25%OFF (sem espaço)
            r'(?:desconto|desc\.?)\s+(?:de\s+)?(\d+)\s*%',  # Desconto de 25%
        ]
        
        for pattern in patterns:
            # Buscar no texto completo primeiro
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                percentage = int(match.group(1))
                # Validar: desconto entre 1% e 99%
                if 1 <= percentage <= 99:
                    return {
                        "raw_text": match.group(0),
                        "percentage": percentage
                    }
            
            # Buscar nas linhas também
            for line in lines:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    percentage = int(match.group(1))
                    if 1 <= percentage <= 99:
                        return {
                            "raw_text": match.group(0),
                            "percentage": percentage
                        }
        
        return None
    
    def _extract_installment(self, text: str, lines: List[str]) -> Optional[Dict]:
        """
        Extrai parcelamento
        Padrões: 12x R$ 10,50 | 12x de 10,50 | em 12x
        """
        # Padrão: número + x + valor
        pattern = r'(\d+)\s*x\s*(?:de\s*)?(?:R\$\s*)?(\d+[.,]\d{2})'
        
        for line in lines:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                num_installments = int(match.group(1))
                value_str = match.group(2)
                value = self._parse_brl_price(value_str)
                
                if value and 1 <= num_installments <= 48:
                    total = num_installments * value
                    return {
                        "raw_text": match.group(0),
                        "installments": num_installments,
                        "value_per_installment": value,
                        "total_value": total,
                        "formatted_per_installment": self._format_brl_price(value),
                        "formatted_total": self._format_brl_price(total)
                    }
        
        return None
    
    def _extract_shipping(self, text: str, lines: List[str]) -> Optional[str]:
        """Extrai informação de frete"""
        shipping_keywords = ['frete', 'entrega', 'envio']
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in shipping_keywords):
                return line.strip()
        
        return None
    
    def _extract_orphan_prices(self, lines: List[str], seen_values: set) -> List[Dict]:
        """
        Detecta números "órfãos" que podem ser preços riscados mal lidos pelo OCR
        
        Exemplos:
        - "816728," → R$ 167,28 (remove primeiro dígito, adiciona vírgula)
        - "5709" → R$ 57,09 (adiciona vírgula)
        - "16728" → R$ 167,28
        
        Heurística:
        - Número isolado com 4-7 dígitos
        - Não tem R$ na frente
        - Aparece entre o título e o preço atual
        """
        orphan_prices = []
        
        for line in lines:
            # Procurar números isolados com 4-7 dígitos (possivelmente com vírgula no final)
            # Exemplo: "816728," ou "5709" ou "16728"
            pattern = r'^\s*(\d{4,7}),?\s*$'
            match = re.match(pattern, line.strip())
            
            if match:
                number_str = match.group(1)
                number = int(number_str)
                
                # Tentar diferentes interpretações
                candidates = []
                
                # Interpretação 1: Remove primeiro dígito se > 6 dígitos
                # "816728" → "16728" → 167.28
                if len(number_str) >= 6:
                    without_first = number_str[1:]
                    if len(without_first) >= 4:
                        # Últimos 2 dígitos são centavos
                        value = int(without_first[:-2]) + int(without_first[-2:]) / 100.0
                        if 10 <= value <= 100000:  # Validar range razoável
                            candidates.append(value)
                
                # Interpretação 2: Últimos 2 dígitos são centavos
                # "5709" → 57.09 | "16728" → 167.28
                if len(number_str) >= 4:
                    value = int(number_str[:-2]) + int(number_str[-2:]) / 100.0
                    if 10 <= value <= 100000:
                        candidates.append(value)
                
                # Interpretação 3: Número inteiro (sem centavos)
                # "5709" → 5709.00
                if 10 <= number <= 100000:
                    candidates.append(float(number))
                
                # Usar o candidato que faz mais sentido (não duplicado)
                for value in candidates:
                    if value not in seen_values:
                        seen_values.add(value)
                        orphan_prices.append({
                            "raw_text": match.group(0),
                            "value": value,
                            "formatted": self._format_brl_price(value),
                            "orphan": True,  # Flag indicando que foi recuperado
                            "original": number_str
                        })
                        break  # Usa apenas primeira interpretação válida
        
        return orphan_prices
    
    def _parse_brl_price(self, price_str: str) -> Optional[float]:
        """
        Converte string de preço brasileiro para float
        Exemplos: "1.234,56" -> 1234.56 | "124" -> 124.0
        """
        try:
            # Remove R$ e espaços
            cleaned = price_str.replace('R$', '').strip()
            
            # Se tem vírgula, assume formato brasileiro (1.234,56)
            if ',' in cleaned:
                # Remove pontos (separador de milhar) e troca vírgula por ponto
                cleaned = cleaned.replace('.', '').replace(',', '.')
            # Se tem apenas ponto, pode ser 1234.56 (já correto) ou 1.234 (milhar)
            elif '.' in cleaned:
                parts = cleaned.split('.')
                # Se última parte tem 2 dígitos, é decimal (1234.56)
                if len(parts[-1]) == 2:
                    pass  # Já está correto
                # Senão, é separador de milhar (1.234)
                else:
                    cleaned = cleaned.replace('.', '')
            
            return float(cleaned)
        except (ValueError, AttributeError):
            return None
    
    def _format_brl_price(self, value: float) -> str:
        """Formata valor como preço brasileiro"""
        return f"R$ {value:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
