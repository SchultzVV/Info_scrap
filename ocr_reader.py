import cv2
import numpy as np
import easyocr
import re
from typing import Dict, List
import pytesseract
from PIL import Image


class OCRReader:
    """
    Classe para leitura de texto (preços) usando OCR
    """
    
    def __init__(self, use_easyocr: bool = True, languages: List[str] = None):
        """
        Inicializa o leitor OCR
        
        Args:
            use_easyocr: Se True usa EasyOCR, se False usa Tesseract
            languages: Lista de idiomas para reconhecimento
        """
        self.use_easyocr = use_easyocr
        self.languages = languages or ['pt', 'en']
        self.reader = None
        
        if self.use_easyocr:
            print(f"Inicializando EasyOCR com idiomas: {self.languages}")
            self.reader = easyocr.Reader(self.languages, gpu=False)
            print("EasyOCR carregado com sucesso!")
        else:
            print("Usando Tesseract OCR")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Pré-processa a imagem para melhorar OCR
        
        Args:
            image: Imagem em formato numpy array (BGR)
        
        Returns:
            Imagem pré-processada
        """
        # Converte para escala de cinza
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Redimensiona se muito pequeno
        height, width = gray.shape
        if height < 50 or width < 50:
            scale = max(50 / height, 50 / width)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Aplica threshold adaptativo
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Remove ruído
        denoised = cv2.medianBlur(thresh, 3)
        
        return denoised
    
    def read_text(self, image: np.ndarray) -> Dict:
        """
        Lê texto da imagem usando OCR
        
        Args:
            image: Imagem recortada contendo o preço
        
        Returns:
            Dicionário com texto, confiança e valor do preço extraído
        """
        # Pré-processa imagem
        processed = self.preprocess_image(image)
        
        if self.use_easyocr:
            result = self._read_with_easyocr(processed)
        else:
            result = self._read_with_tesseract(processed)
        
        # Extrai valor do preço
        price_value = self._extract_price(result['text'])
        result['price_value'] = price_value
        
        return result
    
    def _read_with_easyocr(self, image: np.ndarray) -> Dict:
        """Lê texto usando EasyOCR"""
        results = self.reader.readtext(image)
        
        if not results:
            return {
                'text': '',
                'confidence': 0.0,
                'raw_results': []
            }
        
        # Concatena todos os textos detectados
        texts = []
        confidences = []
        
        for detection in results:
            bbox, text, conf = detection
            texts.append(text)
            confidences.append(conf)
        
        full_text = ' '.join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'text': full_text,
            'confidence': avg_confidence,
            'raw_results': results
        }
    
    def _read_with_tesseract(self, image: np.ndarray) -> Dict:
        """Lê texto usando Tesseract"""
        # Converte para PIL Image
        pil_image = Image.fromarray(image)
        
        # Configuração para números e símbolos
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789R$,.%'
        
        # Extrai texto
        text = pytesseract.image_to_string(pil_image, config=custom_config)
        
        # Extrai dados detalhados
        data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        
        # Calcula confiança média
        confidences = [int(conf) for conf in data['conf'] if conf != '-1']
        avg_confidence = sum(confidences) / len(confidences) / 100 if confidences else 0.0
        
        return {
            'text': text.strip(),
            'confidence': avg_confidence,
            'raw_results': data
        }
    
    def _extract_price(self, text: str) -> float:
        """
        Extrai valor numérico do preço do texto
        
        Args:
            text: Texto contendo o preço
        
        Returns:
            Valor do preço como float (ou None se não encontrado)
        """
        # Remove espaços
        text = text.replace(' ', '')
        
        # Padrões de preço (R$ 36,99 ou 3699 ou 36.99)
        patterns = [
            r'R?\$?\s*(\d+)[,.](\d{2})',  # R$ 36,99 ou 36.99
            r'(\d{2})(\d{2})',             # 3699
            r'(\d+)',                       # 36 ou 3699
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) == 2:
                    # Tem parte inteira e decimal
                    reais = match.group(1)
                    centavos = match.group(2)
                    return float(f"{reais}.{centavos}")
                else:
                    # Só número
                    num = match.group(1)
                    if len(num) >= 3:
                        # Assume últimos 2 dígitos são centavos
                        reais = num[:-2]
                        centavos = num[-2:]
                        return float(f"{reais}.{centavos}")
                    else:
                        return float(num)
        
        return None
    
    def is_loaded(self) -> bool:
        """Verifica se o OCR está carregado"""
        if self.use_easyocr:
            return self.reader is not None
        return True  # Tesseract não precisa carregar modelo
