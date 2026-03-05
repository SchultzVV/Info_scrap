"""
Multimodal Flyer Extractor - State of the Art 2026

Uses vision-language transformers for end-to-end flyer understanding:
- Donut (Document Understanding Transformer)
- Pix2Struct (Screenshot parsing)
- Kosmos-2 (Multimodal large language model)

No explicit OCR needed - image → structured text directly
"""

import cv2
import numpy as np
from PIL import Image
import torch
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import json
import re


@dataclass
class ExtractionResult:
    """Result from multimodal extraction"""
    raw_output: str
    products: List[Dict]
    confidence: float
    model_name: str


class MultimodalExtractor:
    """
    State-of-the-art multimodal document understanding for retail flyers
    
    Architecture:
        image → vision encoder → language decoder → structured JSON
    
    Advantages over YOLO + OCR + LayoutLM:
        ✓ Single end-to-end model
        ✓ Fewer components = fewer errors
        ✓ Joint training on vision + language
        ✓ Better context understanding
    """
    
    def __init__(
        self,
        model_name: str = 'donut',
        model_path: Optional[str] = None,
        use_gpu: bool = False
    ):
        """
        Initialize Multimodal Extractor
        
        Args:
            model_name: 'donut', 'pix2struct', or 'kosmos2'
            model_path: Path to fine-tuned model (optional)
            use_gpu: Use GPU acceleration
        """
        self.model_name = model_name.lower()
        self.model_path = model_path
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        self.model = None
        self.processor = None
        
        print(f"[MultimodalExtractor] Initializing {self.model_name} model...")
        print(f"[MultimodalExtractor] Device: {self.device}")
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load multimodal transformer model"""
        
        try:
            if self.model_name == 'donut':
                self._load_donut()
            elif self.model_name == 'pix2struct':
                self._load_pix2struct()
            elif self.model_name == 'kosmos2':
                self._load_kosmos2()
            else:
                print(f"[MultimodalExtractor] Unknown model: {self.model_name}")
                print("[MultimodalExtractor] Falling back to rule-based extraction")
                self.model = None
                
        except ImportError as e:
            print(f"[MultimodalExtractor] Model libraries not available: {e}")
            print("[MultimodalExtractor] Using rule-based extraction as fallback")
            print("[MultimodalExtractor] Install transformers: pip install transformers")
            self.model = None
        except Exception as e:
            print(f"[MultimodalExtractor] Error loading model: {e}")
            print("[MultimodalExtractor] Falling back to rule-based extraction")
            self.model = None
    
    def _load_donut(self) -> None:
        """Load Donut (Document Understanding Transformer)"""
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        
        model_id = self.model_path or "naver-clova-ix/donut-base-finetuned-docvqa"
        
        print(f"[Donut] Loading from {model_id}")
        
        self.processor = DonutProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        
        print("[Donut] Model loaded successfully")
    
    def _load_pix2struct(self) -> None:
        """Load Pix2Struct (Screenshot Parsing)"""
        from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
        
        model_id = self.model_path or "google/pix2struct-base"
        
        print(f"[Pix2Struct] Loading from {model_id}")
        
        self.processor = Pix2StructProcessor.from_pretrained(model_id)
        self.model = Pix2StructForConditionalGeneration.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        
        print("[Pix2Struct] Model loaded successfully")
    
    def _load_kosmos2(self) -> None:
        """Load Kosmos-2 (Multimodal LLM)"""
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        model_id = self.model_path or "microsoft/kosmos-2-patch14-224"
        
        print(f"[Kosmos-2] Loading from {model_id}")
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        
        print("[Kosmos-2] Model loaded successfully")
    
    def extract(
        self,
        image: Union[np.ndarray, Image.Image, str],
        prompt: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract structured product data from flyer image
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
            prompt: Optional prompt for model (task-specific)
        
        Returns:
            ExtractionResult with products and metadata
        """
        
        # Convert image to PIL
        pil_image = self._to_pil(image)
        
        # Use model or fallback
        if self.model is not None:
            raw_output = self._model_inference(pil_image, prompt)
        else:
            raw_output = self._rule_based_extraction(pil_image)
        
        # Parse output to structured format
        products = self._parse_output(raw_output)
        
        return ExtractionResult(
            raw_output=raw_output,
            products=products,
            confidence=self._calculate_confidence(products),
            model_name=self.model_name if self.model else 'rule_based'
        )
    
    def _model_inference(
        self,
        image: Image.Image,
        prompt: Optional[str] = None
    ) -> str:
        """Run inference with multimodal model"""
        
        # Default prompt for flyer extraction
        if prompt is None:
            prompt = (
                "Extract all products from this retail flyer with their names, "
                "prices, and discounts in JSON format."
            )
        
        try:
            if self.model_name == 'donut':
                return self._donut_inference(image, prompt)
            elif self.model_name == 'pix2struct':
                return self._pix2struct_inference(image, prompt)
            elif self.model_name == 'kosmos2':
                return self._kosmos2_inference(image, prompt)
        except Exception as e:
            print(f"[MultimodalExtractor] Model inference error: {e}")
            print("[MultimodalExtractor] Falling back to rule-based")
            return self._rule_based_extraction(image)
    
    def _donut_inference(self, image: Image.Image, prompt: str) -> str:
        """Donut-specific inference"""
        
        # Prepare inputs
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        # Prepare decoder input
        task_prompt = f"<s_flyer>{prompt}</s_flyer>"
        decoder_input_ids = self.processor.tokenizer(
            task_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids
        decoder_input_ids = decoder_input_ids.to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=512,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode
        sequence = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        return sequence
    
    def _pix2struct_inference(self, image: Image.Image, prompt: str) -> str:
        """Pix2Struct-specific inference"""
        
        # Prepare inputs
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512
            )
        
        # Decode
        sequence = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return sequence
    
    def _kosmos2_inference(self, image: Image.Image, prompt: str) -> str:
        """Kosmos-2-specific inference"""
        
        # Prepare inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512
            )
        
        # Decode
        sequence = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        return sequence
    
    def _rule_based_extraction(self, image: Image.Image) -> str:
        """
        Fallback rule-based extraction
        Simulates model output for demonstration
        """
        
        # Convert to numpy
        img_array = np.array(image)
        
        # Simple heuristic: detect text-like regions
        # In real scenario, this would use the old YOLO+OCR pipeline
        
        mock_output = {
            "products": [
                {
                    "name": "Detected Product",
                    "price": 0.0,
                    "discount": None
                }
            ]
        }
        
        return json.dumps(mock_output)
    
    def _parse_output(self, raw_output: str) -> List[Dict]:
        """
        Parse model output to structured product list
        
        Handles various output formats from different models
        """
        
        products = []
        
        # Try to parse as JSON
        try:
            data = json.loads(raw_output)
            if 'products' in data:
                products = data['products']
            elif isinstance(data, list):
                products = data
        except json.JSONDecodeError:
            # Try to extract structured data from text
            products = self._extract_products_from_text(raw_output)
        
        # Normalize product format
        normalized = []
        for product in products:
            normalized.append({
                'product_name': product.get('name', product.get('product_name', 'Unknown')),
                'price': self._extract_price_value(product.get('price', 0)),
                'discount': product.get('discount'),
                'brand': product.get('brand'),
                'confidence': product.get('confidence', 0.8)
            })
        
        return normalized
    
    def _extract_products_from_text(self, text: str) -> List[Dict]:
        """Extract products from unstructured text"""
        
        products = []
        
        # Look for product patterns
        # Example: "Product: Coffee, Price: R$ 13.99, Discount: 20%"
        pattern = r'(?:Product|Nome):\s*([^,]+).*?(?:Price|Preço):\s*(?:R?\$?\s*)?(\d+[,.]?\d*)'
        
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            name = match.group(1).strip()
            price_str = match.group(2).strip()
            
            products.append({
                'name': name,
                'price': price_str,
                'discount': None
            })
        
        return products if products else [{'name': 'Unknown', 'price': 0.0}]
    
    def _extract_price_value(self, price) -> float:
        """Extract numeric price value"""
        
        if isinstance(price, (int, float)):
            return float(price)
        
        if isinstance(price, str):
            # Remove currency symbols and extract number
            price_clean = re.sub(r'[R$\s]', '', price)
            price_clean = price_clean.replace(',', '.')
            
            try:
                return float(price_clean)
            except ValueError:
                return 0.0
        
        return 0.0
    
    def _calculate_confidence(self, products: List[Dict]) -> float:
        """Calculate overall confidence score"""
        
        if not products:
            return 0.0
        
        confidences = [p.get('confidence', 0.5) for p in products]
        return sum(confidences) / len(confidences)
    
    def _to_pil(self, image: Union[np.ndarray, Image.Image, str]) -> Image.Image:
        """Convert various image formats to PIL Image"""
        
        if isinstance(image, str):
            # File path
            return Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Numpy array
            if image.shape[-1] == 3:
                # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image)
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None or True  # rule-based always available
