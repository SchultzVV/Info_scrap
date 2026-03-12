"""
Simple Image OCR API
Single endpoint: POST /extract - receives image, returns text
Advanced endpoint: POST /analyze - advanced analysis with ROI, proximity, and validation
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pytesseract
from PIL import Image
import io

from analyzer import AdvancedImageAnalyzer
from ecommerce_parser import EcommerceParser
from product_extractor import ProductExtractor

app = FastAPI(
    title="Simple OCR API",
    description="Extract text from images with basic and advanced endpoints",
    version="2.0.0"
)

# Initialize advanced analyzer
analyzer = AdvancedImageAnalyzer()

# Initialize e-commerce parser
ecommerce_parser = EcommerceParser()

# Initialize product extractor (classical CV pipeline)
product_extractor = ProductExtractor()

@app.get("/health")
def health():
    """Health check"""
    return {"status": "healthy"}

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from image
    
    Args:
        file: Image file (PNG, JPG, etc)
    
    Returns:
        JSON with extracted text
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Extract text with Tesseract
        text = pytesseract.image_to_string(image, lang='por+eng')
        
        # Clean text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        return {
            "success": True,
            "text": text,
            "lines": lines,
            "line_count": len(lines)
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Advanced image analysis with automatic detection:
    
    - For product images: ROI detection, proximity analysis, validation
    - For e-commerce pages: Intelligent parsing with price inference
    
    Automatically detects:
    - Product cards (isolated products)
    - E-commerce screenshots (Mercado Livre, Amazon, etc)
    
    Args:
        file: Image file (PNG, JPG, etc)
    
    Returns:
        JSON with structured product information (simplified for end users)
    """
    try:
        # Read image data
        contents = await file.read()
        
        # Try advanced ROI-based analysis first
        result = analyzer.analyze_image(contents)
        
        # Check if ROI analysis found prices
        has_prices = (
            result.get('product', {}).get('current_price') is not None or
            result.get('product', {}).get('old_price') is not None
        )
        
        # If no prices found, try e-commerce parser
        if not has_prices:
            # Extract text for e-commerce parsing
            image = Image.open(io.BytesIO(contents))
            text = pytesseract.image_to_string(image, lang='por+eng')
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            # Parse with e-commerce parser
            ecommerce_result = ecommerce_parser.parse(text, lines)
            
            # If e-commerce parser found prices, use that result
            if ecommerce_result.get('product', {}).get('current_price'):
                result = ecommerce_result
        
        # Simplify response for end users
        product = result.get('product', {})
        simplified_response = {
            "success": True,
            "product": {
                "title": product.get('title'),
                "old_price": product.get('old_price'),
                "current_price": product.get('current_price'),
                "installment": product.get('installment'),
                "discount": product.get('discount'),
            }
        }
        
        # Add shipping if available (from e-commerce parser)
        if product.get('shipping'):
            simplified_response['product']['shipping'] = product.get('shipping')
        
        return simplified_response
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )


@app.post("/product")
async def extract_product(file: UploadFile = File(...)):
    """
    Structured product extraction using classical CV.

    Pipeline:
      1. ROI detection  – find the product info panel via text-density contours
      2. Price CV       – strikethrough detection + OCR-artifact + value ordering
                          → identifies promotional price, old price, or regular price
      3. Title          – longest high-scoring text line in the upper ROI region
      4. Stock/avail.   – regex patterns on full-image OCR text

    Returns:
        {
          "title"      : str,
          "price"      : "R$ NNNN.NN",
          "oldPrice"   : "R$ NNNN.NN" | null,
          "disponivel" : bool,
          "via_webhook": false,
          "stock"      : int | null
        }
    """
    try:
        contents = await file.read()
        result = product_extractor.extract(contents)
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
