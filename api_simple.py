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

app = FastAPI(
    title="Simple OCR API",
    description="Extract text from images with basic and advanced endpoints",
    version="2.0.0"
)

# Initialize advanced analyzer
analyzer = AdvancedImageAnalyzer()

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
    Advanced image analysis with:
    - ROI (Region of Interest) detection
    - Text extraction with bounding boxes
    - Price-title proximity analysis
    - Validation and normalization
    - Structured product information
    
    Args:
        file: Image file (PNG, JPG, etc)
    
    Returns:
        JSON with structured product information:
        - ROIs detected in the image
        - Products with linked prices
        - Validation status
        - Normalized price values
    """
    try:
        # Read image data
        contents = await file.read()
        
        # Run advanced analysis
        result = analyzer.analyze_image(contents)
        
        return result
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
