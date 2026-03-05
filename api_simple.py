"""
Simple Image OCR API
Single endpoint: POST /extract - receives image, returns text
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pytesseract
from PIL import Image
import io

app = FastAPI(
    title="Simple OCR API",
    description="Extract text from images",
    version="1.0.0"
)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
