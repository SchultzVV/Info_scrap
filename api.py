"""
Production FastAPI Server for Retail Flyer Understanding System

RESTful API for extracting structured promotional data from flyer images
using YOLO + OCR + LayoutLM pipeline
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
from typing import Optional
import io
from datetime import datetime

from src.pipeline import FlyerPipeline

# Initialize FastAPI app
app = FastAPI(
    title="Retail Flyer Understanding System",
    description="Production-grade API for extracting structured promotional product data from retail flyer images using YOLO + OCR + LayoutLM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
print("[API] Initializing Flyer Understanding Pipeline...")
pipeline = FlyerPipeline(
    yolo_model_path='yolov8n.pt',
    yolo_confidence=0.25,
    ocr_lang='pt',
    use_gpu=False
)
print("[API] Pipeline ready!")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Retail Flyer Understanding System",
        "version": "1.0.0",
        "description": "Production-grade pipeline for extracting structured promotional data",
        "architecture": {
            "stage_1": "Layout Detection (YOLOv8)",
            "stage_2": "Region Cropping",
            "stage_3": "OCR Extraction (PaddleOCR)",
            "stage_4": "Document Understanding (LayoutLM)",
            "stage_5": "Post-processing (Product-Price Linking)"
        },
        "endpoints": {
            "/analyze-flyer": "POST - Analyze retail flyer image",
            "/health": "GET - Health check",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.post("/analyze-flyer")
async def analyze_flyer(
    file: UploadFile = File(...),
    return_visualization: bool = Query(False, description="Return annotated image"),
    return_debug: bool = Query(False, description="Return debug information")
):
    """
    Analyze retail flyer image and extract structured product data
    
    **Pipeline stages:**
    1. **Layout Detection**: YOLO detects product regions, prices, discounts
    2. **OCR Extraction**: PaddleOCR extracts text with bounding boxes
    3. **Document Understanding**: LayoutLM infers spatial relationships
    4. **Post-processing**: Links products with prices and normalizes data
    
    **Args:**
    - file: Flyer image (PNG, JPG, JPEG)
    - return_visualization: Include annotated image with detections
    - return_debug: Include detailed debug information
    
    **Returns:**
    - Structured JSON with extracted products
    """
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (PNG, JPG, JPEG)"
        )
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image. Please provide a valid image file."
            )
        
        # Process through pipeline
        result = pipeline.process(
            image,
            return_visualization=return_visualization,
            return_debug_info=return_debug
        )
        
        # Add timestamp
        result['timestamp'] = datetime.utcnow().isoformat()
        result['filename'] = file.filename
        
        # Handle visualization
        if return_visualization and 'visualization' in result:
            # Encode visualization as base64
            import base64
            _, buffer = cv2.imencode('.jpg', result['visualization'])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            result['visualization'] = img_base64
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns status of all pipeline components
    """
    health = pipeline.health_check()
    
    return {
        "status": "healthy" if health['pipeline'] == 'healthy' else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "pipeline": health['pipeline'],
        "components": health['components']
    }


@app.get("/info")
async def info():
    """
    Get detailed pipeline information
    """
    return {
        "system": "Retail Flyer Understanding System",
        "version": "1.0.0",
        "architecture": "YOLO + OCR + LayoutLM",
        "components": {
            "layout_detector": {
                "engine": "YOLOv8",
                "classes": [
                    "product_image",
                    "price_tag",
                    "discount_badge",
                    "product_title",
                    "brand_logo",
                    "description"
                ]
            },
            "ocr_engine": {
                "primary": "PaddleOCR",
                "fallback": "EasyOCR",
                "languages": ["pt", "en"]
            },
            "document_understanding": {
                "engine": "LayoutLMv3 (rule-based variant)",
                "capabilities": [
                    "Spatial relationship inference",
                    "Product-price linking",
                    "Discount association"
                ]
            }
        },
        "output_schema": {
            "products": [
                {
                    "product_name": "string",
                    "brand": "string | null",
                    "price": "float",
                    "price_formatted": "string",
                    "discount": "string | null",
                    "bounding_box": "[x1, y1, x2, y2]",
                    "confidence": "float"
                }
            ]
        }
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🏷️  Retail Flyer Understanding System")
    print("=" * 60)
    print("\n🚀 Starting production server...")
    print("📚 Documentation: http://localhost:8000/docs")
    print("🔍 Health check: http://localhost:8000/health")
    print("\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
