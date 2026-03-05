"""
Modern FastAPI Server - State of the Art 2026

Multimodal Document AI API for retail flyer understanding
Uses vision-language transformers (Donut/Pix2Struct/Kosmos-2)

Architecture:
    image → multimodal transformer → structured JSON

No explicit OCR/YOLO needed - end-to-end learning
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import uvicorn
import cv2
import numpy as np
from typing import Optional, List
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipelines.inference_pipeline import FlyerExtractionService

# Initialize FastAPI app
app = FastAPI(
    title="Retail Flyer AI - Multimodal Document Understanding",
    description="State-of-the-art API using vision-language transformers (Donut/Pix2Struct) for extracting structured promotional data from retail flyers. No explicit OCR needed.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
print("[API v2] Initializing Modern Flyer Extraction Service...")
service = FlyerExtractionService(
    model_name='donut',  # or 'pix2struct', 'kosmos2'
    use_gpu=False
)
print("[API v2] Service ready!")


# Request/Response models
class FlyerURLRequest(BaseModel):
    """Request with image URL"""
    image_url: HttpUrl


class ProductResponse(BaseModel):
    """Product data response"""
    product_name: str
    price: float
    price_formatted: str
    discount: Optional[str] = None
    brand: Optional[str] = None
    confidence: float


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Retail Flyer AI - Multimodal Document Understanding",
        "version": "2.0.0",
        "architecture": "State of the Art 2026",
        "approach": {
            "old": "YOLO + OCR + LayoutLM (multi-stage pipeline)",
            "new": "Multimodal Transformer (end-to-end)",
            "models": ["Donut", "Pix2Struct", "Kosmos-2"]
        },
        "advantages": [
            "Single end-to-end model",
            "No explicit OCR needed",
            "Fewer components = fewer errors",
            "Better context understanding",
            "Joint vision + language training"
        ],
        "endpoints": {
            "/flyer/extract": "POST - Extract products from flyer (file upload)",
            "/flyer/extract-url": "POST - Extract products from flyer (URL)",
            "/health": "GET - Health check",
            "/models": "GET - Available models"
        }
    }


@app.post("/flyer/extract")
async def extract_flyer(
    file: UploadFile = File(...),
    remove_duplicates: bool = Query(True, description="Remove duplicate products"),
    sort_by_confidence: bool = Query(True, description="Sort by confidence score"),
    custom_prompt: Optional[str] = Query(None, description="Custom extraction prompt")
):
    """
    Extract structured product data from retail flyer image
    
    **Modern Architecture (2026):**
    ```
    image → multimodal transformer → structured JSON
    ```
    
    **Models:**
    - Donut: Document Understanding Transformer
    - Pix2Struct: Screenshot parsing
    - Kosmos-2: Multimodal LLM
    
    **No explicit OCR needed** - end-to-end vision-language model
    
    **Args:**
    - file: Flyer image (PNG, JPG, JPEG)
    - remove_duplicates: Remove duplicate products
    - sort_by_confidence: Sort results by confidence
    - custom_prompt: Optional custom extraction prompt
    
    **Returns:**
    - Structured JSON with products, prices, discounts
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
        
        # Extract products
        result = service.extract(
            image,
            prompt=custom_prompt,
            remove_duplicates=remove_duplicates,
            sort_by_confidence=sort_by_confidence
        )
        
        # Add timestamp and filename
        result['timestamp'] = datetime.utcnow().isoformat()
        result['filename'] = file.filename
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing flyer: {str(e)}"
        )


@app.post("/flyer/extract-url")
async def extract_flyer_url(
    request: FlyerURLRequest,
    remove_duplicates: bool = Query(True),
    sort_by_confidence: bool = Query(True)
):
    """
    Extract products from flyer image URL
    
    **Input:**
    ```json
    {
      "image_url": "https://example.com/flyer.jpg"
    }
    ```
    
    **Output:**
    ```json
    {
      "products": [
        {
          "product_name": "Café Pilão 500g",
          "price": 13.99,
          "price_formatted": "R$ 13,99",
          "discount": "20%",
          "confidence": 0.93
        }
      ]
    }
    ```
    """
    
    try:
        import requests
        from io import BytesIO
        
        # Download image
        response = requests.get(str(request.image_url), timeout=10)
        response.raise_for_status()
        
        # Decode image
        image_data = BytesIO(response.content)
        nparr = np.frombuffer(image_data.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image from URL"
            )
        
        # Extract products
        result = service.extract(
            image,
            remove_duplicates=remove_duplicates,
            sort_by_confidence=sort_by_confidence
        )
        
        # Add metadata
        result['timestamp'] = datetime.utcnow().isoformat()
        result['source_url'] = str(request.image_url)
        
        return JSONResponse(content=result)
    
    except requests.RequestException as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not download image: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing flyer: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    
    Returns status of all pipeline components
    """
    
    health = service.health_check()
    
    return {
        "status": health['status'],
        "timestamp": datetime.utcnow().isoformat(),
        "components": health['components'],
        "model": health['model'],
        "api_version": "2.0.0",
        "architecture": "Multimodal Transformer"
    }


@app.get("/models")
async def available_models():
    """
    Get available multimodal models
    """
    
    return {
        "available_models": [
            {
                "name": "donut",
                "full_name": "Donut (Document Understanding Transformer)",
                "description": "Specialized for document understanding, no OCR needed",
                "paper": "https://arxiv.org/abs/2111.15664",
                "advantages": ["End-to-end", "Fast inference", "Good accuracy"]
            },
            {
                "name": "pix2struct",
                "full_name": "Pix2Struct",
                "description": "Screenshot parsing model from Google",
                "paper": "https://arxiv.org/abs/2210.03347",
                "advantages": ["Screenshot understanding", "Flexible prompting"]
            },
            {
                "name": "kosmos2",
                "full_name": "Kosmos-2",
                "description": "Multimodal large language model from Microsoft",
                "paper": "https://arxiv.org/abs/2306.14824",
                "advantages": ["Multimodal LLM", "Grounding", "Rich understanding"]
            }
        ],
        "current_model": service.extractor.model_name,
        "note": "Models require transformers library: pip install transformers"
    }


@app.get("/info")
async def info():
    """
    Get detailed pipeline information
    """
    
    return {
        "system": "Retail Flyer AI - Multimodal Document Understanding",
        "version": "2.0.0",
        "architecture": {
            "paradigm": "End-to-end multimodal learning",
            "approach": "Vision-Language Transformers",
            "vs_traditional": {
                "old_pipeline": "YOLO → OCR → LayoutLM → Post-processing",
                "new_pipeline": "Multimodal Transformer → Post-processing",
                "benefits": [
                    "Single model instead of 3+ components",
                    "No explicit OCR errors",
                    "Better spatial understanding",
                    "End-to-end training possible"
                ]
            }
        },
        "models": {
            "current": service.extractor.model_name,
            "supported": ["donut", "pix2struct", "kosmos2"]
        },
        "output_schema": {
            "products": [
                {
                    "product_name": "string",
                    "price": "float",
                    "price_formatted": "string (with currency)",
                    "discount": "string | null",
                    "brand": "string | null",  
                    "confidence": "float (0-1)"
                }
            ],
            "metadata": {
                "num_products": "int",
                "model_used": "string",
                "processing_time_seconds": "float"
            }
        },
        "state_of_the_art": {
            "year": 2026,
            "trend": "Multimodal transformers replacing traditional pipelines",
            "industry_adoption": ["Amazon", "Walmart", "Major retailers"]
        }
    }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 Retail Flyer AI - Multimodal Document Understanding API v2.0")
    print("=" * 70)
    print("\n📚 State of the Art 2026 Architecture")
    print("   Old: YOLO + OCR + LayoutLM")
    print("   New: Multimodal Transformer (end-to-end)")
    print("\n🔬 Models: Donut, Pix2Struct, Kosmos-2")
    print("\n📡 Endpoints:")
    print("   • http://localhost:8000/docs (Interactive API docs)")
    print("   • http://localhost:8000/flyer/extract (Extract from flyer)")
    print("   • http://localhost:8000/health (Health check)")
    print("   • http://localhost:8000/models (Available models)")
    print("\n" + "=" * 70)
    print("\n▶️  Starting server...\n")
    
    uvicorn.run(
        "api_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
