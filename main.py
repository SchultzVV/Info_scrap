from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from typing import List, Dict
import tempfile
import os

from detector import YOLODetector
from ocr_reader import OCRReader

app = FastAPI(
    title="Sistema de Detecção e Leitura de Preços",
    description="API para detectar preços promocionais em imagens usando YOLO + OCR",
    version="1.0.0"
)

# Inicializa os modelos
detector = YOLODetector()
ocr_reader = OCRReader()


@app.get("/")
async def root():
    """Endpoint raiz com informações da API"""
    return {
        "message": "Sistema de Detecção de Preços - YOLO + OCR",
        "version": "1.0.0",
        "endpoints": {
            "/detect-price": "POST - Detecta e lê preços em uma imagem"
        }
    }


@app.post("/detect-price")
async def detect_price(file: UploadFile = File(...)):
    """
    Detecta preços promocionais em uma imagem usando YOLO e OCR
    
    Args:
        file: Arquivo de imagem (PNG, JPG, JPEG)
    
    Returns:
        JSON com os preços detectados e suas coordenadas
    """
    # Valida o tipo de arquivo
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="O arquivo deve ser uma imagem")
    
    try:
        # Lê a imagem do upload
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Não foi possível processar a imagem")
        
        # Detecta objetos com YOLO
        detections = detector.detect(image)
        
        if not detections:
            return JSONResponse(
                content={
                    "success": True,
                    "message": "Nenhum preço detectado na imagem",
                    "detections": []
                }
            )
        
        # Processa cada detecção com OCR
        results = []
        for idx, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            # Recorta a região detectada
            cropped = image[y1:y2, x1:x2]
            
            # Aplica OCR
            ocr_result = ocr_reader.read_text(cropped)
            
            results.append({
                "id": idx + 1,
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                },
                "confidence": float(confidence),
                "class": class_name,
                "ocr_text": ocr_result['text'],
                "ocr_confidence": ocr_result['confidence'],
                "price_value": ocr_result['price_value']
            })
        
        return JSONResponse(
            content={
                "success": True,
                "message": f"{len(results)} preço(s) detectado(s)",
                "detections": results,
                "image_size": {
                    "width": image.shape[1],
                    "height": image.shape[0]
                }
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {str(e)}")


@app.get("/health")
async def health_check():
    """Verifica se a API está funcionando"""
    return {
        "status": "healthy",
        "models": {
            "yolo": detector.is_loaded(),
            "ocr": ocr_reader.is_loaded()
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
