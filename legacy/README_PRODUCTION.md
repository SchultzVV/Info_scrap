# 🏷️ Retail Flyer Understanding System

**Production-grade pipeline for extracting structured promotional product data from retail flyer images**

Uses state-of-the-art Computer Vision + Document AI:
- **YOLOv8** for layout detection
- **PaddleOCR** for text extraction  
- **LayoutLM-inspired** spatial reasoning for document understanding
- **Modular architecture** following SOLID principles

---

## 🎯 What It Does

Transforms unstructured retail flyer images into structured product data:

**Input:** Flyer image  
**Output:** JSON with products, prices, discounts, and brands

```json
{
  "products": [
    {
      "product_name": "Café Pilão 500g",
      "brand": "Pilão",
      "price": 13.99,
      "price_formatted": "R$ 13,99",
      "discount": "20%",
      "bounding_box": [320, 410, 540, 680],
      "confidence": 0.94
    }
  ]
}
```

---

## 🏗️ Architecture

### Pipeline Stages

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Layout Detection (YOLO)                          │
│  Detects: product_image, price_tag, discount_badge,        │
│           product_title, brand_logo, description            │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Region Cropping                                   │
│  Extracts detected regions with coordinates                 │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: OCR Extraction (PaddleOCR)                        │
│  Extracts text with bounding boxes and confidence           │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: Document Understanding (LayoutLM)                 │
│  Infers spatial relationships between elements              │
│  Learns patterns: [product_title] → [price] → [discount]   │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 5: Post-processing                                   │
│  Normalizes prices, links products, extracts discounts      │
└─────────────────────────────────────────────────────────────┘
```

### Module Structure

```
src/
├── detectors/          # YOLO layout detector
│   └── yolo_layout_detector.py
├── ocr/                # PaddleOCR engine
│   └── paddle_ocr_engine.py
├── layout/             # LayoutLM processor
│   └── layoutlm_processor.py
├── linking/            # Product-price linker
│   ├── product_price_linker.py
│   └── price_normalizer.py
└── pipeline/           # Main pipeline orchestrator
    └── flyer_pipeline.py
```

**Each component is:**
- ✅ Independently replaceable
- ✅ Unit testable
- ✅ Follows SOLID principles
- ✅ Has clear interfaces

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Build and start
docker-compose up --build

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Option 2: Local Installation

```bash
# Clone/navigate to directory
cd info_scrap

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Start API server
python api.py
```

---

## 📡 API Usage

### Start Server

```bash
python api.py
```

Server runs at `http://localhost:8000`

### Endpoints

#### `POST /analyze-flyer`

Analyze flyer image and extract products.

**cURL:**
```bash
curl -X POST "http://localhost:8000/analyze-flyer?return_debug=true" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image_example.png"
```

**Python:**
```python
import requests

url = "http://localhost:8000/analyze-flyer"
files = {"file": open("image_example.png", "rb")}
params = {"return_visualization": False, "return_debug": True}

response = requests.post(url, files=files, params=params)
result = response.json()

for product in result['products']:
    print(f"{product['product_name']}: {product['price_formatted']}")
```

**Response:**
```json
{
  "success": true,
  "products": [
    {
      "product_name": "Product Name",
      "brand": "Brand Name",
      "price": 36.99,
      "price_formatted": "R$ 36,99",
      "discount": "20%",
      "bounding_box": [100, 200, 300, 400],
      "confidence": 0.92
    }
  ],
  "metadata": {
    "num_products": 1,
    "num_regions_detected": 5,
    "num_ocr_tokens": 23,
    "processing_time_seconds": 2.5,
    "image_size": [800, 600]
  },
  "timestamp": "2026-03-04T19:30:00",
  "filename": "image_example.png"
}
```

#### `GET /health`

Check API and pipeline health.

```bash
curl http://localhost:8000/health
```

#### `GET /info`

Get detailed pipeline information.

```bash
curl http://localhost:8000/info
```

---

## 🧪 Testing

### Test Pipeline Locally (no API)

```bash
python tests/test_pipeline.py image_example.png
```

Output:
- Console: Processing stages and results
- `pipeline_result.json`: Structured data
- `pipeline_result.jpg`: Annotated visualization

### Test API Endpoints

```bash
# Start server first
python api.py

# In another terminal
python tests/test_api.py image_example.png
```

---

## 🛠️ Configuration

### Environment Variables

Create `.env` file:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# YOLO Configuration
YOLO_MODEL_PATH=yolov8n.pt
YOLO_CONFIDENCE=0.25

# OCR Configuration
OCR_LANG=pt
USE_GPU=false
```

### Custom YOLO Model

Train your own model for retail flyers:

```python
from src.pipeline import FlyerPipeline

pipeline = FlyerPipeline(
    yolo_model_path='models/custom_flyer_yolo.pt',
    yolo_confidence=0.3
)
```

---

## 📊 Performance

**Typical Performance (CPU):**
- Small flyer (800x600): ~2-3 seconds
- Large flyer (1920x1080): ~5-7 seconds

**With GPU:**
- 2-3x faster for YOLO inference
- OCR remains CPU-bound

---

## 🎓 Advanced: Training Custom Models

### YOLO for Flyer Layout

1. **Annotate flyer images** with these classes:
   - `product_image`
   - `price_tag`
   - `discount_badge`
   - `product_title`
   - `brand_logo`
   - `description`

2. **Train YOLOv8:**
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='flyer_dataset.yaml',
    epochs=100,
    imgsz=640
)
```

3. **Use in pipeline:**
```python
pipeline = FlyerPipeline(
    yolo_model_path='runs/detect/train/weights/best.pt'
)
```

### LayoutLM Fine-tuning

For production systems, fine-tune LayoutLMv3:

```python
# TODO: Add LayoutLMv3 fine-tuning example
# Requires labeled flyer documents with entity relationships
```

---

## 📦 Dependencies

### Core
- **FastAPI**: REST API framework
- **Ultralytics**: YOLOv8 implementation
- **PaddleOCR**: Text extraction (falls back to EasyOCR)
- **OpenCV**: Image processing
- **PyTorch**: Deep learning backend

### Optional
- **transformers**: For actual LayoutLMv3 model
- **CUDA**: For GPU acceleration

---

## 🗂️ Project Structure

```
info_scrap/
├── api.py                      # FastAPI server
├── src/
│   ├── detectors/              # YOLO layout detector
│   ├── ocr/                    # PaddleOCR engine
│   ├── layout/                 # LayoutLM processor
│   ├── linking/                # Product-price linking
│   └── pipeline/               # Main pipeline
├── tests/
│   ├── test_pipeline.py        # Pipeline tests
│   └── test_api.py             # API tests
├── models/                     # Model weights
├── examples/                   # Example images
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image
├── docker-compose.yml          # Docker orchestration
└── README.md                   # This file
```

---

## 🐛 Troubleshooting

### PaddleOCR Not Available

System automatically falls back to EasyOCR. To use PaddleOCR:

```bash
pip install paddleocr paddlepaddle
```

### OCR Not Reading Correctly

- Ensure images have good quality/resolution
- Adjust preprocessing in `paddle_ocr_engine.py`
- Try different OCR engines

### YOLO Not Detecting Regions

- Default model is generic YOLOv8n
- **Solution**: Train custom model on retail flyers
- Lower confidence threshold: `yolo_confidence=0.15`

---

## 📚 State of the Art (2025-2026)

Real-world retail systems use:

### Detection
- YOLOv8 / Detectron2

### OCR
- PaddleOCR
- TrOCR (transformer-based)

### Document Understanding
- **LayoutLMv3**: Understands spatial document layout
- **Donut**: End-to-end document understanding

### Advanced (Amazon/Walmart)
```
YOLO → OCR → LayoutLM → Graph Neural Network → Knowledge Graph
```
Links promotions with product catalogs.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Fine-tuned YOLO model for flyers
- [ ] Actual LayoutLMv3 integration
- [ ] Multi-language support
- [ ] Batch processing endpoint
- [ ] Web UI for visualization

---

## 📄 License

MIT License - Use freely for commercial or research purposes.

---

## 🙏 Acknowledgments

Built with:
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [FastAPI](https://fastapi.tiangolo.com/)
- [LayoutLM](https://github.com/microsoft/unilm/tree/master/layoutlm)

---

**Built with ❤️ for Computer Vision + Document AI**
