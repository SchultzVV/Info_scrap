# 🏷️ Retail Flyer Understanding System

Sistema de nível produção para extração de dados promocionais de panfletos de varejo.

## 🎯 Stack Completo

**YOLO** → **OCR** → **LayoutLM** → **Product Linking**

### O que foi criado

✅ **Arquitetura modular seguindo SOLID**  
✅ **Pipeline completo em 5 estágios**  
✅ **API REST com FastAPI**  
✅ **Detecção de layout com YOLO**  
✅ **OCR com PaddleOCR (fallback EasyOCR)**  
✅ **Entendimento espacial estilo LayoutLM**  
✅ **Linking de produto-preço-desconto**  
✅ **Testes automatizados**  
✅ **Docker + Docker Compose**  
✅ **Documentação completa**

---

## 🚀 Start Rápido

### Docker (Recomendado)

```bash
docker-compose up --build
```

API disponível em: http://localhost:8000/docs

### Local

```bash
# Instalar
./install.sh

# Iniciar servidor
./run.sh

# OU
python api.py
```

---

## 📡 Usar a API

### Testar com cURL

```bash
curl -X POST "http://localhost:8000/analyze-flyer?return_debug=true" \
  -F "file=@examples/image_example.png"
```

### Testar com Python

```bash
# Pipeline local (sem API)
python tests/test_pipeline.py examples/image_example.png

# API (servidor deve estar rodando)
python tests/test_api.py examples/image_example.png
```

---

## 📊 Output

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
  ],
  "metadata": {
    "processing_time_seconds": 2.5,
    "num_products": 1
  }
}
```

---

## 🏗️ Arquitetura

```
┌──────────────────────────────────────────┐
│  Stage 1: Layout Detection (YOLO)      │
│  Classes: product_image, price_tag,     │
│           discount_badge, product_title │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  Stage 2: Region Cropping               │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  Stage 3: OCR (PaddleOCR / EasyOCR)     │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  Stage 4: Document Understanding        │
│  Spatial reasoning: title → price       │
└──────────────┬───────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│  Stage 5: Product-Price Linking         │
└──────────────────────────────────────────┘
```

---

## 📁 Estrutura

```
src/
├── detectors/         # YOLO layout detector
├── ocr/               # PaddleOCR engine
├── layout/            # LayoutLM processor
├── linking/           # Product-price linker
└── pipeline/          # Pipeline orchestrator

api.py                 # FastAPI server
tests/                 # Testes
Dockerfile             # Container
```

---

## 📚 Documentação Completa

Veja **README_PRODUCTION.md** para:
- Arquitetura detalhada
- Guias de uso avançado
- Como treinar modelos customizados
- Performance benchmarks
- Troubleshooting

---

## 🎓 Estado da Arte

Este sistema usa as mesmas técnicas de empresas como Amazon e Walmart:

1. **YOLO** para detecção de layout
2. **OCR** para extração de texto
3. **LayoutLM** para entendimento espacial de documentos
4. **Graph Neural Networks** (futuro) para knowledge graphs

---

## ✨ Features

- ✅ Modular e extensível
- ✅ Fácil de trocar componentes
- ✅ Testes automatizados
- ✅ Docker ready
- ✅ Documentação completa
- ✅ API REST profissional
- ✅ Suporte GPU/CPU

---

**Desenvolvido seguindo princípios SOLID e arquitetura de nível produção** 🚀
