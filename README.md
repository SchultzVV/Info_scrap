# 🏷️ Sistema de Detecção e Leitura de Preços - YOLO + OCR

Sistema automatizado para detectar e ler valores promocionais em imagens usando YOLOv8 e OCR (EasyOCR/Tesseract).

## 📋 Características

- 🔍 **Detecção com YOLO**: Utiliza YOLOv8 para detectar regiões de preços
- 📖 **OCR Avançado**: Lê valores usando EasyOCR ou Tesseract
- 🚀 **API RESTful**: Interface FastAPI para integração fácil
- 🎯 **Pré-processamento**: Melhora imagens automaticamente para OCR
- 💰 **Extração de Valores**: Identifica preços em diversos formatos (R$ 36,99, 3699, etc)

## 🛠️ Instalação

### Pré-requisitos

- Python 3.8+
- Tesseract OCR (opcional, se não usar EasyOCR)

#### Instalar Tesseract (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-por
```

#### Instalar Tesseract (macOS):
```bash
brew install tesseract tesseract-lang
```

### Instalação das dependências

```bash
# Clone ou navegue até o diretório
cd /home/v/Desktop/info_scrap

# Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
# venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt
```

## 🚀 Uso

### Iniciar o servidor

```bash
python main.py
```

O servidor estará rodando em `http://localhost:8000`

### Documentação interativa

Acesse a documentação Swagger da API:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### `POST /detect-price`
Detecta e lê preços em uma imagem.

**Exemplo com curl:**
```bash
curl -X POST "http://localhost:8000/detect-price" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image_example.png"
```

**Exemplo com Python:**
```python
import requests

url = "http://localhost:8000/detect-price"
files = {"file": open("image_example.png", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Resposta:**
```json
{
  "success": true,
  "message": "1 preço(s) detectado(s)",
  "detections": [
    {
      "id": 1,
      "bbox": {
        "x1": 100,
        "y1": 150,
        "x2": 250,
        "y2": 200
      },
      "confidence": 0.89,
      "class": "object",
      "ocr_text": "R$ 36,99",
      "ocr_confidence": 0.92,
      "price_value": 36.99
    }
  ],
  "image_size": {
    "width": 800,
    "height": 600
  }
}
```

#### `GET /health`
Verifica o status da API e modelos.

## 📁 Estrutura do Projeto

```
info_scrap/
├── main.py              # API FastAPI principal
├── detector.py          # Módulo de detecção YOLO
├── ocr_reader.py        # Módulo de leitura OCR
├── requirements.txt     # Dependências Python
├── test_api.py         # Script para testar API
├── README.md           # Este arquivo
└── image_example.png   # Imagem de exemplo
```

## 🔧 Configuração Avançada

### Usar modelo YOLO customizado

Se você treinou seu próprio modelo YOLO para detectar preços:

```python
# Em detector.py, modifique o __init__:
detector = YOLODetector(model_path='caminho/para/seu/modelo.pt')
```

### Ajustar threshold de confiança

```python
# Em detector.py:
detector = YOLODetector(confidence_threshold=0.5)
```

### Escolher entre EasyOCR e Tesseract

```python
# Em ocr_reader.py:
ocr_reader = OCRReader(use_easyocr=True)  # EasyOCR (padrão)
# ou
ocr_reader = OCRReader(use_easyocr=False) # Tesseract
```

## 🧪 Testando

Execute o script de teste:

```bash
python test_api.py
```

Ou teste manualmente com a imagem de exemplo:

```bash
curl -X POST "http://localhost:8000/detect-price" \
  -F "file=@image_example.png" | jq
```

## 📦 Dependências Principais

- **FastAPI**: Framework web moderno e rápido
- **Ultralytics**: YOLOv8 para detecção de objetos
- **EasyOCR**: OCR com suporte a múltiplos idiomas
- **OpenCV**: Processamento de imagens
- **PyTorch**: Backend para modelos de deep learning

## 🐛 Solução de Problemas

### Erro: "Tesseract not found"
Instale o Tesseract OCR no seu sistema operacional.

### Erro: CUDA out of memory
O sistema usa CPU por padrão. Para usar GPU, certifique-se de ter PyTorch com suporte CUDA instalado.

### OCR não está lendo corretamente
- Verifique se a imagem tem boa qualidade
- Ajuste o pré-processamento em `ocr_reader.py`
- Experimente alterar entre EasyOCR e Tesseract

### YOLO não detecta preços
- O modelo YOLOv8n padrão é genérico
- Considere treinar um modelo customizado com imagens de preços
- Ajuste o `confidence_threshold`

## 📄 Licença

Este projeto é fornecido como está para fins educacionais e de pesquisa.

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se livre para abrir issues ou pull requests.
