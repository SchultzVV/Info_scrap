# 🚀 Info Scrap - API de Extração e Análise de Produtos

API REST para extração e análise inteligente de informações de produtos a partir de imagens usando OCR (Tesseract) e detecção por ROI.

## 🎯 Início Rápido

### Usando Docker (Recomendado)

```bash
# Subir o serviço
make simple-up

# Parar o serviço
make simple-down
```

A API estará disponível em `http://localhost:8000`

### Usando Python Local

```bash
# Instalar dependências
pip install -r requirements-simple.txt

# Iniciar servidor
python api_simple.py
```

---

## 📡 Endpoints Disponíveis

### 1. `POST /extract` - Extração Básica de Texto

Extrai todo o texto da imagem usando OCR.

**Exemplo de Requisição:**
```bash
curl -X POST "http://localhost:8000/extract" \
  -F "file=@image_example.png"
```

**Exemplo de Resposta:**
```json
{
  "success": true,
  "text": "LOJA OFICIAL APPLE\n\niPhone 16e (128 Gb) - Branco...",
  "lines": [
    "LOJA OFICIAL APPLE",
    "iPhone 16e (128 Gb) - Branco - Distribuidor Autorizado",
    "Por Apple",
    "a$5709",
    "R$ 3.699 36% OFF no Pix",
    "21x R$ 185,44 sem juros"
  ]
}
```

---

### 2. `POST /analyze` - Análise Estruturada de Produto

**🎯 Detecta automaticamente o tipo de imagem e extrai informações estruturadas:**

- **Produtos isolados**: Usa ROI com detecção de strikethrough
- **Screenshots de e-commerce**: Parser inteligente com recuperação de OCR

**Exemplo de Requisição:**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@image_example.png" \
  -s | python -m json.tool
```

**Exemplo de Resposta (Produto Isolado):**
```json
{
  "success": true,
  "product": {
    "title": "iPhone LOJA OFICIAL 16e APPLE (128 Gb) - Branco - Distribuidor Autorizado",
    "old_price": {
      "raw_text": "a$5709",
      "value": 5709.0,
      "formatted": "R$ 5.709,00"
    },
    "current_price": {
      "raw_text": "3.699",
      "value": 3699.0,
      "formatted": "R$ 3.699,00"
    },
    "installment": {
      "raw_text": "21x 185,44",
      "installments": 21,
      "value_per_installment": 185.44,
      "total_value": 3894.24,
      "formatted_total": "R$ 3.894,24",
      "formatted_per_installment": "R$ 185,44"
    },
    "discount": {
      "raw_text": "36%",
      "percentage": 36
    }
  }
}
```

**Exemplo de Resposta (E-commerce Screenshot):**
```json
{
  "success": true,
  "product": {
    "title": "Lava Roupas Sabão Líquido Omo Lavagem Perfeita 7L",
    "old_price": {
      "raw_text": "816728,",
      "value": 167.28,
      "formatted": "R$ 167,28",
      "orphan": true
    },
    "current_price": {
      "raw_text": "R$ 124",
      "value": 124.0,
      "formatted": "R$ 124,00"
    },
    "installment": null,
    "discount": {
      "raw_text": "25% oFF",
      "percentage": 25
    },
    "shipping": "Frete grátis por ser sua primeira compra"
  }
}
```

**🧠 Recursos Inteligentes:**

- ✅ **Detecção Automática**: Identifica se é produto isolado ou página web
- ✅ **Recuperação de OCR**: Corrige erros como "816728" → R$ 167,28
- ✅ **Inferência de Preços**: Calcula preço antigo usando desconto
- ✅ **Filtragem de Interface**: Remove elementos de navegação (buscar, menu, etc)
- ✅ **Detecção de Strikethrough**: Identifica preços riscados por análise visual
- ✅ **Parcelamento Inteligente**: Combina "21x" + valor por proximidade
- ✅ **Validação Automática**: Verifica consistência entre preços e descontos

---

## 🧠 Como Funciona o `/analyze`

O endpoint usa **duas estratégias complementares** com detecção automática:

### 📸 Estratégia 1: ROI Detector (Produtos Isolados)

Para imagens de produtos com fundo limpo (cards, banners, etc):

```
┌───────────────────────────────────────────┐
│  1. Detecção de ROIs                      │
│     Identifica regiões de interesse       │
└──────────────┬────────────────────────────┘
               ↓
┌───────────────────────────────────────────┐
│  2. Extração de Título                    │
│     Busca texto longo na região superior  │
│     (40% superior da imagem)              │
└──────────────┬────────────────────────────┘
               ↓
┌───────────────────────────────────────────┐
│  3. Inferência de ROI de Preços           │
│     Calcula região abaixo do título       │
│     onde os preços devem estar            │
└──────────────┬────────────────────────────┘
               ↓
┌───────────────────────────────────────────┐
│  4. OCR Focado                            │
│     Re-executa OCR apenas na ROI          │
│     para maior precisão                   │
└──────────────┬────────────────────────────┘
               ↓
┌───────────────────────────────────────────┐
│  5. Detecção de Strikethrough             │
│     Análise visual para preços riscados   │
└──────────────┬────────────────────────────┘
               ↓
┌───────────────────────────────────────────┐
│  6. Classificação e Validação             │
│     Aplica regras de negócio              │
└───────────────────────────────────────────┘
```

### 🌐 Estratégia 2: E-commerce Parser (Screenshots Web)

Para capturas de tela de sites como Mercado Livre, Amazon, Magazine Luiza:

```
┌───────────────────────────────────────────┐
│  1. OCR Completo da Página                │
│     Extrai todo o texto visível           │
└──────────────┬────────────────────────────┘
               ↓
┌───────────────────────────────────────────┐
│  2. Filtragem de Interface                │
│     Remove: buscar, menu, cookies, etc    │
└──────────────┬────────────────────────────┘
               ↓
┌───────────────────────────────────────────┐
│  3. Detecção de Padrões Brasileiros       │
│     R$ 1.234,56 | 25% OFF | 12x R$ 10,50  │
└──────────────┬────────────────────────────┘
               ↓
┌───────────────────────────────────────────┐
│  4. Recuperação de OCR                    │
│     "816728" → R$ 167,28                  │
│     Corrige erros de leitura              │
└──────────────┬────────────────────────────┘
               ↓
┌───────────────────────────────────────────┐
│  5. Inferência de Preço Antigo            │
│     Se não detectado: old = current/(1-d) │
└──────────────┬────────────────────────────┘
               ↓
┌───────────────────────────────────────────┐
│  6. Validação e Estruturação              │
│     Verifica: old > current, etc          │
└───────────────────────────────────────────┘
```

### 🎯 Detecção Automática

O sistema tenta **ROI Detector** primeiro. Se não encontrar preços, automaticamente usa **E-commerce Parser**.

---

## 🔍 Regras de Detecção

### 🏷️ Preço Antigo (`old_price`)

**ROI Detector:**
- Padrão `a$` seguido de números (OCR lê R$ riscado como "a$")
- Sempre maior que preço atual

**E-commerce Parser:**
- Números "órfãos" entre título e preço (ex: "816728" → R$ 167,28)
- Inferido do desconto se não detectado: `old_price = current_price / (1 - discount/100)`

### 💰 Preço Atual (`current_price`)

**ROI Detector:**
- Strikethrough visual OU símbolo "R$"
- Sempre menor que preço antigo

**E-commerce Parser:**
- Padrão `R$ \d+` ou `\d+,\d{2}`
- Validação: entre R$ 0,01 e R$ 1.000.000

### 💳 Parcelamento (`installment`)

- Padrão: `\d+x` (ex: "21x") + valor próximo
- Combina por proximidade espacial (< 30px horizontal, < 5px vertical)
- Validação: total >= preço atual

### 🎯 Desconto (`discount`)

- Padrões: `25% OFF`, `25%OFF`, `25% oFF`
- Validação: `((old_price - current_price) / old_price) * 100 ≈ discount`

---

## 🛠️ Tecnologias

- **FastAPI** - Framework web moderno e rápido
- **Tesseract OCR** - Engine de reconhecimento óptico de caracteres
- **OpenCV** - Processamento de imagens e detecção de ROI
- **PIL/Pillow** - Manipulação de imagens
- **Docker** - Containerização e deploy

---

## 📋 Requisitos

- Python 3.8+
- Tesseract OCR instalado no sistema
- Docker (opcional, para deploy containerizado)

---

## 🐳 Docker

### Build Manual
```bash
docker build -f Dockerfile.simple -t simple-ocr-api .
```

### Run Manual
```bash
docker run -d -p 8000:8000 --name simple-ocr-api simple-ocr-api
```

### Ver Logs
```bash
docker logs -f simple-ocr-api
```

### Parar Container
```bash
docker stop simple-ocr-api
docker rm simple-ocr-api
```

---

## 🎯 Nova Estratégia: `/analyze-yolo` - Detecção por Objeto (YOLO)

**[⚠️ Em Desenvolvimento]** Uma nova abordagem usando redes neurais treinadas (YOLO) para detectar campos específicos (título, preços, parcelamento, etc) diretamente da imagem.

### 🚀 Como Usar

```bash
# Teste com imagem de exemplo
curl -X POST "http://localhost:8000/analyze-yolo" \
  -F "file=@examples/image_example.png" \
  -s | python -m json.tool
```

### 📊 Exemplo de Resposta

```json
{
  "success": true,
  "image_dimensions": {
    "width": 1709,
    "height": 925
  },
  "detections": {
    "title": [],
    "old_price": [],
    "current_price": [],
    "installment": [],
    "stock": [],
    "seller": []
  },
  "product": {
    "title": null,
    "old_price": null,
    "current_price": null,
    "installment": null,
    "stock": null,
    "seller": null
  },
  "model_status": {
    "loaded": false,
    "path": "models/best.pt",
    "type": "tiny",
    "device": "cpu"
  }
}
```

### 🧠 Como Funciona

1. **Modelo YOLO Treinado**: Detecta 6 classes de interesse
   - `title` - Título do produto
   - `old_price` - Preço antigo/riscado  
   - `current_price` - Preço atual
   - `installment` - Informação de parcelamento
   - `stock` - Quantidade em estoque
   - `seller` - Informação do vendedor

2. **Bounding Boxes**: Para cada detecção, obtém coordenadas (x1, y1, x2, y2)

3. **Extração Regional**: Aplica OCR **apenas** na região detectada

4. **Parsing Inteligente**: Processa texto de acordo com o tipo de campo

### ⚙️ Status do Modelo

O endpoint retorna `model_status` com informações:

```json
{
  "model_status": {
    "loaded": true,           // Modelo foi carregado com sucesso
    "path": "models/best.pt", // Caminho do modelo
    "type": "custom",         // "custom" (seu modelo) ou "tiny" (teste)
    "device": "cpu"           // Device: "cpu" ou "0" (GPU CUDA)
  }
}
```

### 📋 Fluxo de Carregamento

1. **Verifica** se `models/best.pt` existe
2. **Se existe**: Carrega modelo customizado treinado  
   - `type`: `"custom"`, `loaded`: `true`
3. **Se não existe**: Carrega modelo `yolov8n` (nano) para testes
   - `type`: `"tiny"`, `loaded`: `true`
4. **Se falhar**: Retorna `loaded: false` (sem crash)

### 🎯 Como Treinar seu Modelo

```bash
# 1. Preparar dataset com labels (YOLO format)
#    Estrutura esperada:
#    data/
#    ├── images/
#    │   ├── train/
#    │   └── val/
#    └── labels/
#        ├── train/
#        └── val/

# 2. Executar treinamento (em seu ambiente local)
from ultralytics import YOLO

model = YOLO('yolov8n.yaml')
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    device=0  # GPU
)

# 3. Exportar melhor modelo
best_model = YOLO('runs/detect/train/weights/best.pt')

# 4. Copiar para models/best.pt
cp runs/detect/train/weights/best.pt models/best.pt
```

### 🔧 GPU Support

O container detecta automaticamente:
- ✅ **GPU disponível**: Usa CUDA (muito mais rápido!)
- ❌ **GPU indisponível**: Fallback para CPU (mais lento, 5-10s por imagem)

Para usar GPU no Docker:
```bash
docker run --gpus all -p 8000:8000 ocr-api:latest
```

### 📈 Comparação de Estratégias

| Aspecto | ROI Detector | E-commerce Parser | YOLO |
|---------|--------------|-------------------|------|
| **Tipo** | Clássico OpenCV | Regex + OCR | Deep Learning |
| **Velocidade** | Fast (~1s) | Rápido (~1-2s) | Médio (~2-5s) |
| **Acurácia** | 85-90% | 80-85% | 95%+ |
| **Treinamento** | Não necessário | Não necessário | Requer dataset |
| **Generalização** | Limitada | Limitada | Excelente |
| **Tipos de Input** | Produtos puros | E-commerce | Qualquer tipo |

---

## 🧪 Testes

### Teste Rápido
```bash
# Extração básica
curl -X POST "http://localhost:8000/extract" \
  -F "file=@examples/image_example.png"

# Análise estruturada (ROI + E-commerce)
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@examples/image_example.png" \
  -s | python -m json.tool

# Análise por YOLO (requer modelo treinado em models/best.pt)
curl -X POST "http://localhost:8000/analyze-yolo" \
  -F "file=@examples/image_example.png" \
  -s | python -m json.tool
```

### Health Check
```bash
# Verificar se API está rodando
curl http://localhost:8000/health
# Resposta: {"status": "healthy"}
```

---

## 📁 Estrutura do Projeto

```
.
├── api_simple.py              # 🚀 API FastAPI com endpoints
├── analyzer.py                # 🧠 Lógica de análise avançada por ROI
├── detector.py                # 🔍 Detector YOLO (não usado no modo simples)
├── ocr_reader.py             # 📝 Wrapper do Tesseract OCR
├── Dockerfile.simple          # 🐳 Dockerfile para build
├── docker-compose-simple.yml  # 🐳 Docker Compose
├── requirements-simple.txt    # 📦 Dependências Python
├── Makefile                   # ⚙️ Comandos make para deploy
├── test_simple_api.sh        # 🧪 Script de testes
└── README.md                  # 📖 Esta documentação
```

---

## 🔧 Configuração Avançada

### Variáveis de Ambiente

```bash
# Porta do servidor (default: 8000)
export PORT=8000

# Nível de log
export LOG_LEVEL=info
```

### Linguagens do Tesseract

O OCR está configurado para português e inglês por padrão:
```python
lang='por+eng'
```

Para adicionar outras linguagens:
```bash
# Instalar pacote de idioma
sudo apt-get install tesseract-ocr-fra  # Francês
sudo apt-get install tesseract-ocr-spa  # Espanhol

# Usar múltiplas linguagens
pytesseract.image_to_string(image, lang='por+eng+fra+spa')
```

---

## 📊 Performance

- **Extração básica** (`/extract`): ~500-800ms por imagem (1709x925px)
- **Análise estruturada** (`/analyze`): ~1-2s por imagem
- **Memória**: ~150-200MB por container
- **Concorrência**: Suporta múltiplas requisições simultâneas via Uvicorn

---

## 🐛 Troubleshooting

### Erro: "Tesseract not found"
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-por

# macOS
brew install tesseract tesseract-lang

# Verificar instalação
tesseract --version
```

### Container não inicia
```bash
# Ver logs para identificar erro
docker logs simple-ocr-api

# Reconstruir imagem
docker-compose -f docker-compose-simple.yml build --no-cache
```

### Baixa acurácia no OCR
- Aumentar resolução da imagem
- Aplicar pré-processamento (threshold, denoising)
- Verificar se a imagem tem texto nítido

---

## 🔄 Versões Legadas

Este projeto também contém versões anteriores com outras arquiteturas:

- **v1 (Legacy)**: YOLO + OCR + LayoutLM → [README_PRODUCTION.md](legacy/README_PRODUCTION.md)
- **v2 (Modern)**: Multimodal Transformers → [README_V2.md](legacy/README_V2.md)

A versão atual (Simple API) é otimizada para simplicidade e velocidade.

---

## 🤝 Contribuindo

1. Fork o projeto
2. Crie sua feature branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

---

## 📄 Licença

Este projeto está sob a licença MIT.

---

**Desenvolvido com ❤️ para análise inteligente de produtos** 🚀
