#!/bin/bash

# Script de instalação rápida
# Sistema de Detecção e Leitura de Preços - YOLO + OCR

echo "🏷️  Sistema de Detecção e Leitura de Preços - Instalação"
echo "========================================================"

# Verifica Python
echo ""
echo "1️⃣  Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado!"
    echo "   Instale Python 3.8+ antes de continuar"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✅ $PYTHON_VERSION encontrado"

# Verifica Tesseract (opcional)
echo ""
echo "2️⃣  Verificando Tesseract OCR..."
if ! command -v tesseract &> /dev/null; then
    echo "⚠️  Tesseract não encontrado (opcional)"
    echo "   Para instalar no Ubuntu/Debian:"
    echo "   sudo apt-get install tesseract-ocr tesseract-ocr-por"
else
    TESSERACT_VERSION=$(tesseract --version | head -n 1)
    echo "✅ $TESSERACT_VERSION encontrado"
fi

# Cria ambiente virtual
echo ""
echo "3️⃣  Criando ambiente virtual..."
if [ -d "venv" ]; then
    echo "⚠️  Ambiente virtual já existe"
    read -p "   Deseja recriar? (s/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo "✅ Ambiente virtual recriado"
    fi
else
    python3 -m venv venv
    echo "✅ Ambiente virtual criado"
fi

# Ativa ambiente virtual
echo ""
echo "4️⃣  Ativando ambiente virtual..."
source venv/bin/activate
echo "✅ Ambiente ativado"

# Atualiza pip
echo ""
echo "5️⃣  Atualizando pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✅ Pip atualizado"

# Instala dependências
echo ""
echo "6️⃣  Instalando dependências..."
echo "   (Isso pode levar alguns minutos...)"
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependências instaladas com sucesso!"
else
    echo "❌ Erro ao instalar dependências"
    exit 1
fi

# Baixa modelo YOLO
echo ""
echo "7️⃣  Verificando modelo YOLO..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>&1 | grep -v "Downloading\|Ultralytics"
echo "✅ Modelo YOLO pronto"

# Cria arquivo .env
echo ""
echo "8️⃣  Configurando ambiente..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Arquivo .env criado"
else
    echo "⚠️  Arquivo .env já existe"
fi

# Finaliza
echo ""
echo "========================================================"
echo "✨ Instalação concluída com sucesso!"
echo ""
echo "📚 Próximos passos:"
echo ""
echo "   1. Ative o ambiente virtual:"
echo "      source venv/bin/activate"
echo ""
echo "   2. Inicie o servidor:"
echo "      python main.py"
echo ""
echo "   3. Acesse a documentação:"
echo "      http://localhost:8000/docs"
echo ""
echo "   4. Teste com a imagem de exemplo:"
echo "      python test_api.py"
echo "      ou"
echo "      python test_local.py"
echo ""
echo "========================================================"
