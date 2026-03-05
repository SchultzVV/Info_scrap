FROM python:3.10-slim

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-por \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Define diretório de trabalho
WORKDIR /app

# Copia requirements
COPY requirements.txt .

# Instala dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia código da aplicação
COPY api.py ./
COPY src/ ./src/
COPY pipelines/ ./pipelines/
COPY examples/ ./examples/

# Criar diretórios necessários
RUN mkdir -p /app/models /app/uploads /app/outputs

# Expõe porta
EXPOSE 8000

# Comando padrão - API v1 (Legacy - mais estável)
# Para usar v2 (Modern), mude para: CMD ["python", "src/serving/api_v2.py"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
