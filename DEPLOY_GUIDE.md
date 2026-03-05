# Como Usar a Imagem Docker

## 🚀 Opção 1: Pull da Imagem do GitHub

```bash
# 1. Pull da imagem
docker pull ghcr.io/SEU-USUARIO/SEU-REPO:latest

# 2. Rodar container
docker run -d -p 8000:8000 --name ocr-api ghcr.io/SEU-USUARIO/SEU-REPO:latest

# 3. Testar
curl -X POST http://localhost:8000/extract \
  -F "file=@sua_imagem.png"
```

## 🔨 Opção 2: Build Local

```bash
# Build da imagem
make simple-build

# Subir container
make simple-up

# Testar
curl http://localhost:8000/health
```

## 📦 Configurar GitHub Actions

### 1. Preparar Repositório

```bash
# Inicializar Git (se não foi feito)
git init
git add .
git commit -m "Initial commit: Simple OCR API"

# Adicionar remote do GitHub
git remote add origin https://github.com/SEU-USUARIO/SEU-REPO.git

# Push para GitHub
git branch -M main
git push -u origin main
```

### 2. Configurar Permissões

No GitHub:
1. Vá em **Settings** → **Actions** → **General**
2. Em **Workflow permissions**, selecione:
   - ✅ **Read and write permissions**
3. Clique em **Save**

### 3. Verificar Build

Após o push, vá em:
- **Actions** → Veja o workflow rodando
- **Packages** → Imagem será publicada aqui

### 4. Usar a Imagem

A imagem estará disponível em:
```
ghcr.io/SEU-USUARIO/SEU-REPO:latest
```

## 🌐 Deploy em Produção

### Docker

```bash
docker run -d \
  --name ocr-api \
  -p 8000:8000 \
  --restart unless-stopped \
  ghcr.io/SEU-USUARIO/SEU-REPO:latest
```

### Docker Compose

```yaml
version: '3.8'
services:
  ocr-api:
    image: ghcr.io/SEU-USUARIO/SEU-REPO:latest
    ports:
      - "8000:8000"
    restart: unless-stopped
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ocr-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ocr-api
  template:
    metadata:
      labels:
        app: ocr-api
    spec:
      containers:
      - name: ocr-api
        image: ghcr.io/SEU-USUARIO/SEU-REPO:latest
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: ocr-api
spec:
  selector:
    app: ocr-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 🧪 Testar API

```bash
# Health check
curl http://localhost:8000/health

# Extrair texto
curl -X POST http://localhost:8000/extract \
  -F "file=@image.png" \
  | python -m json.tool

# Documentação
curl http://localhost:8000/docs
# Ou abra no navegador: http://localhost:8000/docs
```

## 📊 Monitoramento

```bash
# Ver logs
docker logs -f ocr-api

# Ver status
docker ps | grep ocr-api

# Ver uso de recursos
docker stats ocr-api
```

## 🔄 Atualizar Imagem

```bash
# Pull da nova versão
docker pull ghcr.io/SEU-USUARIO/SEU-REPO:latest

# Parar container antigo
docker stop ocr-api
docker rm ocr-api

# Subir novo container
docker run -d -p 8000:8000 --name ocr-api ghcr.io/SEU-USUARIO/SEU-REPO:latest
```

## 🎯 Exemplo Completo

```bash
# 1. Clone o repo
git clone https://github.com/SEU-USUARIO/SEU-REPO.git
cd SEU-REPO

# 2. Build local (opcional)
make simple-build

# 3. Rodar
make simple-up

# 4. Testar com uma imagem
curl -X POST http://localhost:8000/extract \
  -F "file=@examples/image_example.png"

# 5. Parar
make simple-down
```

## 🐛 Troubleshooting

### Porta 8000 em uso
```bash
# Usar outra porta
docker run -d -p 8080:8000 --name ocr-api ghcr.io/SEU-USUARIO/SEU-REPO:latest
```

### Ver logs de erro
```bash
docker logs ocr-api
```

### Rebuild sem cache
```bash
docker-compose -f docker-compose-simple.yml build --no-cache
```

---

**Pronto para usar!** ✅
