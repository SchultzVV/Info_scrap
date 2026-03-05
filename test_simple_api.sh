#!/bin/bash

echo "🧪 Testando Simple OCR API..."
echo ""

# Test health
echo "1️⃣ Health Check:"
curl -s http://localhost:8000/health | python3 -m json.tool
echo -e "\n"

# Test extraction
echo "2️⃣ Extração de Texto (examples/image_example.png):"
curl -X POST http://localhost:8000/extract \
  -F "file=@examples/image_example.png" \
  -s | python3 -m json.tool

echo -e "\n✅ Testes completos!"
