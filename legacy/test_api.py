#!/usr/bin/env python3
"""
Script para testar a API de detecção de preços
"""

import requests
import json
import sys
import os
from pathlib import Path


def test_api(image_path: str, api_url: str = "http://localhost:8000"):
    """
    Testa a API com uma imagem
    
    Args:
        image_path: Caminho para a imagem
        api_url: URL base da API
    """
    
    # Verifica se o arquivo existe
    if not os.path.exists(image_path):
        print(f"❌ Erro: Arquivo '{image_path}' não encontrado!")
        sys.exit(1)
    
    print(f"🔍 Testando API com imagem: {image_path}")
    print(f"🌐 URL da API: {api_url}")
    print("-" * 60)
    
    # Testa health check
    print("\n1️⃣  Verificando saúde da API...")
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print("✅ API está saudável!")
            print(f"   Status: {health.get('status')}")
            print(f"   Modelos: {health.get('models')}")
        else:
            print(f"⚠️  API retornou status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Erro: Não foi possível conectar à API!")
        print("   Certifique-se de que o servidor está rodando:")
        print("   python main.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erro: {e}")
        sys.exit(1)
    
    # Envia imagem para detecção
    print(f"\n2️⃣  Enviando imagem para detecção...")
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/png')}
            response = requests.post(
                f"{api_url}/detect-price",
                files=files,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Detecção concluída com sucesso!")
            print("\n" + "=" * 60)
            print("📊 RESULTADOS:")
            print("=" * 60)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("=" * 60)
            
            # Resumo
            if result.get('success'):
                detections = result.get('detections', [])
                print(f"\n📋 Resumo:")
                print(f"   Total de detecções: {len(detections)}")
                
                for det in detections:
                    print(f"\n   🏷️  Detecção #{det['id']}:")
                    print(f"      Texto OCR: {det['ocr_text']}")
                    print(f"      Preço: R$ {det['price_value']}" if det['price_value'] else "      Preço não identificado")
                    print(f"      Confiança YOLO: {det['confidence']:.2%}")
                    print(f"      Confiança OCR: {det['ocr_confidence']:.2%}")
                    print(f"      Posição: ({det['bbox']['x1']}, {det['bbox']['y1']}) -> ({det['bbox']['x2']}, {det['bbox']['y2']})")
            
        else:
            print(f"❌ Erro: API retornou status {response.status_code}")
            print(f"   Resposta: {response.text}")
            
    except Exception as e:
        print(f"❌ Erro durante detecção: {e}")
        sys.exit(1)


def main():
    """Função principal"""
    
    print("🏷️  Sistema de Detecção e Leitura de Preços - YOLO + OCR")
    print("=" * 60)
    
    # Determina caminho da imagem
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Usa imagem de exemplo
        image_path = "image_example.png"
    
    # URL da API
    api_url = os.getenv("API_URL", "http://localhost:8000")
    
    # Executa teste
    test_api(image_path, api_url)
    
    print("\n✨ Teste concluído!")


if __name__ == "__main__":
    main()
