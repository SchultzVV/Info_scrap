#!/usr/bin/env python3
"""
Script para testar o sistema localmente sem API
Útil para debugging e desenvolvimento
"""

import cv2
import sys
import os
from detector import YOLODetector
from ocr_reader import OCRReader


def test_local(image_path: str):
    """
    Testa detecção e OCR localmente
    
    Args:
        image_path: Caminho para a imagem
    """
    
    # Verifica se o arquivo existe
    if not os.path.exists(image_path):
        print(f"❌ Erro: Arquivo '{image_path}' não encontrado!")
        sys.exit(1)
    
    print(f"🔍 Testando sistema localmente com: {image_path}")
    print("=" * 60)
    
    # Carrega imagem
    print("\n1️⃣  Carregando imagem...")
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Erro: Não foi possível carregar a imagem!")
        sys.exit(1)
    print(f"✅ Imagem carregada: {image.shape[1]}x{image.shape[0]}")
    
    # Inicializa detector
    print("\n2️⃣  Inicializando YOLO...")
    detector = YOLODetector(confidence_threshold=0.25)
    print("✅ YOLO inicializado!")
    
    # Detecta objetos
    print("\n3️⃣  Detectando objetos...")
    detections = detector.detect(image)
    print(f"✅ {len(detections)} objeto(s) detectado(s)")
    
    if not detections:
        print("\n⚠️  Nenhum objeto detectado na imagem!")
        print("   Sugestões:")
        print("   - Ajuste o confidence_threshold")
        print("   - Use um modelo YOLO customizado")
        print("   - Verifique se a imagem está no formato correto")
        return
    
    # Inicializa OCR
    print("\n4️⃣  Inicializando OCR...")
    ocr_reader = OCRReader(use_easyocr=True)
    print("✅ OCR inicializado!")
    
    # Processa cada detecção
    print("\n5️⃣  Processando detecções com OCR...")
    print("=" * 60)
    
    for idx, detection in enumerate(detections):
        print(f"\n🏷️  Detecção #{idx + 1}:")
        print(f"   Classe: {detection['class']}")
        print(f"   Confiança: {detection['confidence']:.2%}")
        
        # Recorta região
        x1, y1, x2, y2 = detection['bbox']
        cropped = image[y1:y2, x1:x2]
        
        # Salva recorte para debug
        crop_name = f"crop_{idx + 1}.png"
        cv2.imwrite(crop_name, cropped)
        print(f"   Recorte salvo: {crop_name}")
        
        # Aplica OCR
        print(f"   Executando OCR...")
        ocr_result = ocr_reader.read_text(cropped)
        
        print(f"   Texto: '{ocr_result['text']}'")
        print(f"   Confiança OCR: {ocr_result['confidence']:.2%}")
        
        if ocr_result['price_value']:
            print(f"   💰 Preço: R$ {ocr_result['price_value']:.2f}")
        else:
            print(f"   ⚠️  Preço não identificado")
    
    # Salva imagem com detecções
    print("\n6️⃣  Salvando visualização...")
    result_image = detector.detect_and_visualize(image, save_path='result.png')
    print("✅ Resultado salvo em: result.png")
    
    print("\n" + "=" * 60)
    print("✨ Teste local concluído!")
    print("\nArquivos gerados:")
    print("   - result.png (imagem com detecções)")
    print("   - crop_*.png (recortes das detecções)")


def main():
    """Função principal"""
    
    print("🏷️  Sistema de Detecção e Leitura de Preços - Teste Local")
    print("=" * 60)
    
    # Determina caminho da imagem
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Usa imagem de exemplo
        image_path = "image_example.png"
    
    # Executa teste
    test_local(image_path)


if __name__ == "__main__":
    main()
