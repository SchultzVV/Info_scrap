import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Dict
import os


class YOLODetector:
    """
    Classe para detecção de preços usando YOLO
    """
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.25):
        """
        Inicializa o detector YOLO
        
        Args:
            model_path: Caminho para o modelo YOLO customizado (opcional)
            confidence_threshold: Limite de confiança para detecções
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Carrega o modelo
        if model_path and os.path.exists(model_path):
            print(f"Carregando modelo customizado: {model_path}")
            self.model = YOLO(model_path)
        else:
            # Usa YOLOv8 pré-treinado como fallback
            print("Carregando YOLOv8n pré-treinado...")
            self.model = YOLO('yolov8n.pt')
        
        print(f"Modelo carregado no dispositivo: {self.device}")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detecta objetos (preços) na imagem
        
        Args:
            image: Imagem em formato numpy array (BGR)
        
        Returns:
            Lista de dicionários com detecções
        """
        if self.model is None:
            raise Exception("Modelo não carregado")
        
        # Executa predição
        results = self.model.predict(
            image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        # Processa resultados
        for result in results:
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                # Extrai coordenadas
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Extrai confiança e classe
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class': class_name,
                    'class_id': class_id
                })
        
        # Ordena por confiança (maior primeiro)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def detect_and_visualize(self, image: np.ndarray, save_path: str = None) -> np.ndarray:
        """
        Detecta objetos e desenha as detecções na imagem
        
        Args:
            image: Imagem em formato numpy array (BGR)
            save_path: Caminho para salvar a imagem (opcional)
        
        Returns:
            Imagem com as detecções desenhadas
        """
        detections = self.detect(image)
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class']
            
            # Desenha retângulo
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Adiciona label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Fundo do label
            cv2.rectangle(
                result_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1
            )
            
            # Texto do label
            cv2.putText(
                result_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )
        
        if save_path:
            cv2.imwrite(save_path, result_image)
        
        return result_image
    
    def is_loaded(self) -> bool:
        """Verifica se o modelo está carregado"""
        return self.model is not None
    
    def set_confidence_threshold(self, threshold: float):
        """Define novo limite de confiança"""
        self.confidence_threshold = threshold
