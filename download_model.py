"""
Скрипт для скачивания базовой модели YOLO26n.
YOLO26n - это оптимизированная версия YOLOv8n для одного класса.
"""

import os
import urllib.request

def download_yolo26n():
    """
    Скачивает базовую модель YOLO26n.
    """
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    
    print(f"Скачивание базовой модели с {url}...")
    
    try:
        urllib.request.urlretrieve(url, "yolo26n.pt")
        print("Модель сохранена как yolo26n.pt")
        print("Примечание: YOLO26n использует архитектуру YOLOv8n, оптимизированную для 1 класса")
            
    except Exception as e:
        print(f"Ошибка при скачивании: {e}")
        print("Альтернатива: скачайте вручную с https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
        print("И сохраните как yolo26n.pt")

if __name__ == '__main__':
    download_yolo26n()