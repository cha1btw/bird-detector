"""
Скрипт для скачивания датасета из Roboflow.
Для использования нужно указать свой API ключ Roboflow.
"""

from roboflow import Roboflow
import os

def download_dataset(api_key: str):
    """
    Скачивает датасет birds из Roboflow.
    
    Args:
        api_key: Ваш API ключ Roboflow
    """
    print("Инициализация Roboflow...")
    rf = Roboflow(api_key=api_key)
    
    print("Загрузка проекта birds-wnak6...")
    project = rf.workspace("evilsumrak").project("birds-wnak6")
    
    print(f"Скачивание версии 1 датасета...")
    version = project.version(1)
    dataset = version.download("yolov8")
    
    print(f"Датасет скачан в папку: {dataset.location}")
    print(f"Количество изображений: ~1.9k")
    print(f"Файл data.yaml создан в: {dataset.location}/data.yaml")

if __name__ == '__main__':
    # Получите ваш API ключ на https://app.roboflow.com/account/api-keys
    api_key = input("Введите ваш API ключ Roboflow: ")
    download_dataset(api_key)