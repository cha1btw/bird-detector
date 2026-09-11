"""
Скрипт для обучения YOLO26n модели на датасете птиц.
Использует детальную конфигурацию с фиксированным seed для воспроизводимости.
"""

import os
import random
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path

def set_seed(seed: int = 42):
    """
    Фиксирует random seed для воспроизводимости результатов.
    
    Args:
        seed: Значение seed для фиксации случайности
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed установлен на {seed} для воспроизводимости")

def main():
    # Конфигурация обучения
    config = {
        'model': 'yolo26n.pt',           # Базовая модель YOLO26n
        'data': 'data.yaml',             # Путь к data.yaml от Roboflow
        'epochs': 25,                    # Количество эпох
        'imgsz': 640,                    # Размер изображений
        'batch': 16,                     # Размер батча
        'device': '0' if torch.cuda.is_available() else 'cpu',  # GPU если доступен
        'workers': 8,                     # Количество workers для dataloader
        'seed': 42,                      # Seed для воспроизводимости
        'project': 'bird-detector',      # Папка для сохранения результатов
        'name': 'yolo26_birds',          # Имя эксперимента
        'patience': 10,                  # Early stopping patience
        'save': True,                    # Сохранять чекпоинты
        'plots': True,                   # Строить графики
        'verbose': True,                 # Подробный вывод
    }
    
    print("=" * 50)
    print("Конфигурация обучения:")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    # Проверка наличия data.yaml
    if not Path(config['data']).exists():
        raise FileNotFoundError(
            f"Файл {config['data']} не найден. "
            "Сначала скачайте датасет: python download_dataset.py"
        )
    
    # Установка seed для воспроизводимости
    set_seed(config['seed'])
    
    # Инициализация модели YOLO26n
    print(f"\nЗагрузка базовой модели: {config['model']}")
    print(f"Примечание: Модель yolo26n.pt должна быть в корне проекта")
    model = YOLO(config['model'])
    
    # Запуск процесса обучения
    print(f"\nНачало обучения на {config['epochs']} эпох...")
    print(f"Устройство: {config['device']}")
    
    results = model.train(
        data=config['data'],
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        batch=config['batch'],
        device=config['device'],
        workers=config['workers'],
        seed=config['seed'],
        project=config['project'],
        name=config['name'],
        patience=config['patience'],
        save=config['save'],
        plots=config['plots'],
        verbose=config['verbose'],
    )
    
    print("\n" + "=" * 50)
    print("Обучение завершено!")
    print(f"Результаты сохранены в: {config['project']}/{config['name']}")
    print(f"Лучшие веса: {config['project']}/{config['name']}/weights/best.pt")
    print("=" * 50)
    
    # Вывод метрик
    print("\nМетрики на validation-выборке:")
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
        print(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
        print(f"Recall: {metrics.get('metrics/recall(B)', 'N/A')}")

if __name__ == '__main__':
    main()