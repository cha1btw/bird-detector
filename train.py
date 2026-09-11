"""
Скрипт для обучения YOLO26n модели на датасете птиц.
Использует детальную конфигурацию с фиксированным seed для воспроизводимости.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def set_seed(seed: int = 42):
    """
    Фиксирует random seed для воспроизводимости результатов.

    Args:
        seed: Значение seed для фиксации случайности
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed установлен на {seed} для воспроизводимости")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Обучение YOLO модели для детекции птиц.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="yolo26n.pt", help="Базовая YOLO модель")
    parser.add_argument("--data", default="data.yaml", help="Путь к файлу data.yaml")
    parser.add_argument("--epochs", type=int, default=25, help="Количество эпох обучения")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер изображений")
    parser.add_argument("--batch", type=int, default=16, help="Размер батча")
    parser.add_argument(
        "--device",
        default="0" if torch.cuda.is_available() else "cpu",
        help="Устройство обучения (cpu или cuda device index)",
    )
    parser.add_argument("--workers", type=int, default=8, help="Количество worker процессов")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости")
    parser.add_argument("--project", default="bird-detector", help="Папка для сохранения результатов")
    parser.add_argument("--name", default="yolo26_birds", help="Имя эксперимента")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True, help="Сохранять чекпоинты")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True, help="Формировать графики")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True, help="Подробный вывод")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    config = {
        'model': args.model,
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'seed': args.seed,
        'project': args.project,
        'name': args.name,
        'patience': args.patience,
        'save': args.save,
        'plots': args.plots,
        'verbose': args.verbose,
    }

    print("=" * 50)
    print("Конфигурация обучения:")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 50)

    if not Path(config['data']).exists():
        raise FileNotFoundError(
            f"Файл {config['data']} не найден. "
            "Сначала скачайте датасет: python download_dataset.py"
        )

    set_seed(config['seed'])

    print(f"\nЗагрузка базовой модели: {config['model']}")
    print("Примечание: Модель yolo26n.pt должна быть в корне проекта")
    model = YOLO(config['model'])

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

    print("\nМетрики на validation-выборке:")
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
        print(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
        print(f"Recall: {metrics.get('metrics/recall(B)', 'N/A')}")


if __name__ == '__main__':
    main()