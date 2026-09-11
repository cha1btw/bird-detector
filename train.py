"""
Скрипт для навчання YOLO-моделі для детекції птахів.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Seed встановлено на {seed} для відтворюваності")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Навчання YOLO-моделі для детекції птахів.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", default="yolo26n.pt", help="Базова модель для навчання")
    parser.add_argument("--data", default="data.yaml", help="Шлях до data.yaml")
    parser.add_argument("--epochs", type=int, default=25, help="Кількість епох навчання")
    parser.add_argument("--imgsz", type=int, default=640, help="Розмір зображень")
    parser.add_argument("--batch", type=int, default=16, help="Розмір батчу")
    parser.add_argument(
        "--device",
        default="0" if torch.cuda.is_available() else "cpu",
        help="Пристрій навчання (cpu або індекс GPU)",
    )
    parser.add_argument("--workers", type=int, default=8, help="Кількість worker-процесів")
    parser.add_argument("--seed", type=int, default=42, help="Seed для відтворюваності")
    parser.add_argument("--project", default="bird-detector", help="Папка для результатів")
    parser.add_argument("--name", default="yolo26_birds", help="Ім'я експерименту")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True, help="Зберігати чекпоінти")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True, help="Будувати графіки")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True, help="Показувати детальний лог")
    return parser


def resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def main():
    parser = build_parser()
    args = parser.parse_args()

    weights_path = resolve_path(args.weights)
    data_path = resolve_path(args.data)

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Файл ваг не знайдено: {weights_path}. "
            "Спочатку завантажте базову модель: python download_model.py"
        )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Файл {data_path} не знайдено. "
            "Спочатку завантажте датасет: python download_dataset.py"
        )

    config = {
        "weights": str(weights_path),
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "save": args.save,
        "plots": args.plots,
        "verbose": args.verbose,
    }

    print("=" * 60)
    print("Конфігурація навчання:")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 60)

    set_seed(config["seed"])

    print(f"\nЗавантаження базової моделі: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"\nПочаток навчання на {config['epochs']} епох...")
    print(f"Пристрій: {config['device']}")

    results = model.train(
        data=str(data_path),
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=config["device"],
        workers=config["workers"],
        seed=config["seed"],
        project=config["project"],
        name=config["name"],
        patience=config["patience"],
        save=config["save"],
        plots=config["plots"],
        verbose=config["verbose"],
    )

    print("\n" + "=" * 60)
    print("Навчання завершено!")
    print(f"Результати збережено в: {config['project']}/{config['name']}")
    print(f"Найкращі ваги: {config['project']}/{config['name']}/weights/best.pt")
    print("=" * 60)

    print("\nМетрики на validation-вибірці:")
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        print(f"mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
        print(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
        print(f"Recall: {metrics.get('metrics/recall(B)', 'N/A')}")


if __name__ == "__main__":
    main()