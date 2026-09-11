import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Детекція птахів на відео за допомогою YOLO model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True, help="Шлях до вхідного відеофайлу")
    parser.add_argument("--output", type=str, required=True, help="Шлях до вихідного відеофайлу")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(ROOT / "weights" / "best.pt"),
        help="Шлях до ваг моделі",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Поріг довіри для виявлення",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    weights_path = Path(args.weights).expanduser()

    if not input_path.exists():
        raise FileNotFoundError(f"Вхідне відео не знайдено: {input_path}")

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Файл ваг не знайдено: {weights_path}. "
            "Спочатку скачайте або збережіть модель у weights/best.pt"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не вдалося відкрити відео: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Не вдалося створити вихідний файл: {output_path}")

    print(f"Початок обробки відео: {input_path}...")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False, conf=args.conf)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Оброблено {frame_count} кадрів...")

    cap.release()
    out.release()

    print(f"Обробку завершено! Результат збережено у {output_path}")
    print(f"Загальна кількість кадрів: {frame_count}")


if __name__ == "__main__":
    main()
