import cv2
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Детекція птахів на відео за допомогою YOLO26")
    parser.add_argument('--input', type=str, required=True, help="Шлях до вхідного відеофайлу")
    parser.add_argument('--output', type=str, required=True, help="Шлях для збереження обробленого відео")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Завантаження навчених ваг
    model = YOLO('weights/best.pt')
    
    # Відкриття відеофайлу
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Помилка: Не вдалося відкрити відео {args.input}")
        return

    # Отримання параметрів відео для збереження
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Налаштування кодека та VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Початок обробки відео: {args.input}...")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Інференс моделі на кадрі
        results = model(frame, verbose=False)
        
        # Метод plot() малює bounding box, клас та confidence
        annotated_frame = results[0].plot() 
        
        # Запис кадру у вихідний файл
        out.write(annotated_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Оброблено {frame_count} кадрів...")

    cap.release()
    out.release()
    print(f"Обробку завершено! Результат збережено у {args.output}")

if __name__ == '__main__':
    main()
