import cv2
import numpy as np
import argparse
from ultralytics import YOLO  # Импортируем YOLOv8 из ultralytics

# Функция для выделения линий разметки
def detect_road_markings(frame, lowCannyThresh=100, highCannyThresh=200):
    """
    Обрабатывает изображение для выделения линий разметки на дороге.
    
    Аргументы:
        frame (np.ndarray): Входное изображение в формате BGR.
        lowCannyThresh (int): Минимальный порог для Canny edge detection.
        highCannyThresh (int): Максимальный порог для Canny edge detection.
    
    Возвращает:
        output (np.ndarray): Копия изображения с нарисованными линиями разметки.
        lines (list): Список координат линий разметки.
        line_boxes (list): Список боксов для линий разметки.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, lowCannyThresh, highCannyThresh)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    
    output = frame.copy()
    line_boxes = []  # Список для хранения координат боксов вокруг линий разметки
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_boxes.append([x1, y1, x2, y2])  # Формируем бокс вокруг линии

    return output, lines, line_boxes

# Функция для детекции людей и машин с использованием YOLOv8
def detect_objects(frame, model):
    """
    Детектирует людей и машины на изображении с помощью модели YOLOv8.
    
    Аргументы:
        frame (np.ndarray): Входное изображение в формате BGR.
        model (YOLO): Загруженная модель YOLOv8.
    
    Возвращает:
        people_boxes (np.ndarray): Список прямоугольников вокруг людей.
        car_boxes (np.ndarray): Список прямоугольников вокруг машин.
    """
    results = model(frame)  # Детекция объектов с использованием YOLOv8
    
    # Извлекаем результат для первого изображения
    boxes = results[0].boxes  # Получаем bounding boxes для первого результата
    people_boxes = []
    car_boxes = []

    for box in boxes:
        # Извлекаем координаты коробки: [x_center, y_center, width, height]
        if box.cls == 0:  # Проверяем, что класс объекта - "человек" (class 0 в COCO dataset)
            x_center, y_center, width, height = box.xywh[0]
            xmin = int(x_center - width / 2)
            ymin = int(y_center - height / 2)
            xmax = int(x_center + width / 2)
            ymax = int(y_center + height / 2)
            people_boxes.append([xmin, ymin, xmax, ymax])  # Добавляем координаты для прямоугольника

        elif box.cls == 2:  # Проверяем, что класс объекта - "машина" (class 2 в COCO dataset)
            x_center, y_center, width, height = box.xywh[0]
            xmin = int(x_center - width / 2)
            ymin = int(y_center - height / 2)
            xmax = int(x_center + width / 2)
            ymax = int(y_center + height / 2)
            car_boxes.append([xmin, ymin, xmax, ymax])  # Добавляем координаты для прямоугольника

    return np.array(people_boxes), np.array(car_boxes)  # Преобразуем в numpy массив для удобства работы


# Проверка пересечения двух отрезков
def do_lines_intersect(line1, line2):
    """
    Проверяет, пересекаются ли два отрезка.
    
    Аргументы:
        line1 (list): Координаты первого отрезка (x1, y1, x2, y2).
        line2 (list): Координаты второго отрезка (x1, y1, x2, y2).
    
    Возвращает:
        bool: True, если отрезки пересекаются, иначе False.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    # Используем детектор пересечения отрезков через векторное произведение
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw((x1, y1), (x3, y3), (x4, y4)) != ccw((x2, y2), (x3, y3), (x4, y4)) and ccw((x1, y1), (x2, y2), (x3, y3)) != ccw((x1, y1), (x2, y2), (x4, y4))

# Проверка пересечения линии с прямоугольником
def is_line_in_rectangle(line, rect):
    """
    Проверяет, пересекает ли линия разметки прямоугольник.
    
    Аргументы:
        line (list): Координаты линии (x1, y1, x2, y2).
        rect (list): Координаты прямоугольника (xmin, ymin, xmax, ymax).
    
    Возвращает:
        bool: True, если линия пересекает прямоугольник, иначе False.
    """
    x1, y1, x2, y2 = line
    xmin, ymin, xmax, ymax = rect

    # Проверяем, пересекает ли линия прямоугольник
    rect_lines = [
        [xmin, ymin, xmax, ymin],  # Верхняя граница
        [xmin, ymax, xmax, ymax],  # Нижняя граница
        [xmin, ymin, xmin, ymax],  # Левая граница
        [xmax, ymin, xmax, ymax]   # Правая граница
    ]
    
    for rect_line in rect_lines:
        if do_lines_intersect([x1, y1, x2, y2], rect_line):
            return True  # Линия пересекает прямоугольник
    return False

# Проверка пересечения с нижней частью прямоугольника
def is_point_near_line(line, x_start, x_end, y, threshold=10):
    """
    Проверяет, пересекает ли линия разметки нижнюю часть прямоугольника.
    
    Аргументы:
        line (list): Координаты линии (x1, y1, x2, y2).
        x_start (int): Начальная точка по оси X.
        x_end (int): Конечная точка по оси X.
        y (int): Координата по оси Y, где проверяется пересечение.
        threshold (int): Пороговое значение расстояния от точки до линии для учета пересечения.
    
    Возвращает:
        bool: True, если линия пересекает прямоугольник, иначе False.
    """
    x1, y1, x2, y2 = line
    
    # Для каждой точки от x_start до x_end (по оси X) проверяем расстояние до линии
    for x in range(x_start, x_end + 1):
        num = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        denom = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distance = num / denom
        if distance < threshold:
            return True
    return False

# Функция для вывода предупреждения
def draw_warning(frame, message):
    """
    Отображает предупреждающее сообщение на экране.
    
    Аргументы:
        frame (np.ndarray): Изображение на экране.
        message (str): Сообщение для отображения.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, message, (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

# Функция для рисования прямоугольников вокруг людей
def draw_rectangle(frame, boxes, color=(0, 0, 255), thickness=3):
    """
    Рисует прямоугольники вокруг объектов (людей или машин).
    
    Аргументы:
        frame (np.ndarray): Изображение на экране.
        boxes (list): Список координат прямоугольников.
        color (tuple): Цвет прямоугольника (по умолчанию красный).
        thickness (int): Толщина линии прямоугольника.
    """
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)  # Красный прямоугольник вокруг человека

# Функция для рисования боксов вокруг линий разметки (зелёный цвет)
def draw_line_boxes(frame, line_boxes, color=(0, 255, 0), thickness=2):
    """
    Рисует прямоугольники вокруг линий разметки.
    
    Аргументы:
        frame (np.ndarray): Изображение на экране.
        line_boxes (list): Список координат линий разметки.
        color (tuple): Цвет прямоугольников (по умолчанию зелёный).
        thickness (int): Толщина линии прямоугольников.
    """
    for (x1, y1, x2, y2) in line_boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)  # Зеленый бокс вокруг линии разметки

# Парсинг аргументов командной строки с помощью argparse
def parse_args():
    """
    Разбирает аргументы командной строки для пути к видеофайлу.
    
    Возвращает:
        args (Namespace): Разобранные аргументы командной строки.
    """
    parser = argparse.ArgumentParser(description="Video processing with YOLOv8 for detecting people and cars.")
    parser.add_argument('--video', type=str, required=True, help="Path to the input video file.")
    return parser.parse_args()

def main():
    # Получаем параметры из командной строки
    args = parse_args()

    video_path = args.video  # Путь к видеофайлу
    model_path = "yolov8n.pt"  # Модель по умолчанию

    video_capture = cv2.VideoCapture(video_path)

    # Загружаем модель YOLOv8
    model = YOLO(model_path)  # Загрузить модель YOLOv8

    prev_frame = None
    while True:
        good, frame = video_capture.read()
        if not good:
            print("Не удалось загрузить кадр или достигнут конец видео")
            break

        # Детекция разметки
        road_markings, lines, line_boxes = detect_road_markings(frame)

        # Детекция людей и машин
        people_boxes, car_boxes = detect_objects(frame, model)

        # Отладочный вывод: проверим, детектируются ли люди
        if len(people_boxes) > 0:
            print(f"Detected {len(people_boxes)} people")
        
        if len(car_boxes) > 0:
            print(f"Detected {len(car_boxes)} cars")

        # Рисуем прямоугольники вокруг людей
        draw_rectangle(frame, people_boxes)

        # Рисуем прямоугольники вокруг машин
        draw_rectangle(frame, car_boxes, color=(0, 255, 255))  # Желтые прямоугольники для машин

        # Рисуем боксы для линий разметки
        draw_line_boxes(frame, line_boxes)

        # Проверка, пересекает ли человек разметку (проверка для нижней части человека)
        for (x1, y1, x2, y2) in people_boxes:
            # Получаем точки по нижней границе (от x до x + w) на уровне y = ymax
            x_start = int(x1)
            x_end = int(x2)
            person_bottom_y = int(y2)

            for line in lines:
                # Исключаем линии, которые пересекают прямоугольник человека
                if not is_line_in_rectangle(line[0], (x1, y1, x2, y2)):
                    if is_point_near_line(line[0], x_start, x_end, person_bottom_y, threshold=10):
                        draw_warning(frame, "DANGER: Person crossing road marking")
                        break
        
        # Проверка, пересекает ли машина разметку
        for (x1, y1, x2, y2) in car_boxes:
            # Получаем точки по нижней границе (от x до x + w) на уровне y = ymax
            x_start = int(x1)
            x_end = int(x2)
            car_bottom_y = int(y2)

            for line in lines:
                # Исключаем линии, которые пересекают прямоугольник машины
                if not is_line_in_rectangle(line[0], (x1, y1, x2, y2)):
                    if is_point_near_line(line[0], x_start, x_end, car_bottom_y, threshold=10):
                        draw_warning(frame, "DANGER: Car crossing road marking")
                        break

        # Устанавливаем размер окна
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video", 800, 600)

        # Отображение на экране
        cv2.imshow("Video", frame)

        # Выход из программы по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame = frame

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
