import sys
import cv2
import numpy as np
import logging
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QScrollArea, QHBoxLayout
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtCore import QTimer, Qt, QSize
from deepface import DeepFace
import os
import time

# Настройка логирования
logging.basicConfig(filename='app.log', level=logging.DEBUG)


class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()

        # Изначально используем темную тему
        self.is_dark_theme = True

        # Элементы интерфейса
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.capture_button = QPushButton("Сделать фото", self)
        self.capture_button.clicked.connect(self.capture_photo)
        self.add_to_db_button = QPushButton("Добавить в базу", self)
        self.add_to_db_button.clicked.connect(self.add_to_database)
        self.toggle_theme_button = QPushButton(self)
        self.toggle_theme_button.setIcon(QIcon('sun_icon.png'))  # Иконка для кнопки смены темы
        self.toggle_theme_button.setIconSize(QSize(24, 24))
        self.toggle_theme_button.setStyleSheet("border: none;")  # Убираем границы
        self.toggle_theme_button.clicked.connect(self.toggle_theme)
        self.result_label = QLabel(self)
        self.result_label.setObjectName("result_label")
        self.photo_label = QLabel(self)
        self.photo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Основной макет
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.capture_button)
        main_layout.addWidget(self.add_to_db_button)
        main_layout.addWidget(self.photo_label)
        main_layout.addWidget(self.result_label)

        # Прокручиваемая область
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(QWidget())
        scroll_area.widget().setLayout(main_layout)

        # Макет для основного окна
        layout = QVBoxLayout()
        layout.addWidget(self.toggle_theme_button, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        layout.addWidget(scroll_area)

        # Устанавливаем макет
        self.setLayout(layout)

        # Захват видео с веб-камеры
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)

        # Загрузка эталонных изображений и создание эмбеддингов
        self.target_folder = "target"
        self.ensure_target_folder_exists()
        self.target_images = self.load_target_images()
        self.embeddings = self.compute_embeddings_for_targets()

        # Загрузка модели для обнаружения лиц
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Устанавливаем начальную тему
        self.apply_theme()

    def ensure_target_folder_exists(self):
        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)

    def load_target_images(self):
        images = []
        try:
            for image_name in os.listdir(self.target_folder):
                image_path = os.path.join(self.target_folder, image_name)
                if os.path.isfile(image_path):
                    images.append(image_path)
        except Exception as e:
            logging.error(f"Ошибка при загрузке эталонных изображений: {str(e)}")
        return images

    def compute_embeddings_for_targets(self):
        embeddings = []
        for img_path in self.target_images:
            try:
                result = DeepFace.represent(img_path=img_path, model_name="VGG-Face")
                embeddings.append(result[0]['embedding'])
            except Exception as e:
                logging.error(f"Ошибка при вычислении эмбеддинга для {img_path}: {str(e)}")
        return embeddings

    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        except Exception as e:
            logging.error(f"Ошибка при обновлении кадра: {str(e)}")

    def capture_photo(self):
        try:
            ret, frame = self.cap.read()
            if ret:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5,
                                                           minSize=(30, 30))

                if len(faces) == 0:
                    self.result_label.setText("Лица не обнаружены")
                    return

                # Определение самого центрального и ближайшего лица
                center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
                best_face = None
                min_distance = float('inf')
                for (x, y, w, h) in faces:
                    face_center_x, face_center_y = x + w // 2, y + h // 2
                    distance = np.sqrt((face_center_x - center_x) ** 2 + (face_center_y - center_y) ** 2)

                    if distance < min_distance:
                        min_distance = distance
                        best_face = (x, y, w, h)

                if best_face:
                    x, y, w, h = best_face
                    face_image = frame[y:y + h, x:x + w]

                    # Сохранение сделанного фото
                    cv2.imwrite("captured_face.jpg", cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))

                    result_text = "Not Recognized"
                    min_distance = float('inf')

                    for i, target_img_path in enumerate(self.target_images):
                        try:
                            result = DeepFace.verify(img1_path="captured_face.jpg", img2_path=target_img_path)
                            if result['verified']:
                                distance = result['distance']
                                if distance < min_distance:
                                    min_distance = distance
                                    result_text = f"Recognized, Distance: {distance:.2f}"
                        except Exception as e:
                            logging.error(f"Ошибка при сравнении с {target_img_path}: {str(e)}")
                            result_text = "Error Comparing"

                    # Добавление текста на изображение
                    frame_with_rect = frame.copy()
                    cv2.rectangle(frame_with_rect, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame_with_rect, result_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                                2)

                    # Преобразование изображения в формат QImage
                    rgb_image_with_rect = cv2.cvtColor(frame_with_rect, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image_with_rect.shape
                    bytes_per_line = ch * w
                    qt_image_with_rect = QImage(rgb_image_with_rect.data, w, h, bytes_per_line,
                                                QImage.Format.Format_RGB888)

                    # Обновление QLabel для отображения сделанного фото
                    self.photo_label.setPixmap(QPixmap.fromImage(qt_image_with_rect))
                    self.result_label.setText(result_text)
                else:
                    self.result_label.setText("Лицо не найдено")
        except Exception as e:
            logging.error(f"Ошибка при захвате фото: {str(e)}")
            self.result_label.setText("Ошибка захвата фото")

    def add_to_database(self):
        try:
            ret, frame = self.cap.read()
            if ret:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5,
                                                           minSize=(30, 30))

                if len(faces) == 0:
                    self.result_label.setText("Лица не обнаружены")
                    return

                # Определение самого центрального и ближайшего лица
                center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
                best_face = None
                min_distance = float('inf')

                for (x, y, w, h) in faces:
                    face_center_x, face_center_y = x + w // 2, y + h // 2
                    distance = np.sqrt((face_center_x - center_x) ** 2 + (face_center_y - center_y) ** 2)

                    if distance < min_distance:
                        min_distance = distance
                        best_face = (x, y, w, h)

                if best_face:
                    x, y, w, h = best_face
                    face_image = frame[y:y + h, x:x + w]

                    # Создание уникального имени файла на основе времени
                    unique_filename = f"person_{int(time.time())}.jpg"
                    new_image_path = os.path.join(self.target_folder, unique_filename)
                    cv2.imwrite(new_image_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))

                    # Обновление списка изображений и эмбеддингов
                    self.target_images.append(new_image_path)
                    self.embeddings = self.compute_embeddings_for_targets()

                    # Обновление UI
                    self.result_label.setText("Добавлено в базу данных")
                else:
                    self.result_label.setText("Лицо не найдено")
        except Exception as e:
            logging.error(f"Ошибка при добавлении в базу: {str(e)}")
            self.result_label.setText("Ошибка добавления в базу")

    def toggle_theme(self):
        self.is_dark_theme = not self.is_dark_theme
        self.apply_theme()

    def apply_theme(self):
        if self.is_dark_theme:
            self.setStyleSheet("""
                QWidget {
                    background-color: #2e2e2e;
                    color: #ffffff;
                }
                QPushButton {
                    background-color: #444444;
                    border: 2px solid #666666;
                    border-radius: 10px;
                    padding: 10px;
                    font-size: 16px;
                }
                QPushButton:hover {
                    background-color: #555555;
                }
                QLabel {
                    font-size: 16px;
                }
                QLabel#result_label {
                    font-size: 24px;
                    font-weight: bold;
                    color: #00ff00;
                }
            """)
        else:
            self.setStyleSheet("""
                QWidget {
                    background-color: #f0f0f0;
                    color: #000000;
                }
                QPushButton {
                    background-color: #dddddd;
                    border: 2px solid #cccccc;
                    border-radius: 10px;
                    padding: 10px;
                    font-size: 16px;
                }
                QPushButton:hover {
                    background-color: #cccccc;
                }
                QLabel {
                    font-size: 16px;
                }
                QLabel#result_label {
                    font-size: 24px;
                    font-weight: bold;
                    color: #ff0000;
                }
            """)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

    def closeEvent(self, event):
        try:
            self.cap.release()
        except Exception as e:
            logging.error(f"Ошибка при закрытии приложения: {str(e)}")


if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        window = FaceRecognitionApp()
        window.setWindowTitle("Face Recognition")
        window.resize(1280, 960)  # Начальный размер окна
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        logging.error(f"Ошибка при запуске приложения: {str(e)}")
