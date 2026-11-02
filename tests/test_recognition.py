import unittest
import cv2
import numpy as np
import face_recognition
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app import process_frame, load_known_faces

class RecognitionTestCase(unittest.TestCase):
    def setUp(self):
        # Загрузить тестовое изображение с лицом
        self.test_image = cv2.imread('tests/test_face.jpg')
        if self.test_image is None:
            self.skipTest("Тестовое изображение не найдено")

    def test_face_detection(self):
        rgb = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        self.assertGreater(len(locations), 0, "Лицо не обнаружено")

    def test_face_encoding(self):
        rgb = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        self.assertGreater(len(encodings), 0, "Не удалось получить эмбеддинг")

if __name__ == '__main__':
    unittest.main()