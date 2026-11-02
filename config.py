import os
from pathlib import Path

# Базовый путь проекта
BASE_DIR = Path(__file__).resolve().parent

# Пути к каталогам
DATABASE_PATH = BASE_DIR / 'database' / 'faces.db'
KNOWN_FACES_DIR = BASE_DIR / 'known_faces'
UNKNOWN_FACES_DIR = BASE_DIR / 'unknown_faces'
EXPORTS_DIR = BASE_DIR / 'exports'
STATIC_TEMP_DIR = BASE_DIR / 'static' / 'temp'

# Создание каталогов при их отсутствии
for directory in [KNOWN_FACES_DIR, UNKNOWN_FACES_DIR, EXPORTS_DIR, STATIC_TEMP_DIR, BASE_DIR / 'database']:
    directory.mkdir(parents=True, exist_ok=True)

# Параметры распознавания
RECOGNITION_THRESHOLD = 0.6          # Порог схожести (меньше = строже)
FACE_DETECTION_MODEL = 'hog'          # 'hog' (быстро) или 'cnn' (точно, но требует GPU)
FACE_DETECTION_UPSAMPLES = 1          # Увеличение разрешения для детекции (0-2)

# Параметры обработки видео
FRAME_PROCESS_INTERVAL = 2.0          # Интервал между кадрами в автоматическом режиме (сек)
DEFAULT_CAMERA_ID = 0                  # ID встроенной камеры
CAMERA_RESOLUTION = (640, 480)         # Разрешение кадра (ширина, высота)

# Настройки веб-сервера
HOST = '127.0.0.1'
PORT = 5000
DEBUG = False                          # В продакшене всегда False

# Конфигурация Flask
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
MAX_CONTENT_LENGTH = 16 * 1024 * 1024   # Максимальный размер загружаемого файла (16MB)

# Настройки базы данных
DATABASE_TIMEOUT = 30.0
DATABASE_JOURNAL_MODE = 'WAL'

# Параметры логирования
LOG_LEVEL = 'INFO'
LOG_FILE = BASE_DIR / 'face_recognition.log'
LOG_MAX_SIZE = 10 * 1024 * 1024   # 10 MB
LOG_BACKUP_COUNT = 5

# Очистка данных
CLEANUP_UNKNOWN_FACES_DAYS = 30    # Удалять неизвестные лица старше N дней
CLEANUP_TEMP_FILES_DAYS = 7        # Удалять временные файлы старше N дней

# Производительность
USE_GPU = False                     # Использовать GPU (если доступно)