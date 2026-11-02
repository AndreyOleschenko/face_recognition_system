import os
import cv2
import face_recognition
import numpy as np
import sqlite3
import threading
import time
import base64
import pandas as pd
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_file, Response, jsonify, flash, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
from config import *

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Глобальные переменные состояния
auto_recognition_active = False
auto_recognition_thread = None
current_camera_id = 1
known_faces_cache = {'encodings': [], 'ids': [], 'names': []}

# ====================== Работа с базой данных ======================
def get_db_connection():
    """Возвращает соединение с БД, путь задан в config.py"""
    conn = sqlite3.connect(str(DATABASE_PATH))  # Path -> str для совместимости
    conn.row_factory = sqlite3.Row
    # Включаем поддержку внешних ключей и WAL режим
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(f"PRAGMA journal_mode = {DATABASE_JOURNAL_MODE}")
    return conn

def init_database():
    """Создание таблиц при первом запуске."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Таблица people
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                middle_name TEXT,
                image_path TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Таблица cameras
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Таблица events
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                camera_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                is_unknown INTEGER NOT NULL DEFAULT 0,
                image_path TEXT,
                confidence REAL,
                FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE SET NULL,
                FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
            )
        ''')
        # Индексы
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_camera_id ON events(camera_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_person_id ON events(person_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_people_name ON people(last_name, first_name)')
        conn.commit()

# ====================== Кэш известных лиц ======================
def update_known_faces_cache():
    """Загружает все известные лица из БД и вычисляет их эмбеддинги."""
    global known_faces_cache
    print("Обновление кэша известных лиц...")
    known_faces_cache = {'encodings': [], 'ids': [], 'names': []}
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, first_name, last_name, image_path FROM people')
        rows = cursor.fetchall()
    for row in rows:
        person_id, first_name, last_name, image_filename = row
        image_path = KNOWN_FACES_DIR / image_filename
        if not image_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model=FACE_DETECTION_MODEL)
        if face_locations:
            encodings = face_recognition.face_encodings(rgb, face_locations)
            if encodings:
                known_faces_cache['encodings'].append(encodings[0])
                known_faces_cache['ids'].append(person_id)
                known_faces_cache['names'].append(f"{last_name} {first_name}")
    print(f"Кэш обновлён. Загружено лиц: {len(known_faces_cache['encodings'])}")

def load_known_faces():
    """Возвращает кортеж (эмбеддинги, id, имена) из кэша. Если кэш пуст, обновляет."""
    if not known_faces_cache['encodings']:
        update_known_faces_cache()
    return (known_faces_cache['encodings'],
            known_faces_cache['ids'],
            known_faces_cache['names'])

# ====================== Обработка видеокадра ======================
def process_frame(frame, camera_id):
    """Детекция и распознавание лиц на одном кадре. Возвращает кадр с аннотациями."""
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model=FACE_DETECTION_MODEL)
        if not face_locations:
            return frame

        face_encodings = face_recognition.face_encodings(rgb, face_locations)
        known_encodings, known_ids, known_names = load_known_faces()

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            person_id = None
            person_name = "Неизвестное лицо"
            is_unknown = True
            confidence = 0.0

            if known_encodings:
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_idx = np.argmin(distances)
                best_distance = distances[best_idx]
                if best_distance < RECOGNITION_THRESHOLD:
                    person_id = known_ids[best_idx]
                    person_name = known_names[best_idx]
                    is_unknown = False
                    confidence = 1.0 - best_distance

            # Сохраняем неизвестное лицо
            saved_image_path = None
            if is_unknown:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                filename = f"unknown_{timestamp}.jpg"
                filepath = UNKNOWN_FACES_DIR / filename
                face_img = frame[top:bottom, left:right]
                if face_img.size > 0:
                    cv2.imwrite(str(filepath), face_img)
                    saved_image_path = filename

            # Запись события в БД
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO events (person_id, camera_id, timestamp, is_unknown, image_path, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (person_id, camera_id, datetime.now().isoformat(), is_unknown, saved_image_path, confidence))
                conn.commit()

            # Визуализация
            frame = draw_annotation(frame, top, right, bottom, left, person_name)

        return frame
    except Exception as e:
        print(f"Ошибка в process_frame: {e}")
        return frame

def draw_annotation(frame, top, right, bottom, left, name):
    """Рисует рамку и подпись на кадре."""
    try:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        color = (255, 0, 0) if name == "Неизвестное лицо" else (0, 255, 0)
        draw.rectangle([left, top, right, bottom], outline=color, width=2)

        # Попытка загрузить шрифт с поддержкой кириллицы
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
            except:
                font = ImageFont.load_default()
        draw.text((left, top - 20), name, fill=color, font=font)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Ошибка в draw_annotation: {e}")
        return frame

# ====================== Генератор видеопотока ======================
def generate_frames(camera_id):
    """Генератор кадров для потоковой передачи (MJPEG)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT url FROM cameras WHERE id = ?', (camera_id,))
        row = cursor.fetchone()
    if not row:
        yield b'Camera not found'
        return
    url = row['url']
    cap = cv2.VideoCapture(0 if url == '0' else url)
    if not cap.isOpened():
        yield b'Cannot open camera'
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = process_frame(frame, camera_id)
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)  # ~30 FPS
    cap.release()

# ====================== Маршруты Flask ======================
@app.route('/')
def index():
    """Главная страница с дашбордом."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        # Статистика
        cursor.execute("SELECT COUNT(*) FROM people")
        people_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM cameras")
        cameras_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM events")
        events_count = cursor.fetchone()[0]

        # Последние события
        cursor.execute('''
            SELECT e.timestamp, p.first_name, p.last_name, c.name, e.is_unknown
            FROM events e
            LEFT JOIN people p ON e.person_id = p.id
            LEFT JOIN cameras c ON e.camera_id = c.id
            ORDER BY e.timestamp DESC
            LIMIT 15
        ''')
        events = cursor.fetchall()

        # Список людей
        cursor.execute("SELECT * FROM people ORDER BY last_name, first_name")
        people = cursor.fetchall()

        # Список камер
        cursor.execute("SELECT * FROM cameras ORDER BY name")
        cameras = cursor.fetchall()

    return render_template('index.html',
                           people_count=people_count,
                           cameras_count=cameras_count,
                           events_count=events_count,
                           events=events,
                           people=people,
                           cameras=cameras,
                           auto_recognition=auto_recognition_active)

@app.route('/export_events_range', methods=['GET', 'POST'])
def export_events_range():
    if request.method == 'POST':
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        # Здесь можно реализовать экспорт за период
        pass
    today = datetime.now().strftime('%Y-%m-%d')
    return render_template('export_range.html', today=today)

@app.route('/known_faces/<filename>')
def serve_known_face(filename):
    try:
        return send_from_directory(KNOWN_FACES_DIR, filename)
    except:
        return send_from_directory('static/img', 'default-face.jpg')

@app.route('/unknown_faces/<filename>')
def serve_unknown_face(filename):
    try:
        return send_from_directory(UNKNOWN_FACES_DIR, filename)
    except:
        return send_from_directory('static/img', 'default-face.jpg')

@app.route('/add_person', methods=['GET', 'POST'])
def add_person():
    """Добавление нового человека."""
    if request.method == 'POST':
        first_name = request.form['first_name'].strip()
        last_name = request.form['last_name'].strip()
        middle_name = request.form.get('middle_name', '').strip()
        image_file = request.files.get('image')

        if not first_name or not last_name:
            flash('Имя и фамилия обязательны', 'error')
            return redirect(request.url)
        if not image_file:
            flash('Выберите изображение', 'error')
            return redirect(request.url)

        # Сохраняем файл
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(f"{last_name}_{first_name}_{timestamp}.jpg")
        filepath = KNOWN_FACES_DIR / filename
        image_file.save(filepath)

        # Добавляем запись в БД
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO people (first_name, last_name, middle_name, image_path)
                VALUES (?, ?, ?, ?)
            ''', (first_name, last_name, middle_name, filename))
            conn.commit()

        update_known_faces_cache()
        flash(f'Человек {last_name} {first_name} добавлен', 'success')
        return redirect(url_for('index'))

    return render_template('add_person.html')

@app.route('/delete_person/<int:person_id>')
def delete_person(person_id):
    """Удаление человека (запись из БД и файл изображения)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT image_path FROM people WHERE id = ?", (person_id,))
        row = cursor.fetchone()
        if row:
            image_path = KNOWN_FACES_DIR / row['image_path']
            if image_path.exists():
                image_path.unlink()
        cursor.execute("DELETE FROM people WHERE id = ?", (person_id,))
        conn.commit()
    update_known_faces_cache()
    flash('Человек удалён', 'success')
    return redirect(url_for('index'))

@app.route('/cameras', methods=['GET', 'POST'])
def cameras_page():
    """Управление камерами."""
    if request.method == 'POST':
        name = request.form['name'].strip()
        url = request.form['url'].strip()
        if name and url:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO cameras (name, url) VALUES (?, ?)", (name, url))
                conn.commit()
            flash('Камера добавлена', 'success')
        else:
            flash('Заполните все поля', 'error')
        return redirect(url_for('cameras_page'))

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cameras ORDER BY name")
        cameras = cursor.fetchall()
    return render_template('cameras.html', cameras=cameras)

@app.route('/delete_camera/<int:camera_id>')
def delete_camera(camera_id):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
        conn.commit()
    flash('Камера удалена', 'success')
    return redirect(url_for('cameras_page'))

@app.route('/video_stream/<int:camera_id>')
def video_stream(camera_id):
    """Эндпоинт для MJPEG-потока."""
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream')
def stream_selector():
    """Страница выбора камеры для просмотра."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cameras ORDER BY name")
        cameras = cursor.fetchall()
    return render_template('stream_selector.html', cameras=cameras)

@app.route('/stream/<int:camera_id>')
def stream_view(camera_id):
    """Просмотр видеопотока с конкретной камеры."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,))
        camera = cursor.fetchone()
    if not camera:
        flash('Камера не найдена', 'error')
        return redirect(url_for('stream_selector'))
    return render_template('stream_view.html', camera=camera)

@app.route('/single_recognition/<int:camera_id>')
def single_recognition(camera_id):
    """Разовое распознавание (один кадр)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,))
        camera = cursor.fetchone()
    if not camera:
        flash('Камера не найдена', 'error')
        return redirect(url_for('index'))

    url = camera['url']
    cap = cv2.VideoCapture(0 if url == '0' else url)
    if not cap.isOpened():
        flash('Не удалось открыть камеру', 'error')
        return redirect(url_for('index'))

    success, frame = cap.read()
    cap.release()
    if not success:
        flash('Не удалось захватить кадр', 'error')
        return redirect(url_for('index'))

    processed = process_frame(frame, camera_id)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"single_{timestamp}.jpg"
    filepath = STATIC_TEMP_DIR / filename
    cv2.imwrite(str(filepath), processed)

    return render_template('single_result.html',
                           image_url=url_for('static', filename=f'temp/{filename}'),
                           camera_name=camera['name'],
                           timestamp=timestamp)

@app.route('/events')
def events_page():
    """Журнал событий с фильтрацией."""
    return render_template('events.html')

@app.route('/api/events')
def api_events():
    """API для получения событий (с фильтрацией и пагинацией)."""
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    camera_id = request.args.get('camera_id', type=int)
    is_unknown = request.args.get('is_unknown', type=int)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = '''
            SELECT e.id, e.timestamp, p.id as person_id, p.last_name, p.first_name,
                   c.id as camera_id, c.name as camera_name, e.is_unknown, e.image_path, e.confidence
            FROM events e
            LEFT JOIN people p ON e.person_id = p.id
            LEFT JOIN cameras c ON e.camera_id = c.id
            WHERE 1=1
        '''
        params = []
        if camera_id:
            query += ' AND e.camera_id = ?'
            params.append(camera_id)
        if is_unknown is not None:
            query += ' AND e.is_unknown = ?'
            params.append(is_unknown)

        # Подсчёт общего количества
        count_query = f"SELECT COUNT(*) FROM ({query})"
        cursor.execute(count_query, params)
        total = cursor.fetchone()[0]

        # Сортировка и пагинация
        query += ' ORDER BY e.timestamp DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        cursor.execute(query, params)
        rows = cursor.fetchall()

        events = []
        for row in rows:
            event = dict(row)
            # Форматирование времени
            dt = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
            event['display_time'] = dt.strftime('%Y-%m-%d %H:%M:%S')
            events.append(event)

    return jsonify({'success': True, 'events': events, 'total': total})

@app.route('/export_events')
def export_events():
    """Экспорт всех событий в CSV."""
    with get_db_connection() as conn:
        df = pd.read_sql_query('''
            SELECT e.timestamp, p.last_name, p.first_name, c.name as camera,
                   CASE WHEN e.is_unknown = 1 THEN 'Неизвестный' ELSE 'Известный' END as status,
                   e.confidence
            FROM events e
            LEFT JOIN people p ON e.person_id = p.id
            LEFT JOIN cameras c ON e.camera_id = c.id
            ORDER BY e.timestamp DESC
        ''', conn)

    filename = f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = EXPORTS_DIR / filename
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    return send_file(filepath, as_attachment=True, download_name=filename)

@app.route('/toggle_auto_recognition', methods=['POST'])
def toggle_auto_recognition():
    """Включение/выключение автоматического распознавания."""
    global auto_recognition_active, auto_recognition_thread, current_camera_id
    camera_id = request.form.get('camera_id', 1, type=int)

    if not auto_recognition_active:
        # Запуск
        auto_recognition_active = True
        current_camera_id = camera_id
        auto_recognition_thread = threading.Thread(target=auto_recognition_worker,
                                                   args=(camera_id, FRAME_PROCESS_INTERVAL),
                                                   daemon=True)
        auto_recognition_thread.start()
        flash('Автоматическое распознавание запущено', 'success')
    else:
        # Остановка
        auto_recognition_active = False
        flash('Автоматическое распознавание остановлено', 'info')
    return redirect(url_for('index'))

def auto_recognition_worker(camera_id, interval):
    """Фоновый поток для автоматического распознавания."""
    global auto_recognition_active
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT url FROM cameras WHERE id = ?", (camera_id,))
        row = cursor.fetchone()
    if not row:
        print(f"Камера {camera_id} не найдена")
        auto_recognition_active = False
        return
    url = row['url']
    cap = cv2.VideoCapture(0 if url == '0' else url)
    if not cap.isOpened():
        print(f"Не удалось открыть камеру {camera_id}")
        auto_recognition_active = False
        return

    while auto_recognition_active:
        success, frame = cap.read()
        if success:
            process_frame(frame, camera_id)
        time.sleep(interval)

    cap.release()
    print(f"Поток авто-распознавания для камеры {camera_id} остановлен")

@app.route('/api/auto_recognition_status')
def auto_recognition_status():
    return jsonify({'active': auto_recognition_active, 'camera_id': current_camera_id})

# ====================== Инициализация при запуске ======================
# Создаём таблицы БД, если их нет (выполняется при импорте модуля)
init_database()

# ====================== Запуск (только для локальной разработки) ======================
if __name__ == '__main__':
    # Предварительно загружаем кэш для удобства (можно и не делать)
    update_known_faces_cache()
    # Запускаем встроенный сервер Flask
    app.run(host='0.0.0.0', port=8000, debug=True)