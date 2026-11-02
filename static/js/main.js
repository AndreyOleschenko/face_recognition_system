// Обновление статуса авто-распознавания каждые 3 секунды
function updateAutoRecognitionStatus() {
    fetch('/api/auto_recognition_status')
        .then(response => response.json())
        .then(data => {
            const statusElement = document.getElementById('auto-recognition-status');
            const buttonElement = document.querySelector('form[action*="toggle_auto_recognition"] button');
            if (statusElement) {
                statusElement.textContent = data.active ? 'ВКЛ' : 'ВЫКЛ';
                statusElement.className = data.active ? 'status-active' : 'status-inactive';
            }
            if (buttonElement) {
                buttonElement.textContent = data.active ? 'Остановить' : 'Запустить';
                buttonElement.className = data.active ? 'btn btn-danger' : 'btn btn-success';
            }
        })
        .catch(error => console.error('Ошибка обновления статуса:', error));
}

// Автоматическое обновление
if (document.getElementById('auto-recognition-status')) {
    setInterval(updateAutoRecognitionStatus, 3000);
}

// Подтверждение удаления
document.querySelectorAll('.btn-delete').forEach(btn => {
    btn.addEventListener('click', (e) => {
        if (!confirm('Вы уверены?')) {
            e.preventDefault();
        }
    });
});

// Предпросмотр изображения при добавлении человека
function previewImage(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const preview = document.getElementById('preview');
            if (preview) {
                preview.src = e.target.result;
                preview.style.display = 'block';
            }
        };
        reader.readAsDataURL(input.files[0]);
    }
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    // Применить предпросмотр, если есть поле file
    const fileInput = document.getElementById('image');
    if (fileInput) {
        fileInput.addEventListener('change', function() { previewImage(this); });
    }
});