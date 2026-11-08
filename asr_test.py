
import time
from pathlib import Path

import torch
import whisper


# Настройки эксперимента
MODEL_NAME = "base"
AUDIO_PATH = Path("samples/audio/question.ogg")

# Проверяем, существует ли аудиофайл
if not AUDIO_PATH.exists():
    raise FileNotFoundError(
        f"Не найден аудиофайл: {AUDIO_PATH}\n"
        "Сначала запиши вопрос и сохрани его по этому пути."
    )

# Загружаем модель
print(f"Загружаем Whisper {MODEL_NAME}...")
start = time.perf_counter()

model = whisper.load_model(MODEL_NAME, device="cpu")

load_time = time.perf_counter() - start
print(f"Модель загружена за {load_time:.2f} сек.")

# Распознаём речь
print("Распознаём аудио...")
start = time.perf_counter()

result = model.transcribe(
    str(AUDIO_PATH),
    language="ru",
    task="transcribe",
    fp16=False,
)

asr_time = time.perf_counter() - start

# Выводим результат
print("\n--- Результат ---")
print(f"Текст: {result['text'].strip()}")
print(f"Время распознавания: {asr_time:.2f} сек.")
print(f"Потоки PyTorch: {torch.get_num_threads()}")
