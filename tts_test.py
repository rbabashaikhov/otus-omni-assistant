
import time
import wave
from pathlib import Path

import torch


# Настройки
MODEL_NAME = "v5_ru"
SPEAKER = "aidar"
SAMPLE_RATE = 48000

OUTPUT_PATH = Path("samples/output/answer.wav")

TEXT = (
    "На фотографии изображён открытый холодильник "
    "с различными продуктами. В нём есть молоко, "
    "яйца, сыр, овощи и фрукты."
)


def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Используем процессор
    device = torch.device("cpu")
    torch.set_num_threads(4)

    # Загружаем модель
    print(f"Загружаем Silero {MODEL_NAME}...")

    start = time.perf_counter()

    model, example_text = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker=MODEL_NAME,
        trust_repo=True,
    )

    model.to(device)

    load_time = time.perf_counter() - start

    print(f"Модель загружена за {load_time:.2f} сек.")

    # Синтезируем речь
    print(f"Голос: {SPEAKER}")
    print(f"Текст: {TEXT}")

    start = time.perf_counter()

    with torch.inference_mode():
        audio = model.apply_tts(
            text=TEXT,
            speaker=SPEAKER,
            sample_rate=SAMPLE_RATE,
        )

    tts_time = time.perf_counter() - start

    # Преобразуем тензор в PCM 16-bit
    pcm = (
        audio.detach()
        .cpu()
        .clamp(-1, 1)
        .mul(32767)
        .to(torch.int16)
        .numpy()
    )

    # Сохраняем WAV
    with wave.open(str(OUTPUT_PATH), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())

    duration = len(audio) / SAMPLE_RATE

    print("\n--- TTS-001 ---")
    print(f"Файл: {OUTPUT_PATH}")
    print(f"Время синтеза: {tts_time:.2f} сек.")
    print(f"Длительность аудио: {duration:.2f} сек.")
    print(f"Размер файла: {OUTPUT_PATH.stat().st_size / 1024:.1f} КБ")


if __name__ == "__main__":
    main()
