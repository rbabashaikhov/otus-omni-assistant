
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from pipeline import OmniAssistant


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "experiments" / "results"


def run_experiment(name, image_path, audio_path):
    """Запускает один эксперимент и сохраняет доказательства."""

    image_path = Path(image_path).resolve()
    audio_path = Path(audio_path).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    # Создаём отдельную папку для запуска
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    experiment_dir = RESULTS_DIR / f"{name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем исходные файлы
    saved_image = experiment_dir / (
        f"input_image{image_path.suffix.lower()}"
    )
    saved_audio = experiment_dir / (
        f"input_audio{audio_path.suffix.lower()}"
    )

    shutil.copy2(image_path, saved_image)
    shutil.copy2(audio_path, saved_audio)

    # Запускаем полный пайплайн
    assistant = OmniAssistant()

    result = assistant.run_omni_assistant(
        image=image_path,
        audio_query=audio_path,
    )

    # Сохраняем голосовой ответ
    saved_answer = experiment_dir / "answer.wav"

    shutil.copy2(
        result["audio_path"],
        saved_answer,
    )

    # Формируем результаты измерений
    report = {
        "experiment": name,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "models": {
            "asr": "whisper-base",
            "vlm": "qwen3.6-35b",
            "tts": "silero-v5-ru-aidar",
        },
        "inputs": {
            "image": saved_image.name,
            "audio": saved_audio.name,
        },
        "transcription": result["question"],
        "answer": result["answer"],
        "output_audio": saved_answer.name,
        "latency_seconds": result["timings"],
    }

    report_path = experiment_dir / "result.json"

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n===== ЭКСПЕРИМЕНТ СОХРАНЁН =====")
    print(f"Название: {name}")
    print(f"Папка: {experiment_dir}")
    print(f"Транскрипция: {result['question']}")
    print(f"Ответ: {result['answer']}")

    for stage, seconds in result["timings"].items():
        print(f"{stage}: {seconds:.2f} с")

    print(f"\nОтчёт: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Тестирование Omni Assistant"
    )

    parser.add_argument("--name", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--audio", required=True)

    args = parser.parse_args()

    run_experiment(
        name=args.name,
        image_path=args.image,
        audio_path=args.audio,
    )
