
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import whisper
from scipy.io import wavfile


BASE_DIR = Path(__file__).resolve().parent

INPUT_AUDIO = (
    BASE_DIR / "samples/audio/fridge_question.ogg"
)

OUTPUT_DIR = (
    BASE_DIR / "experiments/results/asr_noise"
)

SAMPLE_RATE = 16000
SNR_LEVELS = [15, 5]

# Фиксируем генератор случайных чисел,
# чтобы шум можно было воспроизвести.
RANDOM_SEED = 42

# Заполним после проверки фактически
# произнесённого текста.
REFERENCE_TEXT = ""


def normalize_text(text):
    """Нормализация текста для сравнения."""
    text = text.lower().replace("ё", "е")
    return re.findall(r"\w+", text)


def word_error_rate(reference, hypothesis):
    """
    WER = (замены + удаления + вставки)
          / количество слов эталона.

    Используем расстояние Левенштейна
    на уровне слов.
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)

    if not ref:
        return None

    previous = list(range(len(hyp) + 1))

    for i, ref_word in enumerate(ref, start=1):
        current = [i]

        for j, hyp_word in enumerate(hyp, start=1):
            cost = int(ref_word != hyp_word)

            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )

        previous = current

    return previous[-1] / len(ref)


def add_noise(audio, snr_db, rng):
    """Добавляем белый шум с заданным SNR."""

    signal_power = np.mean(audio.astype(np.float64) ** 2)

    if signal_power <= 0:
        raise ValueError("Аудиосигнал пустой или тихий.")

    noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=audio.shape,
    )

    noise = noise.astype(np.float64)

    # Нормируем случайный шум
    # по среднеквадратичной мощности.
    noise /= np.sqrt(np.mean(noise ** 2))

    target_noise_power = (
        signal_power / (10 ** (snr_db / 10))
    )

    noise *= np.sqrt(target_noise_power)

    noisy_audio = audio.astype(np.float64) + noise

    return noisy_audio.astype(np.float32)


def main():
    if not INPUT_AUDIO.is_file():
        raise FileNotFoundError(INPUT_AUDIO)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Whisper сам декодирует OGG через FFmpeg
    # и преобразует его в 16 кГц, моно.
    original_audio = whisper.load_audio(
        str(INPUT_AUDIO)
    )

    rng = np.random.default_rng(RANDOM_SEED)

    recordings = {
        "original": original_audio,
    }

    for snr in SNR_LEVELS:
        recordings[f"snr_{snr}db"] = add_noise(
            original_audio,
            snr,
            rng,
        )

    # Общая нормализация предотвращает
    # clipping и сохраняет соотношение
    # мощности речи и шума.
    peak = max(
        float(np.max(np.abs(audio)))
        for audio in recordings.values()
    )

    scale = min(1.0, 0.95 / max(peak, 1e-12))

    for name, audio in recordings.items():
        pcm = np.round(
            np.clip(audio * scale, -1, 1) * 32767
        ).astype(np.int16)

        output_path = OUTPUT_DIR / f"{name}.wav"

        wavfile.write(
            output_path,
            SAMPLE_RATE,
            pcm,
        )

    print("Загружаем Whisper base...")
    torch.set_num_threads(4)

    model = whisper.load_model(
        "base",
        device="cpu",
    )

    results = []

    for name in recordings:
        audio_path = OUTPUT_DIR / f"{name}.wav"

        print(f"\nРаспознаём: {name}")

        start = time.perf_counter()

        result = model.transcribe(
            str(audio_path),
            language="ru",
            task="transcribe",
            fp16=False,
            temperature=0,
        )

        latency = time.perf_counter() - start
        transcription = result["text"].strip()

        wer = (
            word_error_rate(
                REFERENCE_TEXT,
                transcription,
            )
            if REFERENCE_TEXT.strip()
            else None
        )

        record = {
            "scenario": name,
            "audio_file": audio_path.name,
            "transcription": transcription,
            "latency_seconds": round(latency, 3),
            "wer": round(wer, 4) if wer is not None else None,
        }

        results.append(record)

        print("Текст:", transcription)
        print(f"Время: {latency:.2f} сек.")

        if wer is not None:
            print(f"WER: {wer:.2%}")

    report = {
        "model": "whisper-base",
        "device": "cpu",
        "sample_rate": SAMPLE_RATE,
        "random_seed": RANDOM_SEED,
        "noise_type": "white_gaussian",
        "snr_levels_db": SNR_LEVELS,
        "reference_text": REFERENCE_TEXT or None,
        "results": results,
    }

    report_path = OUTPUT_DIR / "report.json"

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n========== ИТОГ ==========")

    for result in results:
        print(
            f"{result['scenario']}: "
            f"{result['latency_seconds']:.2f} сек."
        )
        print(result["transcription"])

    print(f"\nОтчёт: {report_path}")


if __name__ == "__main__":
    main()
