
import base64
import mimetypes
import os
import time
import wave
from uuid import uuid4
from pathlib import Path

import torch
import whisper
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

# Пути относительно расположения pipeline.py
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "samples" / "output"

# Настройки моделей
WHISPER_MODEL = "base"
SILERO_MODEL = "v5_ru"
SILERO_VOICE = "aidar"
SAMPLE_RATE = 48000


class OmniAssistant:
    def __init__(self):
        """
        Инициализируем модели и API-клиент один раз.
        """
        print("Инициализация Omni Assistant...")

        self.folder = os.environ["YANDEX_CLOUD_FOLDER"]
        self.model_name = os.environ["YANDEX_CLOUD_MODEL"]

        self.client = OpenAI(
            api_key=os.environ["YANDEX_CLOUD_API_KEY"],
            base_url="https://ai.api.cloud.yandex.net/v1",
            project=self.folder,
            timeout=90.0,
        )

        # 1. Whisper
        print("Загружаем Whisper...")
        self.whisper_model = whisper.load_model(
            WHISPER_MODEL,
            device="cpu",
        )

        # 2. Silero — загружаем из локального кэша
        print("Загружаем Silero из локального кэша...")

        silero_repo = (
            Path.home()
            / ".cache/torch/hub/snakers4_silero-models_master"
        )

        if not silero_repo.is_dir():
            raise FileNotFoundError(
                f"Локальный кэш Silero не найден: {silero_repo}"
            )

        self.silero_model, _ = torch.hub.load(
            repo_or_dir=str(silero_repo),
            model="silero_tts",
            language="ru",
            speaker=SILERO_MODEL,
            source="local",
        )

        self.silero_model.to("cpu")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print("Все компоненты готовы!\n")

    def recognize_speech(self, audio_path):
        """
        ASR: аудио -> текст.
        """
        result = self.whisper_model.transcribe(
            str(audio_path),
            language="ru",
            task="transcribe",
            fp16=False,
        )

        return result["text"].strip()

    def analyze_image(self, image_path, question):
        """
        VLM: изображение + вопрос -> текстовый ответ.
        """
        image_path = Path(image_path)

        mime_type, _ = mimetypes.guess_type(
            image_path.name
        )

        if mime_type not in ("image/png", "image/jpeg"):
            raise ValueError(
                "Поддерживаются изображения PNG и JPEG"
            )

        image_base64 = base64.b64encode(
            image_path.read_bytes()
        ).decode("ascii")

        image_url = (
            f"data:{mime_type};base64,{image_base64}"
        )

        response = self.client.responses.create(
            model=(
                f"gpt://{self.folder}/{self.model_name}"
            ),
            instructions=(
                "Ты мультимодальный голосовой ассистент. "
                "Отвечай на русском языке кратко, "
                "не более двух предложений. "
                "Опирайся только на изображение "
                "и вопрос пользователя. "
                "Не придумывай невидимые детали."
            ),
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": question,
                        },
                        {
                            "type": "input_image",
                            "image_url": image_url,
                        },
                    ],
                }
            ],
            reasoning={"effort": "none"},
            max_output_tokens=2048,
            store=False,
        )

        print("\n--- Диагностика VLM ---")
        print("Status:", response.status)
        print("Incomplete details:", response.incomplete_details)
        print("Usage:", response.usage)
        print("Output:", repr(response.output_text))

        if response.status != "completed":
            raise RuntimeError(
                f"VLM не завершила генерацию: {response.status}"
            )

        answer = response.output_text.strip()

        if not answer:
            raise RuntimeError(
                "VLM вернула пустой текстовый ответ"
            )

        return answer

    def synthesize_speech(self, text):
        """
        TTS: текст -> WAV-файл.
        """
        with torch.inference_mode():
            audio = self.silero_model.apply_tts(
                text=text,
                speaker=SILERO_VOICE,
                sample_rate=SAMPLE_RATE,
            )

        # Преобразуем сигнал в PCM 16-bit
        pcm = (
            audio.detach()
            .cpu()
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .numpy()
        )

        # Уникальный путь на каждый запрос: прежние записи не перезаписываются.
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"omni_answer_{uuid4().hex}.wav"

        # Режим xb атомарно создаёт новый файл и запрещает перезапись.
        # При ошибке записи удаляем только недописанный новый файл.
        with output_path.open("xb") as output_file:
            try:
                with wave.open(output_file, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(SAMPLE_RATE)
                    wav.writeframes(pcm.tobytes())
            except Exception:
                output_path.unlink(missing_ok=True)
                raise

        return output_path

    def run_omni_assistant(self, image, audio_query):
        """
        Полный пайплайн:
        аудио -> текст -> VLM -> голос.
        """
        image = Path(image)
        audio_query = Path(audio_query)

        if not image.is_file():
            raise FileNotFoundError(image)

        if not audio_query.is_file():
            raise FileNotFoundError(audio_query)

        timings = {}
        total_start = time.perf_counter()

        # Этап 1: ASR
        print("1. Распознаём голос...")

        start = time.perf_counter()
        question = self.recognize_speech(audio_query)
        timings["ASR"] = time.perf_counter() - start

        if not question:
            raise ValueError(
                "Whisper не распознал речь"
            )

        print(f"Вопрос: {question}")

        # Этап 2: VLM
        print("\n2. Анализируем изображение...")

        start = time.perf_counter()
        answer = self.analyze_image(image, question)
        timings["VLM"] = time.perf_counter() - start

        print(f"Ответ: {answer}")

        # Этап 3: TTS
        print("\n3. Синтезируем речь...")

        start = time.perf_counter()
        audio_output = self.synthesize_speech(answer)
        timings["TTS"] = time.perf_counter() - start

        timings["TOTAL"] = (
            time.perf_counter() - total_start
        )

        return {
            "question": question,
            "answer": answer,
            "audio_path": str(audio_output),
            "timings": timings,
        }


if __name__ == "__main__":
    assistant = OmniAssistant()

    result = assistant.run_omni_assistant(
        image=BASE_DIR / "samples/images/fridge.png",
        audio_query=BASE_DIR / "samples/audio/fridge_question.ogg",
    )

    print("\n========== РЕЗУЛЬТАТ ==========")
    print("Вопрос:", result["question"])
    print("Ответ:", result["answer"])
    print("Аудиофайл:", result["audio_path"])

    print("\nЗадержки:")
    for name, seconds in result["timings"].items():
        print(f"  {name}: {seconds:.2f} сек.")
