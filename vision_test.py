
import base64
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

folder = os.environ["YANDEX_CLOUD_FOLDER"]
api_key = os.environ["YANDEX_CLOUD_API_KEY"]
model_name = os.environ["YANDEX_CLOUD_MODEL"]

image_path = Path("samples/images/fridge.png")

if not image_path.is_file():
    raise FileNotFoundError(f"Нет изображения: {image_path}")

# Преобразуем бинарный файл в текст Base64.
# Data URL указывает API, что передаём PNG.
image_base64 = base64.b64encode(
    image_path.read_bytes()
).decode("ascii")

image_url = f"data:image/png;base64,{image_base64}"

client = OpenAI(
    api_key=api_key,
    base_url="https://ai.api.cloud.yandex.net/v1",
    project=folder,
    timeout=90.0,
)

question = (
    "Что изображено на фотографии? "
    "Перечисли основные продукты. "
    "Ответь на русском языке двумя предложениями."
)

start = time.perf_counter()

response = client.responses.create(
    model=f"gpt://{folder}/{model_name}",
    instructions=(
        "Ты визуальный ассистент. "
        "Описывай только то, что действительно видно. "
        "Не придумывай отсутствующие предметы."
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
    max_output_tokens=1024,
    store=False,
)

elapsed = time.perf_counter() - start

print("\n--- VLM-001 ---")
print("Ответ:", response.output_text)
print(f"Latency: {elapsed:.2f} сек.")
