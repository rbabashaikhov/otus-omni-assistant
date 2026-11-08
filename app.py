
import gradio as gr

from pipeline import OmniAssistant


# Загружаем модели один раз при запуске приложения.
print("Запускаем Omni Assistant...")
assistant = OmniAssistant()


def process_request(image_path, audio_path):
    """
    Получает файлы из интерфейса Gradio,
    запускает пайплайн и возвращает результаты.
    """

    if not image_path:
        raise gr.Error("Сначала загрузи фотографию.")

    if not audio_path:
        raise gr.Error(
            "Запиши голосовой вопрос или загрузи аудиофайл."
        )

    try:
        result = assistant.run_omni_assistant(
            image=image_path,
            audio_query=audio_path,
        )
    except Exception as exc:
        # Подробная ошибка остаётся в терминале,
        # чтобы не показывать внутренние данные в UI.
        print(
            f"Ошибка обработки: "
            f"{type(exc).__name__}: {exc}"
        )
        raise gr.Error(
            "Не удалось обработать запрос. "
            "Подробности смотри в терминале VS Code."
        ) from None

    timings = result["timings"]

    latency_text = (
        "### Время обработки\n\n"
        "| Этап | Время |\n"
        "|---|---:|\n"
        f"| Whisper ASR | {timings['ASR']:.2f} с |\n"
        f"| Qwen VLM | {timings['VLM']:.2f} с |\n"
        f"| Silero TTS | {timings['TTS']:.2f} с |\n"
        f"| **Итого** | **{timings['TOTAL']:.2f} с** |"
    )

    return (
        result["question"],
        result["answer"],
        result["audio_path"],
        latency_text,
    )


with gr.Blocks(title="Omni AI Assistant") as demo:

    gr.Markdown(
        """
        # 🎙️ Omni AI Assistant

        ### Мультимодальный ассистент: голос + изображение

        Загрузите фотографию, задайте вопрос голосом
        и получите текстовый и озвученный ответ.

        **Технологии:** Whisper ASR · Qwen VLM · Silero TTS
        """
    )

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                label="📷 Изображение",
                type="filepath",
                sources=["upload"],
            )

            audio_input = gr.Audio(
                label="🎤 Голосовой вопрос",
                sources=["microphone", "upload"],
                type="filepath",
            )

            submit_button = gr.Button(
                "Спросить ассистента",
                variant="primary",
            )

        with gr.Column():

            transcription_output = gr.Textbox(
                label="📝 Распознанный вопрос",
                lines=3,
                interactive=False,
            )

            answer_output = gr.Textbox(
                label="💬 Ответ ассистента",
                lines=5,
                interactive=False,
            )

            audio_output = gr.Audio(
                label="🔊 Голосовой ответ",
                type="filepath",
                interactive=False,
            )

            latency_output = gr.Markdown(
                value="Время обработки появится после запроса."
            )

    submit_button.click(
        fn=process_request,
        inputs=[image_input, audio_input],
        outputs=[
            transcription_output,
            answer_output,
            audio_output,
            latency_output,
        ],
        concurrency_limit=1,
        show_progress="full",
    )

    gr.Markdown(
        """
        ---
        **OTUS · Audio AI · Мультимодальный Omni-ассистент**

        Локальные ASR/TTS · Облачная VLM через Yandex AI Studio
        """
    )


if __name__ == "__main__":
    demo.queue(
        default_concurrency_limit=1,
        max_size=5,
    )

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=False,
        inbrowser=True,
    )
