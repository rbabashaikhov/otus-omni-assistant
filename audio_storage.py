
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def create_audio_path(
    output_dir: Path | None = None,
) -> Path:
    """Создаёт уникальный путь для нового WAV-файла."""

    directory = output_dir or OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S_%f")

    identifier = uuid4().hex[:8]

    return directory / (
        f"answer_{timestamp}_{identifier}.wav"
    )
