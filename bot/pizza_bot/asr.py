"""Распознавание голосовых сообщений (Whisper)."""

import io

import numpy as np
import torch
from pydub import AudioSegment
from transformers import pipeline

SAMPLE_RATE = 16_000


def pick_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SpeechRecognizer:
    def __init__(self, model_name: str, device: str | None = None):
        device = pick_device(device)
        dtype = torch.float32 if device == "cpu" else torch.float16
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            dtype=dtype,
            device=device,
            chunk_length_s=30,
        )

    @staticmethod
    def _decode_ogg(data: bytes) -> np.ndarray:
        audio = AudioSegment.from_file(io.BytesIO(data), format="ogg")
        audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
        return np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0

    def transcribe(self, ogg_bytes: bytes) -> str:
        """Блокирующий вызов — запускать через asyncio.to_thread."""
        audio = self._decode_ogg(ogg_bytes)
        result = self.pipe(
            {"raw": audio, "sampling_rate": SAMPLE_RATE},
            generate_kwargs={"language": "russian", "task": "transcribe", "max_new_tokens": 256},
        )
        return result["text"].strip()
