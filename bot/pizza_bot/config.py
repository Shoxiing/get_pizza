from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[1]  # bot/
DATA_DIR = ROOT_DIR / "data"


class Settings(BaseSettings):
    """Все настройки берутся из переменных окружения / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_api_key: SecretStr | None = None

    # LLM: любой OpenAI-совместимый endpoint (локальный vLLM, HF router, OpenRouter, ...)
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: SecretStr = SecretStr("EMPTY")
    llm_model: str = "Qwen/Qwen3-14B-AWQ"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512
    # json_schema — constrained decoding (vLLM / большинство провайдеров);
    # function_calling — если провайдер не умеет response_format
    llm_structured_method: str = "json_schema"
    # Зашить список товаров в JSON-схему (enum). С vLLM это гарантирует,
    # что модель не выдумает несуществующий товар.
    llm_strict_catalog: bool = True
    # Qwen3 (гибридные модели) — отключить "thinking" через chat template
    llm_disable_thinking: bool = True

    # RAG
    data_path: Path = DATA_DIR / "formatted_data.json"
    prices_path: Path = DATA_DIR / "prices.json"  # опционально: {"Название": цена}
    embedding_model: str = "sergeyzh/LaBSE-ru-turbo"
    embedding_device: str = "cpu"
    retriever_k: int = 12

    # ASR
    asr_model: str = "antony66/whisper-large-v3-russian"
    asr_device: str | None = None  # None -> auto (cuda / mps / cpu)

    request_timeout_s: float = 60.0


settings = Settings()
