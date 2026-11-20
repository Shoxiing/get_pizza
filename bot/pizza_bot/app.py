"""Сборка зависимостей и запуск бота."""

import logging

from aiogram import Bot, Dispatcher

from pizza_bot.asr import SpeechRecognizer
from pizza_bot.catalog import Catalog, Example, load_examples, load_prices
from pizza_bot.config import Settings
from pizza_bot.llm.chain import OrderChain
from pizza_bot.retriever import ExampleRetriever, build_embeddings, build_vectorstore
from pizza_bot.telegram.handlers import build_router

log = logging.getLogger(__name__)


def build_order_chain(
    settings: Settings,
    examples: list[Example] | None = None,
    catalog: Catalog | None = None,
) -> OrderChain:
    """examples — что индексировать (по умолчанию весь датасет), catalog — меню."""
    if examples is None or catalog is None:
        all_examples, full_catalog = load_examples(settings.data_path)
        examples = all_examples if examples is None else examples
        catalog = full_catalog if catalog is None else catalog
    embeddings = build_embeddings(settings.embedding_model, settings.embedding_device)
    retriever = ExampleRetriever(build_vectorstore(examples, embeddings), k=settings.retriever_k)
    return OrderChain(settings, retriever, catalog)


async def run(settings: Settings) -> None:
    if settings.telegram_bot_api_key is None:
        raise RuntimeError("TELEGRAM_BOT_API_KEY не задан")

    chain = build_order_chain(settings)
    log.info("Каталог: %d товаров", len(chain.catalog))
    asr = SpeechRecognizer(settings.asr_model, settings.asr_device)
    prices = load_prices(settings.prices_path)

    bot = Bot(token=settings.telegram_bot_api_key.get_secret_value())
    dp = Dispatcher()
    dp.include_router(build_router(chain, asr, prices, settings.request_timeout_s))
    log.info("Bot started, LLM=%s @ %s", settings.llm_model, settings.llm_base_url)
    await dp.start_polling(bot)
