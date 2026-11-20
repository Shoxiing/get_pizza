import asyncio
import io
import logging

from aiogram import Bot, F, Router
from aiogram.filters import CommandStart
from aiogram.types import Message

from pizza_bot.asr import SpeechRecognizer
from pizza_bot.cart import render_cart
from pizza_bot.llm.chain import OrderChain
from pizza_bot.telegram import texts

log = logging.getLogger(__name__)


def build_router(
    chain: OrderChain,
    asr: SpeechRecognizer,
    prices: dict[str, float],
    timeout_s: float,
) -> Router:
    router = Router()

    async def reply_with_cart(message: Message, text: str) -> None:
        log.info("user=%s query=%r", message.from_user.id if message.from_user else None, text)
        try:
            cart = await asyncio.wait_for(chain.ainvoke(text), timeout=timeout_s)
        except asyncio.TimeoutError:
            await message.answer(texts.TIMEOUT)
            return
        except Exception:
            log.exception("order chain failed")
            await message.answer(texts.ERROR)
            return
        await message.answer(render_cart(cart, prices))

    @router.message(CommandStart())
    async def on_start(message: Message) -> None:
        await message.answer(texts.WELCOME)

    @router.message(F.voice)
    async def on_voice(message: Message, bot: Bot) -> None:
        await message.answer(texts.ACCEPTED)
        buf = io.BytesIO()
        await bot.download(message.voice, destination=buf)
        try:
            # Whisper блокирующий — уводим в поток, чтобы не вешать event loop
            text = await asyncio.to_thread(asr.transcribe, buf.getvalue())
        except Exception:
            log.exception("ASR failed")
            await message.answer(texts.ERROR)
            return
        if not text:
            await message.answer(texts.VOICE_EMPTY)
            return
        await message.answer(texts.RECOGNIZED.format(text=text))
        await reply_with_cart(message, text)

    @router.message(F.text)
    async def on_text(message: Message) -> None:
        await message.answer(texts.ACCEPTED)
        await reply_with_cart(message, message.text)

    return router
