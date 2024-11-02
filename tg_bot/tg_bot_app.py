import asyncio
import os
import io
from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import Message, Voice
from aiogram.exceptions import TelegramBadRequest
from dotenv import load_dotenv
from pydub import AudioSegment
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from chain_setup import create_prompt, create_chain
from prepare_rag import llm, retriever
from aiogram.filters import Command

load_dotenv() 


# Токен телеграм бота и huggingface
BOT_TOKEN = os.getenv("TELEGRAM_BOT_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")



router: Router = Router()

torch_dtype = torch.bfloat16 


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
    setattr(torch.distributed, "is_initialized", lambda : False) 
device = torch.device(device)

print(device)# для вывода в лог

#Инициализация модели speech recognition
processor = WhisperProcessor.from_pretrained("antony66/whisper-large-v3-russian")
whisper = WhisperForConditionalGeneration.from_pretrained(
        "antony66/whisper-large-v3-russian", torch_dtype=torch_dtype,
        low_cpu_mem_usage=True, use_safetensors=True)

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=whisper,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=256,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device)

#Приветственное сообщение
@router.message(Command("start"))
async def send_welcome(message: Message):
    welcome_message = """Добро пожаловать! Я AI-бот по заказу пиццы 🍕.
    Собираю корзину как по голосовому, так и текстовому сообщению. Что бы Вы хотели заказать ?"""
    await message.reply(welcome_message)


#Функция для сохранения голосового сообщения в формате wav
async def save_voice_as_wav(bot: Bot, voice: Voice) -> str:
    """Скачивает голосовое сообщение и сохраняет в формате wav."""

    voice_files_dir = "/tg_bot/voice_files"
    os.makedirs(voice_files_dir, exist_ok=True)

    voice_file_info = await bot.get_file(voice.file_id)
    voice_ogg = io.BytesIO()
    await bot.download_file(voice_file_info.file_path, voice_ogg)
    
    voice_wav_path = f"/tg_bot/voice_files/voice-{voice.file_unique_id}.wav"
    AudioSegment.from_file(voice_ogg, format="ogg").export(
        voice_wav_path, format="wav")

    return voice_wav_path

#Функция формирования промпта и отправка в llm модель, получение заказа
async def get_answer(transcription):

    prompt = create_prompt()
    chain = create_chain(retriever, prompt, llm)
    answer = chain.invoke(transcription)

    # Удаляем первую строку, если она содержит в начале "-" и Assistant
    answer_fm = "\n".join(line for line in answer.splitlines() if not set(line) == {"-"})
    answer_fm = answer_fm.replace("Assistant: ", "", 1).strip()

    return answer_fm




# Функция преобразования голосового сообщения в транскрибаццию
async def handle_voice_message(bot: Bot, message: Message):
    voice_path = await save_voice_as_wav(bot, message.voice)
    asr = asr_pipeline(
        voice_path,
        generate_kwargs={"language": "russian", "max_new_tokens": 256},
        return_timestamps=False
    )
    transcription = asr['text']
    ans = await get_answer(transcription)
    return ans


# Хендлер голосовых сообщений
@router.message(F.content_type == "voice")
async def process_message(message: Message, bot: Bot):
    try:
        await message.answer(text="Сообщение принято! Пожалуйста подождите ...")

        # Устанавливаем общий таймаут в 15 секунд на выполнение транскрипции и получения ответа
        ans = await asyncio.wait_for(handle_voice_message(bot, message), timeout=15)
        await message.answer(ans)

    except asyncio.TimeoutError:
        await message.answer("Превышено время ожидания. Попробуйте еще раз или обратитесь в поддержку.")
    except TelegramBadRequest as e:
        print(f"Ошибка при отправке сообщения: {e}")
        await message.answer("Произошла ошибка при отправке сообщения. Попробуйте еще раз или обратитесь в поддержку.")

async def handle_voice_message(bot: Bot, message: Message):
    voice_path = await save_voice_as_wav(bot, message.voice)
    asr = asr_pipeline(
        voice_path,
        generate_kwargs={"language": "russian", "max_new_tokens": 256},
        return_timestamps=False
    )
    transcription = asr['text']
    ans = await get_answer(transcription)
    return ans




# Хендлер текстовых сообщений
@router.message(F.text)
async def process_message(message: Message, bot: Bot):
    try:
        await message.answer(text="Сообщение принято! Пожалуйста подождите ...")

        # Устанавливаем общий таймаут в 15 секунд на выполнение функции get_answer
        ans = await asyncio.wait_for(get_answer(message.text), timeout=15)
        await message.answer(ans)

    except asyncio.TimeoutError:
        await message.answer("Превышено время ожидания. Попробуйте еще раз или обратитесь в поддержку.")
    except TelegramBadRequest as e:
        print(f"Ошибка при отправке сообщения: {e}")
        await message.answer("Произошла ошибка при отправке сообщения. Попробуйте еще раз или обратитесь в поддержку.")



async def main():
    bot: Bot = Bot(token=BOT_TOKEN)
    dp: Dispatcher = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())