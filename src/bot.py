import asyncio
import os
import sys
import logging

from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import (
    Message,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    WebAppInfo,
)
from dotenv import load_dotenv

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Загружаем .env (для локального запуска)
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBAPP_URL = os.getenv("WEBAPP_URL")

# Проверка: если переменных нет, бот не запустится и скажет почему
if not BOT_TOKEN:
    logging.error("❌ BOT_TOKEN is missing in environment variables!")
    sys.exit(1)

if not WEBAPP_URL:
    logging.error("❌ WEBAPP_URL is missing in environment variables!")
    sys.exit(1)

dp = Dispatcher()


def main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Открыть WriteYourMath",
                    web_app=WebAppInfo(url=WEBAPP_URL),
                )
            ]
        ]
    )


@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    text = (
        "Привет! Это ReadMyMath — тренажёр математического почерка.\n\n"
        "Как это работает:\n"
        "1. Открываешь WriteYourMath — холст для рисования.\n"
        "2. Пишешь формулу от руки или обводишь символы в режиме тренировки.\n"
        "3. ReadMyMath распознаёт её в LaTeX или оценивает точность.\n\n"
        "Начни с простого: x + 1."
    )
    await message.answer(text, reply_markup=main_keyboard())


async def main() -> None:
    logging.info(f"🚀 Starting bot with URL: {WEBAPP_URL}")
    bot = Bot(token=BOT_TOKEN)
    try:
        await dp.start_polling(bot)
    except Exception as e:
        logging.error(f"Bot crashed: {e}")
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Bot stopped.")