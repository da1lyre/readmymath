import asyncio
import logging
import os
import sys

from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import (
    BufferedInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
    WebAppInfo,
)
from aiohttp import web
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBAPP_URL = os.getenv("WEBAPP_URL")

if not BOT_TOKEN or not WEBAPP_URL:
    sys.exit(1)

dp = Dispatcher()
bot = Bot(token=BOT_TOKEN)


def main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="Открыть WriteYourMath", web_app=WebAppInfo(url=WEBAPP_URL))
    ]])


@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    text = (
        "👋 Добро пожаловать в ReadMyMath!\n\n"
        "Это твой личный тренажёр математического почерка. "
        "Рисуй формулы, тренируй написание символов и получай мгновенную обратную связь.\n\n"
        "👇 Нажми кнопку ниже, чтобы начать."
    )
    await message.answer(text, reply_markup=main_keyboard())

async def handle_send_photo(request: web.Request):
    try:
        data = await request.post()
        user_id = data.get("user_id")
        photo_file = data.get("photo")

        if not user_id or not photo_file:
            return web.json_response({"error": "Missing data"}, status=400)

        photo_bytes = photo_file.file.read()

        await bot.send_photo(
            chat_id=user_id,
            photo=BufferedInputFile(photo_bytes, filename="formula.png"),
            caption="Your formule 📐"
        )
        return web.json_response({"status": "ok"})
    except Exception as e:
        logging.error(f"Error sending photo: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def start_web_server():
    app = web.Application()
    app.router.add_post('/send_photo', handle_send_photo)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8081)
    await site.start()
    logging.info("Web server for bot started on port 8081")


async def main():
    await start_web_server()
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())