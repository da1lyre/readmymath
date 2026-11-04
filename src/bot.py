import asyncio
import os
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import (
    Message, CallbackQuery,
    InlineKeyboardMarkup, InlineKeyboardButton,
)
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / "infra" / ".env"
load_dotenv(dotenv_path=ENV_PATH)

BOT_TOKEN = os.getenv("BOT_TOKEN")

dp = Dispatcher()


def main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="WriteYourMath (скоро)", callback_data="soon")]
        ]
    )


@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    text = (
        "Привет! Это ReadMyMath — тренажёр математического почерка.\n\n"
        "Как это работает:\n"
        "1. Открываешь WriteYourMath — холст для рисования.\n"
        "2. Пишешь формулу от руки: x + 1, дробь, корень.\n"
        "3. ReadMyMath распознаёт её в LaTeX и говорит, читается ли почерк.\n"
        "4. Возвращаешься, прокачиваешь стрик.\n\n"
        "Начни с простого: x + 1."
    )
    await message.answer(text, reply_markup=main_keyboard())


@dp.callback_query(F.data == "soon")
async def on_soon(callback: CallbackQuery) -> None:
    await callback.answer("Холст появится после Этапа 2", show_alert=True)


async def main() -> None:
    bot = Bot(token=BOT_TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())