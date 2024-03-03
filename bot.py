import asyncio
import logging
import aiogram
from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
TOKEN = " "

logging.basicConfig(level=logging.INFO)
bot = Bot(token=TOKEN)
dp = Dispatcher(bot=bot)

@dp.message(Command("start"))
async def start_handler(message: types.Message):
    user_name = message.from_user.full_name
    await message.reply(f"Привет, {user_name}!")

@dp.message(Command("place"))
async def place(message: types.Message):
    kb = types.ReplyKeyBoardMarkup(resize_keyboard=True)
    kb.add(types.KeyBoardButton(text="Онлайн"))
    kb.add(types.KeyBoardButton(text="В офисе"))
    await message.answer("Онлайн/оффлайн", reply_markup=kb)


@dp.message(Command("exp"))
async def exp(message: types.Message):
    kb = types.ReplyBoardMarkup(resize_keyboard=True)
    kb.add(types.KeyBoardButton(text="Есть опыт работы"))
    kb.add(types.KeyBoardButton(text="Без опыта работы"))
    await message.answer("Опыт", reply_markup=kb)


@dp.message(Command("Activity"))
async def activity(message: types.Message):
    kb = types.ReplyBoardMarkup(resize_keyboard=True)
    kb.add(types.KeyBoardButton(text="IT"))
    kb.add(types.KeyBoardButton(text="Медицина"))
    kb.add(types.KeyBoardButton(text="Преподавание"))
    kb.add(types.KeyBoardButton(text="Дизайн"))
    kb.add(types.KeyBoardButton(types.KeyBoardButton(text="Экономика")))
    await message.answer("Сфера деятельности", reply_markup=kb)
async def main():
    bot = Bot(token="TOKEN")
    dp = Dispatcher()

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
